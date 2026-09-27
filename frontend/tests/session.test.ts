import assert from 'node:assert/strict';
import { test } from 'node:test';
import { MatchSession, type SessionApi } from '../src/session.ts';
import type { SessionResult } from '../src/types.ts';

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

const result = (id = 'one'): SessionResult => ({
  session_id: id,
  width: 100,
  height: 80,
  prompt_count: 0,
  objects: [],
});
const file = (name = 'one.png') => new File(['image'], name);

function api(overrides: Partial<SessionApi> = {}): SessionApi {
  return {
    createSession: async () => result(),
    addPoint: async () => result(),
    addBox: async () => result(),
    updatePoint: async () => result(),
    updateBox: async () => result(),
    deletePoints: async () => result(),
    excludeObject: async () => result(),
    refineResults: async () => result(),
    ...overrides,
  };
}

test('latest upload owns the session even when an older upload finishes later', async () => {
  const first = deferred<SessionResult>();
  const second = deferred<SessionResult>();
  const session = new MatchSession(
    api({
      createSession: (file) =>
        file.name === 'one.png' ? first.promise : second.promise,
    }),
  );
  const a = session.openFile(file());
  const b = session.openFile(file('two.png'));
  second.resolve(result('two'));
  await b;
  first.resolve(result('one'));
  await a;
  assert.equal(session.getSnapshot().sessionId, 'two');
  assert.equal(session.getSnapshot().file?.name, 'two.png');
});

test('stale preview sizes and upload errors cannot replace the current image', async () => {
  const old = deferred<SessionResult>();
  const session = new MatchSession(
    api({
      createSession: (file) =>
        file.name === 'one.png' ? old.promise : Promise.resolve(result('two')),
    }),
  );
  const first = file();
  const pending = session.openFile(first);
  await session.openFile(file('two.png'));
  session.setImageSize(first, { width: 4, height: 4 });
  old.reject(new Error('old upload failed'));
  await pending;
  assert.deepEqual(session.getSnapshot().imageSize, { width: 100, height: 80 });
  assert.equal(session.getSnapshot().error, null);
});

test('consecutive events cannot dispatch two mutations before a UI render', async () => {
  const reply = deferred<SessionResult>();
  let calls = 0;
  const session = new MatchSession(
    api({
      addPoint: () => {
        calls += 1;
        return reply.promise;
      },
    }),
  );
  await session.openFile(file());
  const first = session.submitPoint([10, 20]);
  const second = session.submitPoint([20, 30]);
  assert.equal(calls, 1);
  assert.equal(session.getSnapshot().phase, 'inferencing');
  reply.resolve(result());
  await Promise.all([first, second]);
  assert.deepEqual(session.getSnapshot().prompts, [
    { kind: 'point', point: [10, 20], positive: true },
  ]);
});

test('failed mutations preserve prompts and allow retry', async () => {
  let fail = true;
  const session = new MatchSession(
    api({
      addBox: async () => {
        if (fail) throw new Error('bad box');
        return result();
      },
    }),
  );
  await session.openFile(file());
  await session.submitBox([0, 0, 20, 20]);
  assert.deepEqual(session.getSnapshot().prompts, []);
  assert.equal(session.getSnapshot().phase, 'ready');
  assert.equal(session.getSnapshot().error, 'bad box');
  fail = false;
  await session.submitBox([0, 0, 20, 20]);
  assert.equal(session.getSnapshot().prompts.length, 1);
  assert.equal(session.getSnapshot().error, null);
});

test('a late mutation cannot finish a new upload or restore old prompts', async () => {
  const mutation = deferred<SessionResult>();
  const upload = deferred<SessionResult>();
  const session = new MatchSession(
    api({
      createSession: (file) =>
        file.name === 'two.png' ? upload.promise : Promise.resolve(result()),
      addPoint: () => mutation.promise,
    }),
  );
  await session.openFile(file());
  const change = session.submitPoint([10, 20]);
  const next = session.openFile(file('two.png'));
  mutation.resolve(result());
  await change;
  assert.equal(session.getSnapshot().phase, 'uploading');
  assert.deepEqual(session.getSnapshot().prompts, []);
  upload.resolve(result('two'));
  await next;
});

test('point and box edits and mixed selection removal preserve shared indices', async () => {
  let removed: number[] = [];
  const session = new MatchSession(
    api({
      deletePoints: async (_id, indices) => {
        removed = indices;
        return result();
      },
    }),
  );
  await session.openFile(file());
  await session.submitPoint([5, 5]);
  await session.submitBox([1, 1, 20, 20]);
  await session.editPrompt(0, [6, 7]);
  await session.editBox(1, [2, 2, 21, 21]);
  assert.deepEqual(session.getSnapshot().prompts, [
    { kind: 'point', point: [6, 7], positive: true },
    { kind: 'box', box: [2, 2, 21, 21], positive: true },
  ]);
  session.setSelectedPrompts([0, 1]);
  session.setPositive(false);
  await session.removeSelected();
  assert.deepEqual(removed, [0, 1]);
  assert.deepEqual(session.getSnapshot().prompts, []);
  assert.equal(session.getSnapshot().positive, true);
});
