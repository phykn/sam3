import assert from 'node:assert/strict';
import { test } from 'node:test';
import { moveBox, movePoint, resolveGesture } from '../src/geometry.ts';

const layout = { width: 200, height: 160 };
const image = { width: 100, height: 80 };

test('edge clicks match the backend pixel clamp while boxes retain edge coordinates', () => {
  assert.deepEqual(
    resolveGesture(
      'point',
      { x: 200, y: 160 },
      { x: 200, y: 160 },
      layout,
      image,
      [],
    ),
    { kind: 'point', point: [99, 79] },
  );
  assert.deepEqual(
    resolveGesture(
      'box',
      { x: 200, y: 160 },
      { x: 0, y: 0 },
      layout,
      image,
      [],
    ),
    { kind: 'box', box: [0, 0, 100, 80] },
  );
});

test('small box drags and clicks before layout do not create invalid prompts', () => {
  assert.equal(
    resolveGesture('box', { x: 0, y: 0 }, { x: 5, y: 5 }, layout, image, []),
    null,
  );
  assert.equal(
    resolveGesture(
      'point',
      { x: 0, y: 0 },
      { x: 0, y: 0 },
      { width: 0, height: 0 },
      image,
      [],
    ),
    null,
  );
});

test('selection preserves indices across point and box prompts', () => {
  assert.deepEqual(
    resolveGesture('select', { x: 80, y: 80 }, { x: 0, y: 0 }, layout, image, [
      { kind: 'point', point: [10, 10], positive: true },
      { kind: 'box', box: [20, 20, 30, 30], positive: false },
      { kind: 'point', point: [90, 70], positive: true },
    ]),
    { kind: 'select', indices: [0, 1] },
  );
});

test('moving prompts clamps to the image without changing box dimensions', () => {
  assert.deepEqual(movePoint([10, 10], 100, -50, image), [99, 0]);
  assert.deepEqual(
    moveBox([10, 10, 30, 40], 'move', 100, -50, image),
    [80, 0, 100, 30],
  );
});

test('resizing keeps ordered corners and minimum dimensions', () => {
  assert.deepEqual(
    moveBox([10, 10, 30, 40], 'nw', 100, 100, image),
    [26, 36, 30, 40],
  );
  assert.deepEqual(
    moveBox([10, 10, 30, 40], 'se', 100, 100, image),
    [10, 10, 100, 80],
  );
});
