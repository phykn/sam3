import type {
  Box,
  ImageSize,
  PromptMark,
  PromptPoint,
  PromptTool,
  ResultObject,
  SessionResult,
} from './types';

export type SessionApi = Pick<
  typeof import('./api'),
  | 'createSession'
  | 'addPoint'
  | 'addBox'
  | 'updatePoint'
  | 'updateBox'
  | 'deletePoints'
  | 'excludeObject'
  | 'refineResults'
>;

export type SessionState = {
  phase: 'idle' | 'uploading' | 'ready' | 'inferencing';
  sessionId: string | null;
  file: File | null;
  imageSize: ImageSize | null;
  prompts: PromptMark[];
  objects: ResultObject[];
  positive: boolean;
  tool: PromptTool;
  selectedPrompts: number[];
  error: string | null;
};

const emptyState = (): SessionState => ({
  phase: 'idle',
  sessionId: null,
  file: null,
  imageSize: null,
  prompts: [],
  objects: [],
  positive: true,
  tool: 'point',
  selectedPrompts: [],
  error: null,
});

export class MatchSession {
  private state = emptyState();
  private version = 0;
  private listeners = new Set<() => void>();
  private api: SessionApi;

  constructor(api: SessionApi) {
    this.api = api;
  }

  getSnapshot = () => this.state;

  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  private patch(update: Partial<SessionState>) {
    this.state = { ...this.state, ...update };
    this.listeners.forEach((listener) => listener());
  }

  cancel = () => {
    this.version += 1;
  };

  openFile = async (file: File) => {
    const version = ++this.version;
    this.patch({ ...emptyState(), file, phase: 'uploading' });
    try {
      const data = await this.api.createSession(file);
      if (version !== this.version) return;
      this.patch({
        sessionId: data.session_id,
        imageSize: { width: data.width, height: data.height },
        phase: 'ready',
      });
    } catch (error) {
      if (version === this.version)
        this.patch({ error: message(error), phase: 'idle' });
    }
  };

  setImageSize = (file: File, size: ImageSize) => {
    if (file === this.state.file && this.state.phase === 'uploading') {
      this.patch({ imageSize: size });
    }
  };

  setPositive = (positive: boolean) => {
    this.patch({ positive });
  };
  setTool = (tool: PromptTool) => {
    this.patch({ tool });
  };
  setSelectedPrompts = (selectedPrompts: number[]) => {
    this.patch({ selectedPrompts });
  };

  private async change(
    request: (id: string) => Promise<SessionResult>,
    update: (state: SessionState) => Partial<SessionState> = () => ({}),
  ) {
    const state = this.state;
    if (!state.sessionId || state.phase !== 'ready') return;
    const version = this.version;
    this.patch({ phase: 'inferencing', error: null });
    try {
      const data = await request(state.sessionId);
      if (version !== this.version) return;
      this.patch({ ...update(state), objects: data.objects, phase: 'ready' });
    } catch (error) {
      if (version === this.version)
        this.patch({ error: message(error), phase: 'ready' });
    }
  }

  submitPoint = (point: PromptPoint) => {
    const positive = this.state.positive;
    return this.change(
      (id) => this.api.addPoint(id, point, positive),
      (state) => ({
        prompts: [...state.prompts, { kind: 'point', point, positive }],
        selectedPrompts: [],
      }),
    );
  };

  submitBox = (box: Box) => {
    const positive = this.state.positive;
    return this.change(
      (id) => this.api.addBox(id, box, positive),
      (state) => ({
        prompts: [...state.prompts, { kind: 'box', box, positive }],
        selectedPrompts: [],
      }),
    );
  };

  editPrompt = (index: number, point: PromptPoint) =>
    this.change(
      (id) => this.api.updatePoint(id, index, point),
      (state) => ({
        prompts: state.prompts.map((item, idx) =>
          idx === index && item.kind === 'point' ? { ...item, point } : item,
        ),
      }),
    );

  editBox = (index: number, box: Box) =>
    this.change(
      (id) => this.api.updateBox(id, index, box),
      (state) => ({
        prompts: state.prompts.map((item, idx) =>
          idx === index && item.kind === 'box' ? { ...item, box } : item,
        ),
      }),
    );

  removeSelected = async () => {
    const indices = this.state.selectedPrompts;
    if (!indices.length) return;
    await this.change(
      (id) => this.api.deletePoints(id, indices),
      (state) => {
        const prompts = state.prompts.filter(
          (_, idx) => !indices.includes(idx),
        );
        return {
          prompts,
          selectedPrompts: [],
          ...(!prompts.length ? { positive: true } : {}),
        };
      },
    );
  };

  excludeResult = (item: ResultObject) =>
    this.change(
      (id) => this.api.excludeObject(id, item.object_id),
      (state) => ({
        prompts: [
          ...state.prompts,
          { kind: 'box', box: item.box, positive: false },
        ],
        selectedPrompts: [],
        positive: false,
      }),
    );

  refine = async () => {
    if (this.state.objects.length)
      await this.change((id) => this.api.refineResults(id));
  };
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : 'Something went wrong.';
}
