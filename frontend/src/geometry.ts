import type {
  Box,
  ImageSize,
  PromptMark,
  PromptPoint,
  PromptTool,
} from './types';

export type Point = { x: number; y: number };
export type BoxMode = 'move' | 'nw' | 'ne' | 'sw' | 'se';
type Gesture =
  | { kind: 'point'; point: PromptPoint }
  | { kind: 'box'; box: Box }
  | { kind: 'select'; indices: number[] };

export function clamp(point: Point, size: ImageSize): Point {
  return {
    x: Math.max(0, Math.min(size.width, point.x)),
    y: Math.max(0, Math.min(size.height, point.y)),
  };
}

export function imagePosition(
  point: Point,
  layout: ImageSize,
  image: ImageSize,
): PromptPoint {
  return [
    (point.x / layout.width) * image.width,
    (point.y / layout.height) * image.height,
  ];
}

export function movePoint(
  point: PromptPoint,
  dx: number,
  dy: number,
  size: ImageSize,
): PromptPoint {
  return [
    Math.max(0, Math.min(size.width - 1, point[0] + dx)),
    Math.max(0, Math.min(size.height - 1, point[1] + dy)),
  ];
}

export function resolveGesture(
  tool: PromptTool,
  start: Point,
  end: Point,
  layout: ImageSize,
  image: ImageSize,
  prompts: PromptMark[],
): Gesture | null {
  if (layout.width <= 0 || layout.height <= 0) return null;
  const first = clamp(start, layout);
  const last = clamp(end, layout);
  const moved =
    Math.abs(last.x - first.x) >= 6 || Math.abs(last.y - first.y) >= 6;
  if (tool === 'point') {
    if (moved) return null;
    const point = movePoint(imagePosition(last, layout, image), 0, 0, image);
    return { kind: 'point', point: point.map(Math.round) as PromptPoint };
  }
  if (tool === 'box') {
    if (!moved) return null;
    const a = imagePosition(first, layout, image).map(Math.round);
    const b = imagePosition(last, layout, image).map(Math.round);
    if (Math.abs(b[0] - a[0]) < 4 || Math.abs(b[1] - a[1]) < 4) return null;
    return {
      kind: 'box',
      box: [
        Math.min(a[0], b[0]),
        Math.min(a[1], b[1]),
        Math.max(a[0], b[0]),
        Math.max(a[1], b[1]),
      ],
    };
  }
  if (!moved) return { kind: 'select', indices: [] };
  const x0 = Math.min(first.x, last.x),
    x1 = Math.max(first.x, last.x);
  const y0 = Math.min(first.y, last.y),
    y1 = Math.max(first.y, last.y);
  const indices = prompts.flatMap((item, idx) => {
    const center =
      item.kind === 'point'
        ? item.point
        : [(item.box[0] + item.box[2]) / 2, (item.box[1] + item.box[3]) / 2];
    const x = (center[0] / image.width) * layout.width;
    const y = (center[1] / image.height) * layout.height;
    return x >= x0 && x <= x1 && y >= y0 && y <= y1 ? [idx] : [];
  });
  return { kind: 'select', indices };
}

export function moveBox(
  box: Box,
  mode: BoxMode,
  dx: number,
  dy: number,
  size: ImageSize,
): Box {
  let [x0, y0, x1, y1] = box;
  if (mode === 'move') {
    const width = x1 - x0,
      height = y1 - y0;
    x0 = Math.max(0, Math.min(size.width - width, x0 + dx));
    y0 = Math.max(0, Math.min(size.height - height, y0 + dy));
    return [x0, y0, x0 + width, y0 + height];
  }
  const minWidth = Math.min(4, size.width),
    minHeight = Math.min(4, size.height);
  if (mode.includes('w')) x0 = Math.max(0, Math.min(x1 - minWidth, x0 + dx));
  if (mode.includes('e'))
    x1 = Math.min(size.width, Math.max(x0 + minWidth, x1 + dx));
  if (mode.includes('n')) y0 = Math.max(0, Math.min(y1 - minHeight, y0 + dy));
  if (mode.includes('s'))
    y1 = Math.min(size.height, Math.max(y0 + minHeight, y1 + dy));
  return [x0, y0, x1, y1];
}
