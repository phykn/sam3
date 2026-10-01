import numpy as np


def find_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def extract_roi(value: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    mask = np.asarray(value, dtype=bool)
    box = find_box(mask)
    if box is None:
        return (0, 0, 0, 0), np.zeros((0, 0), dtype=np.uint8)
    x0, y0, x1, y1 = box
    return box, mask[y0:y1, x0:x1].astype(np.uint8)


def restore_mask(
    shape: tuple[int, ...] | list[int],
    box: tuple[int, int, int, int],
    roi: np.ndarray,
) -> np.ndarray:
    out = np.zeros(tuple(shape)[:2], dtype=bool)
    x0, y0, x1, y1 = box
    out[y0:y1, x0:x1] = np.asarray(roi, dtype=bool)
    return out
