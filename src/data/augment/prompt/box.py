import numpy as np

from ....ops.mask import find_box


def jitter_mask_box(
    target: np.ndarray,
    image_shape: tuple[int, ...],
    amount: float = 0.1,
) -> np.ndarray:
    box = find_box(target > 0)
    if box is None:
        raise ValueError("target mask must contain foreground")
    base = np.asarray(box, dtype=np.float32)
    if amount <= 0:
        return base

    height, width = tuple(image_shape)[:2]
    x0, y0, x1, y1 = base
    dx = (x1 - x0) * float(amount)
    dy = (y1 - y0) * float(amount)
    out = np.array(
        [
            x0 + np.random.uniform(-dx, dx),
            y0 + np.random.uniform(-dy, dy),
            x1 + np.random.uniform(-dx, dx),
            y1 + np.random.uniform(-dy, dy),
        ],
        dtype=np.float32,
    )
    out[[0, 2]] = np.clip(out[[0, 2]], 0, width)
    out[[1, 3]] = np.clip(out[[1, 3]], 0, height)

    if out[2] <= out[0] or out[3] <= out[1]:
        return base
    return out
