import numpy as np
import pytest

from src.data.augment.prompt.box import jitter_mask_box
from src.ops.mask import extract_roi, find_box, restore_mask


@pytest.mark.parametrize(
    "mask, expected",
    [
        (np.array([[1]], dtype=bool), (0, 0, 1, 1)),
        (np.eye(4, 6, dtype=bool), (0, 0, 4, 4)),
        (np.ones((3, 5), dtype=np.uint8) * 255, (0, 0, 5, 3)),
        (np.array([[0, 0, 0], [0, -2, 0]]), (1, 1, 2, 2)),
        (np.array([[0, 1, 0], [1, 0, 1]])[:, ::-1], (0, 0, 3, 2)),
        (np.zeros((3, 5), dtype=bool), None),
        (np.zeros((0, 5), dtype=bool), None),
        (np.zeros((3, 0), dtype=bool), None),
    ],
)
def test_mask_geometry_preserves_foreground_and_exclusive_bounds(mask, expected):
    before = mask.copy()
    assert find_box(mask) == expected
    box, roi = extract_roi(mask)
    assert box == ((0, 0, 0, 0) if expected is None else expected)
    assert roi.dtype == np.uint8
    assert set(np.unique(roi)) <= {0, 1}
    restored = restore_mask((*mask.shape, 3), box, roi)
    assert restored.dtype == np.bool_
    np.testing.assert_array_equal(restored, mask.astype(bool))
    np.testing.assert_array_equal(mask, before)
    roi[:] = 0
    np.testing.assert_array_equal(mask, before)


def test_training_box_uses_positive_pixels_and_keeps_float_coordinates():
    mask = np.zeros((4, 6), dtype=np.float32)
    mask[0, 0] = -1
    mask[1:3, 2:5] = 0.5
    box = jitter_mask_box(mask, (*mask.shape, 3), amount=0)
    assert box.dtype == np.float32
    np.testing.assert_array_equal(box, [2, 1, 5, 3])


def test_training_box_rejects_empty_foreground():
    with pytest.raises(ValueError):
        jitter_mask_box(np.zeros((4, 6), dtype=np.uint8), (4, 6, 3), amount=0)
