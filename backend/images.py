import base64
from io import BytesIO

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

MAX_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 40_000_000
COLORS = ("#64D9C2", "#78A7FF", "#F5B95F", "#C995FF", "#FF8A74")


def read_image(file: UploadFile) -> Image.Image:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(415, "upload an image file")
    raw = file.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "image file is too large")
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.width * image.height > MAX_PIXELS:
                raise HTTPException(413, "image dimensions are too large")
            return image.convert("RGB")
    except Image.DecompressionBombError as error:
        raise HTTPException(413, "image dimensions are too large") from error
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(400, "image could not be decoded") from error


def mask_uri(roi, color):
    value = np.asarray(roi, dtype=bool)
    rgba = np.zeros((*value.shape, 4), dtype=np.uint8)
    rgb = tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
    rgba[..., :3] = rgb
    rgba[..., 3] = value * 132
    buffer = BytesIO()
    Image.fromarray(rgba).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def pack_objects(objects):
    out = []
    for index, item in enumerate(objects):
        color = COLORS[index % len(COLORS)]
        out.append(
            {
                "object_id": item["object_id"],
                "box": list(item["box"]),
                "mask": mask_uri(item["roi"], color),
                "color": color,
                "metrics": item["metrics"],
            }
        )
    return out
