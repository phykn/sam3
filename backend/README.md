# SAM 3 API

Install the Python dependencies described in the root README, then start the
FastAPI server from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Run one API worker: the model and up to eight image sessions live in that
process. The server uses CUDA when available and falls back to CPU otherwise;
CPU inference is slower. The first image upload loads the local checkpoint.

The default files are `weight/sam3.1_multiplex.pt` and `weight/visual_token.pt`,
resolved from the repository root. Override them with `SAM3_WEIGHT` and
`SAM3_VISUAL`; choose a device with `SAM3_DEVICE`. Weights are not downloaded by
the server.

Uploads are limited to 25 MiB and 40 million pixels, with dimensions checked
before pixel decoding. HTTP routes and validation live in `app.py`; `runtime.py`
owns model loading, inference serialization, session eviction, and shutdown
cleanup. `create_app()` creates an independent runtime and can accept an injected
runtime for HTTP integration tests. Upload validation and mask response encoding
live in `images.py`. Model inference
uses the public `src.predict.GroundPredictor` interface. Box and point prompts
share one index sequence within each image session.

`GET /api/health` reports the selected device and whether the model is loaded.
Start the [frontend](../frontend/README.md) to upload images and edit prompts.
