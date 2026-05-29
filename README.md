# MediaKit AI

MediaKit AI is an extensible, modular MVP for image capabilities (generation, editing, etc.) built with FastAPI. It leverages a pluggable provider architecture to support multiple local or remote AI backends easily (supporting local Stable Diffusion 1.5 via Hugging Face `diffusers` and text-to-image endpoints via `ollama`).

## Architecture

```
mediakit-ai/
├── app/
│   ├── main.py                    # FastAPI app entry point
│   ├── core/
│   │   ├── config.py              # Settings (env vars, provider configs)
│   │   └── exceptions.py          # Custom app exceptions
│   │
│   ├── providers/                 # 🔑 THE HEART — swap AI backends here
│   │   ├── base.py                # Abstract ImageProvider interface
│   │   ├── registry.py            # Provider registry (factory pattern)
│   │   ├── ollama/
│   │   │   ├── __init__.py
│   │   │   └── provider.py        # OllamaProvider(ImageProvider)
│   │   └── diffusers/
│   │       └── provider.py        # DiffusersProvider(ImageProvider) - SD 1.5
│   │
│   ├── controllers/
│   │   └── image_controller.py    # Route handlers (thin layer)
│   │
│   ├── services/
│   │   └── image_service.py       # Business logic, provider orchestration
│   │
│   ├── schemas/
│   │   └── image_schemas.py       # Pydantic request/response models
│   │
│   └── routers/
│       └── image_router.py        # FastAPI router wiring
│
├── tests/
│   └── test_image_generation.py
├── .env.example
├── requirements.txt
└── README.md
```

## Setup & Running

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   Copy the example environment file and configure it:
   ```bash
   cp .env.example .env
   ```

3. **Run the Server**
   Start the FastAPI development server:
   ```bash
   # Run via Python directly
   python app/main.py
   
   # Or run via Uvicorn
   uvicorn app.main:app --reload
   ```

4. **Verify the Endpoint**
   Send a request to generate an image (supports `negative_prompt`, `num_inference_steps`, and `guidance_scale`):
   ```bash
   curl -X POST "http://localhost:8000/api/v1/images/generate" \
        -H "Content-Type: application/json" \
        -d '{"prompt": "A beautiful sunset over the mountains", "negative_prompt": "blurry, low quality", "width": 512, "height": 512, "num_inference_steps": 25, "guidance_scale": 7.5}' \
        --output output.png
   ```

## Providers Configuration

The active provider is configured via environment variables:
- `IMAGE_PROVIDER=diffusers`: Runs Stable Diffusion 1.5 locally (utilizing PyTorch). Optimized for systems with RAM/GPU constraints. Uses device-auto-detection (`cuda`, `mps`, or `cpu`) and attention-slicing.
- `IMAGE_PROVIDER=ollama`: Communicates with a running Ollama server.
- Additional providers can be registered in `app/providers/registry.py` and activated by setting `IMAGE_PROVIDER`.

## Testing

Run the test suite using pytest:
```bash
pytest
```

## Environment Variables

Use `.env.example` as a template — it is safe to check into source control and contains placeholders for sensitive values. Do NOT commit a filled `.env` with real credentials or API tokens.

Important variables:
- `IMAGE_PROVIDER` — `ollama` or `diffusers` (set to `ollama` to enable OCR and Ollama-based image generation).
- `OLLAMA_BASE_URL` — HTTP URL of your local Ollama server (default: `http://localhost:11434`).
- `OLLAMA_MODEL` — default image generation model (e.g. `x/flux2-klein:4b`).
- `OLLAMA_OCR_MODEL` — OCR model name (e.g. `deepseek-ocr:latest`).
- `HF_API_TOKEN` — (optional) Hugging Face API token if using HF features.

To create a local `.env` from the example:
```bash
cp .env.example .env
# edit .env and provide any required tokens or model names
```

## OCR (deepseek-ocr)

The project exposes an OCR endpoint that uses the configured `OLLAMA_OCR_MODEL` via the Ollama HTTP API.

- FastAPI route: `POST /api/v1/images/ocr` — accepts a multipart file upload (`file`).

Example curl request:
```bash
curl -X POST "http://localhost:8000/api/v1/images/ocr" \
   -F "file=@img.png" \
   -H "Accept: application/json"
```

If the server returns a 502 (ProviderError), check that your Ollama server is running and that the OCR model is loaded. Verify locally with:
```bash
# start/load the model (interactive)
ollama run deepseek-ocr:latest

# or list models (if supported by your Ollama version)
ollama list
```

For quick local debugging you can run the helper script `test6.py` in the repository root; it will call the Ollama HTTP API and print the extracted OCR text for `img.png`:
```bash
python test6.py
```

