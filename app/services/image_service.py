from app.core.config import settings
from app.providers.registry import provider_registry
from app.schemas.image_schemas import ImageGenerationRequest

class ImageService:
    def __init__(self):
        pass

    def _get_provider(self):
        """
        Resolve and instantiate the configured image provider.
        """
        provider_name = settings.IMAGE_PROVIDER
        provider_cls = provider_registry.get(provider_name)
        
        # Instantiate with specific configs depending on provider type
        if provider_name.lower() == "ollama":
            return provider_cls(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL
            )
        
        # Default fallback for other registered providers (e.g., diffusers)
        return provider_cls()

    async def initialize(self) -> None:
        """
        Trigger startup initialization hooks for the active provider.
        """
        provider = self._get_provider()
        if hasattr(provider, "initialize"):
            await provider.initialize()

    async def generate_image(self, request: ImageGenerationRequest) -> bytes:
        """
        Generate image by routing the request to the configured provider.
        """
        provider = self._get_provider()
        
        # Prepare arguments (filtering out None to use provider default arguments)
        kwargs = {
            "width": request.width,
            "height": request.height,
            "seed": request.seed,
            "num_inference_steps": request.num_inference_steps,
            "negative_prompt": request.negative_prompt,
            "guidance_scale": request.guidance_scale
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        return await provider.generate_image(
            prompt=request.prompt,
            **filtered_kwargs
        )

    async def ocr_image(self, image_bytes: bytes) -> str:
        """
        Perform OCR on image bytes using Ollama's deepseek-ocr model.
        """
        import base64
        import httpx
        from app.core.exceptions import ProviderError

        base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        ocr_model = settings.OLLAMA_OCR_MODEL

        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        # Try configured model first, then fallback to :latest if no tag was provided.
        model_candidates = [ocr_model]
        if ":" not in ocr_model:
            model_candidates.append(f"{ocr_model}:latest")

        url = f"{base_url}/api/generate"
        last_error = None

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                for model in model_candidates:
                    payload = {
                        "model": model,
                        "prompt": "Recognize the text in this image.",
                        "images": [encoded_image],
                        "stream": False
                    }

                    try:
                        response = await client.post(url, json=payload)
                        if response.status_code == 200:
                            result = response.json()
                            return result.get("response", "").strip()

                        last_error = ProviderError(
                            "ollama",
                            f"Ollama server returned status code {response.status_code}: {response.text}"
                        )
                        continue
                    except httpx.RequestError as e:
                        last_error = ProviderError("ollama", f"Connection to Ollama server failed: {str(e)}")
                        continue

            if last_error is not None:
                raise last_error
            raise ProviderError("ollama", "Unexpected error occurred during OCR: no response from Ollama server")
        except Exception as e:
            if isinstance(e, ProviderError):
                raise e
            raise ProviderError("ollama", f"Unexpected error occurred during OCR: {str(e)}")
