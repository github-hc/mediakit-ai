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
