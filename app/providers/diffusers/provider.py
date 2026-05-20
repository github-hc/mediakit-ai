import io
import logging
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from app.providers.base import ImageProvider
from app.core.exceptions import ProviderError

logger = logging.getLogger(__name__)

class DiffusersProvider(ImageProvider):
    # Class-level caching to prevent loading the model multiple times
    _pipeline: StableDiffusionPipeline | None = None
    _device: str = "cpu"

    def __init__(self):
        pass

    async def initialize(self) -> None:
        """
        Load the Stable Diffusion 1.5 pipeline on the best available device.
        """
        if DiffusersProvider._pipeline is not None:
            return

        logger.info("Initializing Stable Diffusion 1.5 ...")
        
        # 1. Device detection
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # Float16 works well and saves VRAM/RAM on GPU platforms (CUDA, MPS)
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        logger.info(f"Using device: {device}, dtype: {dtype}")

        try:
            # 2. Load model
            pipeline = StableDiffusionPipeline.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5",
                torch_dtype=dtype,
                safety_checker=None,           # Disable safety checker for performance/speed
                requires_safety_checker=False,
            )
            
            # Use faster scheduler (DPM-Solver) to achieve high quality in fewer steps
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                pipeline.scheduler.config
            )
            pipeline = pipeline.to(device)

            # 3. Apply memory/speed optimizations for consumer hardware
            if device in ("cpu", "mps"):
                pipeline.enable_attention_slicing()

            DiffusersProvider._pipeline = pipeline
            DiffusersProvider._device = device
            logger.info("Stable Diffusion 1.5 model successfully loaded.")
            
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion 1.5 model: {str(e)}")
            raise ProviderError("diffusers", f"Model load failure: {str(e)}")

    async def generate_image(self, prompt: str, **kwargs) -> bytes:
        """
        Generate image using the loaded Stable Diffusion 1.5 pipeline.
        """
        # Ensure model is initialized
        if DiffusersProvider._pipeline is None:
            await self.initialize()

        pipeline = DiffusersProvider._pipeline
        device = DiffusersProvider._device

        # Dimensions must be multiples of 8 for Stable Diffusion 1.5
        width = int(kwargs.get("width") or 512)
        height = int(kwargs.get("height") or 512)
        w = (width // 8) * 8
        h = (height // 8) * 8

        steps = int(kwargs.get("num_inference_steps") or 25)
        guidance_scale = float(kwargs.get("guidance_scale") or 7.5)
        negative_prompt = kwargs.get(
            "negative_prompt", 
            "blurry, bad quality, distorted, ugly, watermark"
        )

        logger.info(f"Generating image prompt='{prompt[:50]}...' size={w}x{h} steps={steps}")

        try:
            # Set random seed if provided
            generator = None
            seed = kwargs.get("seed")
            if seed is not None:
                generator = torch.Generator(device=device).manual_seed(seed)

            # Perform inference
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=w,
                height=h,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

            image = result.images[0]

            # Save PNG image to binary buffer
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return buf.getvalue()

        except Exception as e:
            logger.error(f"Error during Stable Diffusion inference: {str(e)}")
            raise ProviderError("diffusers", f"Inference failure: {str(e)}")
