from typing import Optional
from pydantic import BaseModel, Field

class ImageGenerationRequest(BaseModel):
    prompt: str = Field(
        ..., 
        description="The text prompt describing the image to generate.",
        min_length=1
    )
    negative_prompt: Optional[str] = Field(
        default="blurry, bad quality, distorted, ugly, watermark",
        description="What to avoid in the generated image."
    )
    width: Optional[int] = Field(
        default=512, 
        description="Width of the generated image (must be positive).",
        gt=0
    )
    height: Optional[int] = Field(
        default=512, 
        description="Height of the generated image (must be positive).",
        gt=0
    )
    seed: Optional[int] = Field(
        default=None, 
        description="Random seed for generation reproducibility."
    )
    num_inference_steps: Optional[int] = Field(
        default=25, 
        description="Number of denoising/inference steps.",
        gt=0
    )
    guidance_scale: Optional[float] = Field(
        default=7.5,
        description="How closely the model follows the prompt (CFG scale).",
        ge=1.0,
        le=20.0
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "A highly detailed futuristic city at night, cyber-punk style",
                "negative_prompt": "blurry, bad quality, distorted, ugly, watermark",
                "width": 512,
                "height": 512,
                "seed": 42,
                "num_inference_steps": 25,
                "guidance_scale": 7.5
            }
        }
