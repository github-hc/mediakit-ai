from fastapi import APIRouter, UploadFile, File
from app.schemas.image_schemas import ImageGenerationRequest
from app.controllers.image_controller import ImageController

router = APIRouter(prefix="/images", tags=["images"])

# Instantiate the controller
controller = ImageController()

@router.post(
    "/generate", 
    summary="Generate an image", 
    description="Generate an image from a prompt using the configured backend provider.",
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Returns the generated image file in PNG format."
        }
    }
)
async def generate_image(request: ImageGenerationRequest):
    return await controller.generate_image(request)

@router.post(
    "/ocr",
    summary="Perform OCR on an image",
    description="Perform OCR on the uploaded image using the configured Ollama deepseek-ocr model."
)
async def get_image_ocr(file: UploadFile = File(...)):
    return await controller.get_image_ocr(file)
