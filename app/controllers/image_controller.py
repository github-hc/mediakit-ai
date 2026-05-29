from fastapi import Response, UploadFile, File
from app.schemas.image_schemas import ImageGenerationRequest
from app.services.image_service import ImageService

class ImageController:
    def __init__(self):
        self.image_service = ImageService()

    async def generate_image(self, request: ImageGenerationRequest) -> Response:
        """
        Controller action to generate image and return it as a binary stream.
        """
        image_bytes = await self.image_service.generate_image(request)
        return Response(content=image_bytes, media_type="image/png")

    async def get_image_ocr(self, file: UploadFile = File(...)) -> dict:
        """
        Controller action to perform OCR on the uploaded image using Ollama.
        """
        image_bytes = await file.read()
        ocr_text = await self.image_service.ocr_image(image_bytes)
        return {"text": ocr_text}
