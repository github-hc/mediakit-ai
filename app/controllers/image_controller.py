from fastapi import Response, UploadFile, File, Form
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

    async def ask_image(self, file: UploadFile = File(...), prompt: str = Form(...)) -> dict:
        """
        Controller action to send an image and a user prompt to granite3.2-vision:latest
        and return the model's text response.
        """
        image_bytes = await file.read()
        answer = await self.image_service.ask_image(image_bytes, prompt)
        return {"response": answer}
