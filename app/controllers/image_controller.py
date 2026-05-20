from fastapi import Response
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
