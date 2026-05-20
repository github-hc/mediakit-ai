from unittest.mock import patch

# IMPORTANT: Mock the DiffusersProvider initialization BEFORE importing app/TestClient.
# This prevents pytest from attempting to load/download Stable Diffusion 1.5 during FastAPI startup.
initialize_patcher = patch("app.providers.diffusers.provider.DiffusersProvider.initialize")
mock_initialize = initialize_patcher.start()

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from app.core.exceptions import ProviderError

client = TestClient(app)

def test_health_endpoint():
    """Test the base health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "mediakit-ai"
    assert data["status"] == "healthy"

@patch("app.providers.ollama.provider.OllamaProvider.generate_image")
def test_generate_image_ollama_success(mock_generate_image):
    """Test generating an image using the Ollama provider (mocked)."""
    # Ensure Ollama provider is active
    settings.IMAGE_PROVIDER = "ollama"
    
    # Generate mock binary PNG bytes for mock return value
    from io import BytesIO
    from PIL import Image
    img = Image.new("RGB", (256, 256), color=(70, 130, 180)) # steel blue
    buf = BytesIO()
    img.save(buf, format="PNG")
    mock_generate_image.return_value = buf.getvalue()
    
    payload = {
        "prompt": "A sunset view from the workspace window",
        "width": 256,
        "height": 256,
        "seed": 123
    }
    
    response = client.post("/api/v1/images/generate", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0  # Content should be non-empty binary data

@patch("app.providers.diffusers.provider.DiffusersProvider.generate_image")
def test_generate_image_diffusers_success(mock_generate_image):
    """Test generating an image using the local Diffusers provider (mocked)."""
    # Ensure diffusers provider is active
    settings.IMAGE_PROVIDER = "diffusers"
    
    # Generate mock binary PNG bytes for mock return value
    from io import BytesIO
    from PIL import Image
    img = Image.new("RGB", (256, 256), color=(219, 112, 147)) # pale violet red
    buf = BytesIO()
    img.save(buf, format="PNG")
    mock_generate_image.return_value = buf.getvalue()
    
    payload = {
        "prompt": "A fantasy mountain castle",
        "negative_prompt": "blurry, dark",
        "width": 256,
        "height": 256,
        "num_inference_steps": 20,
        "guidance_scale": 8.0,
        "seed": 789
    }
    
    response = client.post("/api/v1/images/generate", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0

def test_generate_image_validation_error():
    """Test validation errors for invalid request payloads."""
    # Empty prompt
    payload = {
        "prompt": "",
        "width": 256
    }
    response = client.post("/api/v1/images/generate", json=payload)
    assert response.status_code == 422

    # Negative width
    payload = {
        "prompt": "Valid prompt",
        "width": -10
    }
    response = client.post("/api/v1/images/generate", json=payload)
    assert response.status_code == 422

@patch("app.providers.ollama.provider.OllamaProvider.generate_image")
def test_generate_image_provider_error(mock_generate_image):
    """Test mapping of ProviderError to 502 Bad Gateway response."""
    # Set mock to raise ProviderError
    mock_generate_image.side_effect = ProviderError("ollama", "Connection failed to server.")
    
    settings.IMAGE_PROVIDER = "ollama"
    
    payload = {
        "prompt": "Trigger a mock provider failure",
        "width": 256
    }
    
    response = client.post("/api/v1/images/generate", json=payload)
    assert response.status_code == 502
    data = response.json()
    assert data["error"] == "ProviderError"
    assert "ollama" in data["message"]

def teardown_module(module):
    """Clean up patcher after all tests have executed."""
    initialize_patcher.stop()
