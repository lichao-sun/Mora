import io

import requests
from PIL import Image

from mora.actions.action import Action
from mora.messages import Message

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": "Bearer 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"}


class GenerateImageWithTextAPI(Action):
    """Generate Image with Text Action"""
    name: str = "Generate Image with Text"

    async def run(self, message: Message):
        """Run action"""
        payload = {
            "inputs": message.content,
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        image_bytes = response.content

        image = Image.open(io.BytesIO(image_bytes))
        return Message(image_content=image)
