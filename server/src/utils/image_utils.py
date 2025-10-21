import base64
from io import BytesIO
from typing import List, Tuple

from PIL import Image


def resize_images(
    images: List[Image.Image],
    image_size: List[int] | Tuple[int],
) -> List[Image.Image]:
    images_resized = [image.resize((image_size[0], image_size[1])) for image in images]
    if images_resized[0].mode != 'RGB':
        images_resized = [image.convert('RGB') for image in images_resized]
    return images_resized

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def base64_to_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    return Image.open(BytesIO(base64.b64decode(base64_string)))
