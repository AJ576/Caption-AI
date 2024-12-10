from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from google.cloud import aiplatform
import base64
import vertexai
from google.oauth2 import service_account
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.vision_models import ImageGenerationModel

import base64

PROJECT_ID = "softwaredesign-443701"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
credentials = service_account.Credentials.from_service_account_file('vertexai_key.json')
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

from vertexai.preview.generative_models import GenerativeModel
gemini_model = GenerativeModel("gemini-1.5-flash")


def generate_caption(image_path):
    # Load BLIP processor and model
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # Open image
    image = Image.open(image_path)

    custom_prompt = "What are objects in the image"
    # Preprocess image and custom prompt
    inputs = blip_processor(images=image, return_tensors="pt")
    
    # Generate caption
    out = blip_model.generate(**inputs)

    # Decode and return caption
    return blip_processor.decode(out[0], skip_special_tokens=True)

def gemini_cap(caption):
    prompt = f"""
    Here is a simple description of an image: {caption}.
    Please generate a caption that looks like an instagram caption. If there is a lack of info, work with what you have.

    """

    # Generate the caption
    response = gemini_model.generate_content(contents=[prompt])
    response_text = response.text.strip()

    return response_text


if __name__ == '__main__':
    image_path = r"img\4.jpg"  # Update to your actual image path
    caption = generate_caption(image_path)
    print("Generated Caption:", caption)
    caption = gemini_cap(caption)
    print("Generated Caption:", caption)

