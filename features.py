from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# Load pre-trained CLIP model and processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Open the image
image = Image.open(r"img\2.jpg")

# Preprocess image
inputs = processor(images=image, return_tensors="pt", padding=True)

# Get image features from CLIP model
with torch.no_grad():
    image_features = model.get_image_features(**inputs)

# You can then use image_features as a representation for generating captions
print(type(image_features))
clip_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
image_description = f"An image showing the features: {clip_features.mean().item()}"
print(f"\n\n{image_description}")

print("END END END")
