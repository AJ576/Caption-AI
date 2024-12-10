from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

def answer_visual_question(image_path, question):
    # Load BLIP processor and VQA model
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base")

    # Open the image
    image = Image.open(image_path)

    # Preprocess the inputs with both image and question
    inputs = blip_processor(images=image, text=question, return_tensors="pt")

    # Generate the answer
    out = blip_model.generate(**inputs, max_length=50)

    # Decode the output to get the answer
    answer = blip_processor.decode(out[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    # Path to your image and a question
    image_path = r"img\1.jpg"
    question = "What is the person in the image doing?"

    # Generate answer
    answer = answer_visual_question(image_path, question)
    print("Answer:", answer)
