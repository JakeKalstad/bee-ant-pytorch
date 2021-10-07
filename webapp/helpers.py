import io
import torch

import torchvision.transforms as transforms
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

classes = ["Ant", "Bee"]


def get_prediction(app, image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    model = app.model.to(device)
    tensor = tensor.to(device)
    output = model(tensor.unsqueeze(0))
    predicted = classes[output[0].argmax(0)]
    return output[0].argmax(0), predicted


def transform_image(image_bytes):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return preprocess(image)
