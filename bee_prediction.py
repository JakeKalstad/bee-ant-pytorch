from PIL import Image
import torch
from torch import nn
from torchvision import models

from torchvision import transforms
from beelib import imshow, class_names, device

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)


model_ft.load_state_dict(torch.load("bee-model.pth"))

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model_ft.eval()

print("start")
input()

img = Image.open("test/ddd.png").convert('RGB')
img2 = Image.open("test/abcd.png").convert('RGB')

print(class_names)

x = preprocess(img)
x2 = preprocess(img2)


output = model_ft(x.unsqueeze(0).cuda())
predicted = class_names[output[0].argmax(0)]
print(predicted + " Wanted ants")

output = model_ft(x2.unsqueeze(0).cuda())
predicted = class_names[output[0].argmax(0)]
print(predicted + " Wanted bees")
