import torch
from torchvision.models import resnet18, resnet34, resnet50, mobilenet_v2
from torchvision.io import image as Image

image = Image.read_image(r"E:\data\datasets\face\11785105.jpg").type(torch.float32)

image = image.unsqueeze(0)  # 升维
model18 = resnet18(pretrained=True)

model18.eval()
model_mobilenet = mobilenet_v2(pretrained=True)
pred18 = model18(image)
pred_mobilenet = model_mobilenet(image)

pass
