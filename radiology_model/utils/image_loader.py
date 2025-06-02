# radiology_model/utils/image_loader.py
from PIL import Image
import torchvision.transforms as transforms
import config

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)
