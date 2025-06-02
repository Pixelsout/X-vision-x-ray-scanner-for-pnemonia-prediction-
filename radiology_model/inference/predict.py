import torch
from torchvision import transforms
from PIL import Image
from radiology_model.training.model import build_model

def predict(image_path, model_path="radiology_model.pth"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(image.to(device))
        _, preds = torch.max(outputs, 1)
        prob = torch.nn.functional.softmax(outputs, dim=1)[0][preds].item()

    print(f"Prediction: {'PNEUMONIA' if preds.item() == 1 else 'NORMAL'} with confidence {prob:.2f}")
