import sys
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax().item()
        score = output[:, class_idx]
        score.backward()
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, ...] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        return heatmap

    def close(self):
        for handle in self.hook_handles:
            handle.remove()


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0), image


def load_model(path):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model


def predict(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence, prediction = torch.max(probs, 1)
    return prediction.item(), confidence.item()


def save_heatmap(original_img, heatmap, output_path="heatmap_output.png", alpha=0.4):
    heatmap_np = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)

    # Convert PIL image to numpy RGB
    img_np = np.array(original_img)

    # Resize heatmap to match image size
    heatmap_color = cv2.resize(heatmap_color, (img_np.shape[1], img_np.shape[0]))
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend heatmap with original image
    superimposed_img = heatmap_color * alpha + img_np * (1 - alpha)
    superimposed_img = superimposed_img.astype(np.uint8)

    result = Image.fromarray(superimposed_img)
    result.save(output_path)


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_single_image.py <image_path>")
        return

    image_path = sys.argv[1]
    model_path = "radiology_model.pth"
    class_names = ['NORMAL', 'PNEUMONIA']

    model = load_model(model_path)
    input_tensor, original_img = preprocess_image(image_path)

    pred_idx, confidence = predict(model, input_tensor)
    print(f"Prediction: {class_names[pred_idx]} ({confidence * 100:.2f}%)")

    # GradCAM on last conv layer of ResNet18
    gradcam = GradCAM(model, model.layer4[1].conv2)
    heatmap = gradcam.generate(input_tensor, class_idx=pred_idx)
    save_heatmap(original_img, heatmap, output_path="heatmap_output.png", alpha=0.4)
    gradcam.close()


if __name__ == "__main__":
    main()
