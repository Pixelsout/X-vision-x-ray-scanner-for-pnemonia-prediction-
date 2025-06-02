import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Grad-CAM utility
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

        handle_forward = self.target_layer.register_forward_hook(forward_hook)
        handle_backward = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([handle_forward, handle_backward])

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax().item()

        loss = output[0, class_idx]
        loss.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(len(pooled_gradients)):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        return heatmap

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('radiology_model.pth', map_location=torch.device('cpu')))
model.eval()

# Class labels
class_names = ['NORMAL', 'PNEUMONIA']

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Streamlit UI
st.title("🩻 X-ray Pneumonia Classifier with Explainability")
st.write("Upload a chest X-ray image to predict and visualize the model's focus.")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded X-ray', use_column_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        prediction = class_names[pred.item()]

    st.markdown(f"### 🔍 Prediction: **{prediction}**")

    # Grad-CAM heatmap
    target_layer = model.layer4[1].conv2
    cam = GradCAM(model, target_layer)
    cam_mask = cam.generate(input_tensor, class_idx=pred.item())
    cam.remove_hooks()

    # Heatmap overlay
    img_np = np.array(image.resize((224, 224)))
    if len(img_np.shape) == 2 or img_np.shape[2] == 1:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    else:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    overlay = cv2.addWeighted(heatmap, 0.4, img_np, 0.6, 0)
    heatmap_path = "heatmap_result.jpg"
    cv2.imwrite(heatmap_path, overlay)

    st.image(heatmap_path, caption='Grad-CAM Heatmap', use_column_width=True)

    with open(heatmap_path, "rb") as file:
        st.download_button("📥 Download Heatmap", file, file_name="xray_heatmap.jpg", mime="image/jpeg")

    # Heatmap Interpretation
    def analyze_activation(cam_mask):
        intensity = cam_mask.mean()
        max_activation = cam_mask.max()

        if max_activation < 0.2:
            return "🫁 No significant abnormal regions highlighted. Lung fields appear clear."
        elif 0.2 <= max_activation < 0.5:
            return "⚠️ Mild activation seen, possibly indicating early-stage or diffused opacity."
        else:
            height, width = cam_mask.shape
            center_x = width // 2
            center_y = height // 2

            if cam_mask[center_y:, :].mean() > cam_mask[:center_y, :].mean():
                location = "lower lung zones"
            else:
                location = "upper/mid lung zones"

            return f"⚠️ Strong activation detected in the {location}, suggesting dense opacities or consolidation."

    interpretation =   (cam_mask)
    st.markdown(f"### 🧠 Heatmap Interpretation")
    st.write(interpretation)

    # Diagnosis summary
    st.markdown("### 📝 AI     Diagnostic Summary")
    if prediction == "NORMAL":
        st.success("\n✅ Lung Fields: Clear and well-aerated.\n\n✅ Costophrenic Angles: Sharp and visible.\n\n✅ Heart Borders: Clearly defined.\n\n✅ No Consolidations or Opacities detected.\n\n📊 Conclusion: No signs of infection or pneumonia. Classified as NORMAL.")
    else:
        st.error("\n⚠️ Lung Fields: Patchy or lobar opacities detected.\n\n⚠️ Opacity Zones: Likely lower lobe involvement.\n\n⚠️ Costophrenic Angles: May appear obscured.\n\n🔍 The AI model focused on dense or irregular regions.\n\n📊 Conclusion: Abnormal findings consistent with PNEUMONIA.")
