import torch
import torch.nn.functional as F
import cv2
import numpy as np

def generate_heatmap(model, image_tensor, target_layer, class_idx):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    output = model(image_tensor)
    class_score = output[0, class_idx]
    class_score.backward()

    grads_val = gradients[0]
    activations_val = activations[0]

    weights = grads_val.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations_val).sum(dim=1, keepdim=True)

    cam = F.relu(cam)
    cam = cam.squeeze().detach().numpy()
    cam = cv2.resize(cam, (224, 224))
    cam -= np.min(cam)
    cam /= np.max(cam)
    return cam
