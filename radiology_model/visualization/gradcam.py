import cv2
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image


def get_heatmap(image_path):
    model = models.resnet18(pretrained=True)
    finalconv_name = 'layer4'
    model.eval()

    def hook_function(module, input, output):
        global features
        features = output

    model._modules.get(finalconv_name).register_forward_hook(hook_function)

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)
    output = model(input_tensor)

    pred_class = output.argmax().item()
    grads = torch.autograd.grad(output[0][pred_class], features)[0]
    pooled_grads = torch.mean(grads, dim=[0, 2, 3])
    for i in range(features.shape[1]):
        features[0, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(features, dim=1).squeeze().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed = heatmap_img * 0.4 + img
    cv2.imwrite('heatmap_output.jpg', overlayed)
    return 'heatmap_output.jpg'
