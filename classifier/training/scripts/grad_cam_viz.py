import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models
from dataset import CytologyDataset 

DEVICE = torch.device("cuda:6")
MODEL_PATH = "./best_model.pth"
IMG_DIR = "./images"
# order must match classifier label integers 0-4
CLASSES = ['NILM', 'LSIL', 'HSIL', 'ADENO', 'SCC']

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # In full_backward_hook, grad_output is a tuple
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_image, class_idx):
        output = self.model(input_image)
        self.model.zero_grad()
        
        score = output[0, class_idx]
        score.backward()
        
        # Calculate weights from gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # Process heatmap
        heatmap = F.relu(heatmap)
        heatmap /= (torch.max(heatmap) + 1e-8)
        return heatmap.detach().cpu().numpy()

def visualize_result(img_id, label_int):
    # Model Setup
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    target_layer = model.layer4[-1]
    cam = GradCAM(model, target_layer)

    import pandas as pd
    dummy_df = pd.DataFrame({'unique_id': [img_id], 'label_int': [label_int]})
    dataset = CytologyDataset(dummy_df, IMG_DIR, is_train=False)
    input_tensor, _, _ = dataset[0]
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    heatmap = cam.generate_heatmap(input_tensor, label_int)

    # Un-normalize for display
    img_display = input_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    img_display = (img_display * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    img_display = np.clip(img_display, 0, 1)

    # Visualization Processing
    heatmap_resized = cv2.resize(heatmap, (1024, 1024))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
    
    result = heatmap_color * 0.4 + img_display * 0.6

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(img_display)
    ax[0].set_title(f"Original: {CLASSES[label_int]}")
    ax[1].imshow(result)
    ax[1].set_title(f"Grad-CAM Attention ({CLASSES[label_int]})")
    
    save_name = f"./grad_cam/gradcam_{img_id}_{CLASSES[label_int]}.png"
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()
    print(f"Success! Saved: {save_name}")

if __name__ == "__main__":
    visualize_result("003982", 1)