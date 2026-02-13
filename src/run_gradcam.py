import torch
import cv2
import numpy as np
from torchvision import transforms

from src.dataset import VideoDataset
from src.model import FightDetector
from src.gradcam import GradCAM, overlay_cam_on_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load model
model = FightDetector(num_classes=2)
model.load_state_dict(torch.load("models/fight_model.pth", map_location=device))
model.to(device)
model.eval()

# Choose target conv layer
target_layer = model.backbone.layer4[1].conv2
gradcam = GradCAM(model.backbone, target_layer)

# Load one sample
dataset = VideoDataset("data/mini-test", transform=transform)
frames, label = dataset[0]

# Take one frame
frame = frames[0].unsqueeze(0).to(device)   # shape: (1, 3, 224, 224)

# Forward through backbone directly for prediction
with torch.no_grad():
    output = model.backbone(frame)
    _, pred = torch.max(output, 1)

# Generate Grad-CAM
cam = gradcam.generate(frame, pred.item())


# Convert tensor to image
# Load original frame again for visualization
import cv2

video_path = dataset.video_paths[0]
cap = cv2.VideoCapture(video_path)
ret, original_frame = cap.read()
cap.release()

original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
original_frame = cv2.resize(original_frame, (224, 224))


overlay = overlay_cam_on_image(original_frame, cam)


cv2.imwrite("gradcam_result.jpg", overlay)

print("Grad-CAM saved as gradcam_result.jpg")
