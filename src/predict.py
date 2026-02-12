import torch
from torchvision import transforms
from dataset import VideoDataset
from model import FightDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = FightDetector(num_classes=2)
model.load_state_dict(torch.load("models/fight_model.pth", map_location=device))
model.to(device)
model.eval()

dataset = VideoDataset("data/mini_test", transform=transform)

video, label = dataset[0]
video = video.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(video)
    _, prediction = torch.max(output, 1)

classes = ["Fight", "NonFight"]

print("Actual:", classes[label])
print("Predicted:", classes[prediction.item()])
