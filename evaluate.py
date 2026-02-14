import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import os

# ======= Paths =======
VAL_PATH = "dataset_split/val"
MODEL_PATH = "best_model.pth"

# ======= Device =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======= Transform =======
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ======= Dataset & Loader =======
val_dataset = datasets.ImageFolder(VAL_PATH, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ======= Model =======
num_classes = len(val_dataset.classes)
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ======= Evaluation =======
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=val_dataset.classes))
