import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
from torchvision.models import ResNet50_Weights

MODEL_PATH = "best_model.pth"
IMAGE_PATH = "download (1).jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Data transform ======
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ====== Model ======
num_classes = 15  # عدل حسب عدد الكلاسات
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ====== أسماء الكلاسات ======
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# ====== Load image ======
img = Image.open(IMAGE_PATH).convert("RGB")
img = transform(img).unsqueeze(0).to(device)

# ====== Prediction ======
with torch.no_grad():
    outputs = model(img)
    _, pred = torch.max(outputs,1)
    print(f"Predicted class: {class_names[pred.item()]}")
