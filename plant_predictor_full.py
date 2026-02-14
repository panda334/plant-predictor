import streamlit as st
import torch
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
from collections import deque


MODEL_PATH = "best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 15
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

CLASS_NAMES = [
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

if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=5)  


st.set_page_config(page_title="🌱 Plant Disease Predictor", page_icon="🌿")
st.title("🌱 Plant Disease Predictor")
st.write("Upload a plant image and the model will predict its class!")

uploaded_file = st.file_uploader("Upload a plant image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=400)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        prediction = CLASS_NAMES[pred.item()]

    st.success(f"Predicted Class: **{prediction}**")

    st.session_state.history.appendleft((img, prediction))


if st.session_state.history:
    st.write("### 📜 Last 5 Predictions")
    for i, (hist_img, hist_pred) in enumerate(st.session_state.history):
        st.image(hist_img, width=150, caption=f"{hist_pred}")
