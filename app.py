import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Running on device: {DEVICE}")

st.set_page_config(page_title="Leaf Disease Classifier", page_icon="🌿")
st.title("🌿 Leaf Disease Classifier")
st.write("Upload a leaf image and the model will predict the disease.")


@st.cache_resource
def load_classes(path="class_mapping.json"):
    with open(path) as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return [idx_to_class[i] for i in range(len(idx_to_class))]

class_names = load_classes()

@st.cache_resource
def load_model(path="vgg16_best_leaf_state_dict.pth", device=DEVICE):
    from torchvision import models
    import torch.nn as nn

    model = models.vgg16(weights=None)
    num_classes = 38
    model.classifier[6] = nn.Linear(4096, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


model = load_model()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_class = class_names[pred_idx]
        confidence = probs[0][pred_idx].item()
    st.write(f"### ✅ Predicted Class: {pred_class}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")
