import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, models
import huggingface_hub
from huggingface_hub import hf_hub_download


def pred_and_plot(image, model, transform, class_names, device="cpu"):
    transformed_image = transform(image)

    model.eval()
    with torch.inference_mode():
        pred_logit = model(transformed_image.unsqueeze(0).to(device))
        pred_label = torch.argmax(torch.softmax(pred_logit, dim=1), dim=1)

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(f"Prediction: {class_names[pred_label.item()]}")
    ax.axis("off")
    return fig


@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    num_classes = 2  # Cat and Dog
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    repo_id = "anurag2506/cats_dogs_classification"
    model_path = hf_hub_download(repo_id=repo_id, filename="model.pth")
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    model.eval()
    return model


classification_model = load_model()
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Adjust if different normalization
    ]
)

class_names = {0: "Cat", 1: "Dog"}


st.title("Image Classification App with Plot")
st.write("Upload an image to classify and visualize the result!")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing and Classifying...")

    fig = pred_and_plot(image, classification_model, transform, class_names)

    st.pyplot(fig)
