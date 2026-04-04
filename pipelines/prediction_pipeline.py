import torch
from PIL import Image
import torchvision.transforms as transforms
from src.model import CNNModel
from utils.model_save import load_model

# Class labels
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def predict_image(image_path, model_path):

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = CNNModel(input_channels=1, num_classes=10)
    model = load_model(model, model_path)
    model.to(device)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path)
    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]
