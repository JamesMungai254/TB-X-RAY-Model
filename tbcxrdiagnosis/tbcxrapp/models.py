from django.db import models  # Assuming this is part of a Django app
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import os
# The model architecture
class TBModel(nn.Module):
    def __init__(self, num_classes=2):
        super(TBModel, self).__init__()
        # Directly initialize the ResNet model
        self.resnet = models.resnet18(pretrained=True)  # Using pretrained weights for better accuracy
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
    
# Load the full model directly
def load_model():
    device = torch.device("cpu")  # Ensure it's loaded on CPU
    
    # Create an instance of your model
    model = TBModel()  # Ensure TBModel is defined in models.py
    
    # Load the state dictionary
    model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)  # Load parameters into the model
    
    model.to(device)  # Move to the correct device
    model.eval()  # Set to evaluation mode
    
    return model


# Preprocessing the uploaded image
def process_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to ensure consistent size
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for pre-trained models
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB mode
    except FileNotFoundError:
        raise FileNotFoundError(f"The image file '{image_path}' was not found.")
    except Exception as e:
        raise ValueError(f"Error processing the image: {e}")
    
    image = preprocess(image).unsqueeze(0)  # Add batch dimension (necessary for models)
    return image


# Make prediction
def predict(image_path):
    model = load_model()  # Load the trained model
    image_tensor = process_image(image_path)  # Process the image
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_tensor = image_tensor.to(device)  # Move image to the same device as the model
    
    with torch.no_grad():
        outputs = model(image_tensor)  # Forward pass
        _, predicted = torch.max(outputs, 1)  # Get the class index with the highest probability
        
    return predicted.item()  # Return 0 for normal, 1 for tuberculosis
