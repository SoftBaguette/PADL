import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class TimePredictionNetwork(nn.Module):
    def __init__(self):
        super(TimePredictionNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 28 * 28, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def predict(images):
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply transformations to images
    images = transform(images)

    # Determine which device the input tensor is on
    device = torch.device("cuda" if images.is_cuda else "cpu")

    model = TimePredictionNetwork()
    # Move to same device as input images
    model = model.to(device)
    # Load network weights
    model.load_state_dict(torch.load('weights.pkl', map_location=device))
    # Put model in evaluation mode
    model.eval()

    with torch.no_grad():
        # Pass images to model
        predicted_times = model(images)

    # Return predicted times
    return predicted_times
