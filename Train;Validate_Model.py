import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import timm  # Import timm for Swin Transformer
from torch.utils.data import Dataset
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score

class XRayDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.fractured_train_images = sorted(os.listdir(os.path.join(data_dir, "train", "fractured")))
        self.non_fractured_train_images = sorted(os.listdir(os.path.join(data_dir, "train", "not fractured")))
        self.fractured_val_images = sorted(os.listdir(os.path.join(data_dir, "val", "fractured")))
        self.non_fractured_val_images = sorted(os.listdir(os.path.join(data_dir, "val", "not fractured")))

    def __len__(self):
        return len(self.fractured_train_images) + len(self.non_fractured_train_images) + len(self.fractured_val_images) + len(self.non_fractured_val_images)

    def __getitem__(self, idx):
        if idx < len(self.fractured_train_images):
            image_path = os.path.join(self.data_dir, "train", "fractured", self.fractured_train_images[idx])
            label = 1
        elif idx < len(self.fractured_train_images) + len(self.non_fractured_train_images):
            image_path = os.path.join(self.data_dir, "train", "not fractured", self.non_fractured_train_images[idx - len(self.fractured_train_images)])
            label = 0
        elif idx < len(self.fractured_train_images) + len(self.non_fractured_train_images) + len(self.fractured_val_images):
            image_path = os.path.join(self.data_dir, "val", "fractured", self.fractured_val_images[idx - len(self.fractured_train_images) - len(self.non_fractured_train_images)])
            label = 1
        else:
            image_path = os.path.join(self.data_dir, "val", "not fractured", self.non_fractured_val_images[idx - len(self.fractured_train_images) - len(self.non_fractured_train_images) - len(self.fractured_val_images)])
            label = 0

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)

data_dir = "/content/xray-dataset"
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Reduce image size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = XRayDataset(data_dir, transform)

model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=2)  # Load pretrained model

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

batch_size = 8  # Reduce batch size
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {running_loss/len(train_loader)}")

    model.eval()
    running_loss = 0.0
    val_predictions = []
    val_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            val_predictions.extend(torch.argmax(outputs, 1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(val_labels, val_predictions)
    precision = precision_score(val_labels, val_predictions)

    #print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {running_loss/len(val_loader)}, Validation Accuracy: {accuracy}, Validation Precision: {precision}")
    print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {running_loss/len(val_loader)}")
    print(f"Epoch [{epoch+1}/{num_epochs}] Validation Accuracy: {accuracy}")
    print(f"Epoch [{epoch+1}/{num_epochs}] Validation Precision: {precision}")
