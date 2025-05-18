import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class CaratRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        self.base = resnet18(pretrained=True)
        self.base.fc = nn.Linear(self.base.fc.in_features, 1)

    def forward(self, x):
        return self.base(x)

class EmeraldDataset(Dataset):
    def __init__(self, img_dir, labels_dict, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = os.listdir(img_dir)
        self.labels_dict = labels_dict

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.labels_dict[img_name]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

labels_dict = {
    'emerald1.jpg': 1.2,
    'emerald2.jpg': 0.9,
    # add all image:carat pairs here
}

dataset = EmeraldDataset('path/to/images', labels_dict, transform=transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CaratRegressor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        preds = model(images).squeeze()
        loss = F.mse_loss(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
