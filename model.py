import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np



model = models.alexnet(pretrained=True)

num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 52) # Replace num_classes with the number of classes in your dataset

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a standard size
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_dataset = torchvision.datasets.ImageFolder(root='./Images', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

def train_model(num_epochs=10):
    for epoch in range(num_epochs):  # Replace num_epochs with the desired number of epochs
        training_accuracy = 0
        totals = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            outputs_clone = outputs.clone().detach().numpy()

            labels_clone = labels.clone().detach().numpy()

            predicted_labels = np.argmax(outputs_clone, axis=1)
            matches = np.count_nonzero(predicted_labels - labels_clone)

            matches = len(labels_clone) - matches
            training_accuracy += matches

            totals += len(predicted_labels)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print("training accuracy: " + str(training_accuracy) + "/" + str(totals))
        print('Epoch: {}/{} | Batch: {}/{} | Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

    return model

