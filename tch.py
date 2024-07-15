import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import resnet50
import torch.optim as optim
import torch.nn as nn
import time

transform = transforms.Compose([
    # resize image
    transforms.Resize((224,224)),
    # convert image to matrix
    transforms.ToTensor(),
])
# Load dataset
train_dataset = datasets.ImageFolder(root='./train',transform=transform)
val_dataset = datasets.ImageFolder(root='./val',transform=transform)
test_dataset = datasets.ImageFolder(root='./test',transform=transform)

# create image loader
train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=True)

# init resnet50
model = resnet50(pretrained = True)
num_classes = 5
# fully connected layer is linear 
# new fc will have the same input as before but different number of outputs(usually the number of classes)
model.fc = nn.Linear(in_features = model.fc.in_features, out_features= num_classes)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

epochs = 10


for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs,labels in train_loader:
        # acquired batch images
        # inputs: [batch_size, channels, height, width]
        # labels: [batch_size]
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)


    epoch_loss = running_loss / len(train_loader.dataset) # type: ignore
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    model.eval()
    test_running_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_running_corrects += torch.sum(preds == labels.data)

    test_acc = test_running_corrects.double() / len(test_loader.dataset) # type: ignore
    print(f"Test Accuracy: {test_acc:.4f}")










