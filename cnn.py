import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
import time

# Create directories
Path("static").mkdir(exist_ok=True)

# Model architectures
class SimpleCNN(nn.Module):
    def __init__(self, conv1_channels=16, conv1_kernel=3, tensor_size=(28, 28)):
        super(SimpleCNN, self).__init__()
        self.tensor_size = tensor_size
        
        # Calculate padding to maintain size
        padding = conv1_kernel // 2
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, conv1_channels, kernel_size=conv1_kernel, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Calculate output size after first conv layer
        conv1_output_size_x = tensor_size[0] // 2  # After MaxPool
        conv1_output_size_y = tensor_size[1] // 2  # After MaxPool
        
        # Second layer parameters calculated dynamically
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv1_channels, conv1_channels*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Calculate final output size for FC layer
        final_size = (conv1_output_size_x // 2) * (conv1_output_size_y // 2) * (conv1_channels*2)
        self.fc1 = nn.Linear(final_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ResidualCNN(nn.Module):
    def __init__(self):
        super(ResidualCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        identity = self.conv1(x)
        out = self.conv2(identity)
        out += identity  # Residual connection
        out = self.pool(out)
        out = self.conv3(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# Model factory
def get_model(model_name, conv_params=None):
    if conv_params is None:
        conv_params = {
            'conv1_channels': 16,
            'conv1_kernel': 3,
            'tensor_size': (28, 28)
        }
    
    models = {
        'simple': lambda: SimpleCNN(**conv_params),
        'deep': DeepCNN,
        'residual': ResidualCNN
    }
    return models[model_name]()

class ModelTrainer:
    def __init__(self, model_name, learning_rate, batch_size, optimizer_name='adam', conv_params=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = get_model(model_name, conv_params).to(self.device)
        
        # Initialize optimizer based on choice
        if optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), 
                                     lr=learning_rate,
                                     momentum=0.9)  # Added momentum for better SGD performance
        else:
            self.optimizer = optim.Adam(self.model.parameters(), 
                                      lr=learning_rate)
        
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.current_epoch = 0
        self.total_epochs = 0

    def save_progress(self, training_data):
        # Save current progress to JSON file
        with open('static/training_data.json', 'w') as f:
            json.dump(training_data, f)

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return 100. * correct / total

    def train_epoch(self, train_loader, epoch, epochs, training_data, model_name):
        self.current_epoch = epoch
        self.total_epochs = epochs
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Save progress more frequently (every batch)
            self.train_losses.append(loss.item())
            self.train_accuracies.append(100. * correct / total)
            
            # Update training data and save
            training_data[model_name] = {
                'losses': self.train_losses,
                'train_accuracies': self.train_accuracies,
                'test_accuracies': self.test_accuracies,
                'current_epoch': self.current_epoch + 1,
                'total_epochs': self.total_epochs
            }
            self.save_progress(training_data)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        return total_loss / len(train_loader), 100. * correct / total

def main():
    print("Waiting for training parameters from web interface...")
    while not Path('static/start_training').exists():
        time.sleep(1)
        continue
    
    # Remove the flag file
    Path('static/start_training').unlink()
    
    # Load training parameters
    try:
        with open('static/training_params.json', 'r') as f:
            params = json.load(f)
    except FileNotFoundError:
        print("No training parameters found. Please use the web interface to start training.")
        return

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    test_dataset = torch.utils.data.Subset(test_dataset, indices=range(1000))
    
    print(f"Dataset sizes: Training={len(train_dataset)}, Test={len(test_dataset)}")

    # Create trainers for both models
    trainers = {
        'model1': ModelTrainer(
            params['model1']['type'],
            params['model1']['learning_rate'],
            params['model1']['batch_size'],
            params['model1']['optimizer']
        ),
        'model2': ModelTrainer(
            params['model2']['type'],
            params['model2']['learning_rate'],
            params['model2']['batch_size'],
            params['model2']['optimizer']
        )
    }

    # Training loops
    epochs = params['epochs']

    # Initialize training data dictionary
    training_data = {
        'model1': {
            'losses': [], 
            'train_accuracies': [], 
            'test_accuracies': [],
            'current_epoch': 0,
            'total_epochs': epochs
        },
        'model2': {
            'losses': [], 
            'train_accuracies': [], 
            'test_accuracies': [],
            'current_epoch': 0,
            'total_epochs': epochs
        }
    }

    # Save initial state
    with open('static/training_data.json', 'w') as f:
        json.dump(training_data, f)

    for model_name, trainer in trainers.items():
        print(f"\nTraining {model_name}...")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=trainer.batch_size,
            shuffle=True, 
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=100,
            shuffle=True,
            pin_memory=True
        )

        start_time = time.time()
        
        for epoch in range(epochs):
            loss, acc = trainer.train_epoch(train_loader, epoch, epochs, training_data, model_name)
            test_acc = trainer.evaluate(test_loader)
            trainer.test_accuracies.append(test_acc)
            
            # Update and save progress after each epoch
            training_data[model_name]['test_accuracies'] = trainer.test_accuracies
            trainer.save_progress(training_data)

            elapsed_time = time.time() - start_time
            print(f'Epoch {epoch+1}: Loss={loss:.4f}, Train Acc={acc:.2f}%, '
                  f'Test Acc={test_acc:.2f}%, Time={elapsed_time:.1f}s')

        # Save final results for this model
        training_data[model_name] = {
            'losses': trainer.train_losses,
            'train_accuracies': trainer.train_accuracies,
            'test_accuracies': trainer.test_accuracies,
            'training_time': elapsed_time,
            'current_epoch': epochs,
            'total_epochs': epochs
        }
        trainer.save_progress(training_data)

if __name__ == '__main__':
    main() 