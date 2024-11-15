# MNIST CNN Training Visualizer

A real-time visualization system for training and comparing multiple CNN architectures on the MNIST dataset.

## Overview

This application provides a web-based interface to monitor CNN training progress in real-time. It consists of:
- A 4-layer CNN model for MNIST digit classification
- Real-time training visualization with dual-axis plots
- Live accuracy and loss monitoring
- Final model evaluation with visual results

## System Architecture

### 1. Neural Network (cnn.py)
The CNN architecture consists of:
- Input Layer: Accepts 28x28 grayscale images
- Two Convolutional Layers:
  - Conv1: 1→16 channels, 3x3 kernel, ReLU activation, 2x2 max pooling
  - Conv2: 16→32 channels, 3x3 kernel, ReLU activation, 2x2 max pooling
- Two Fully Connected Layers:
  - FC1: 32*7*7 → 128 neurons
  - FC2: 128 → 10 neurons (output layer)

Key Features:
- CUDA support for GPU acceleration
- Batch size of 512 for efficient training
- Adam optimizer
- CrossEntropyLoss for classification
- Real-time metric logging
- Best model checkpoint saving

### 2. Web Server (server.py)
A Flask-based server that:
- Serves the visualization interface
- Provides REST endpoints for real-time data
- Handles JSON data communication
- Routes:
  - `/`: Main visualization page
  - `/get_training_data`: JSON endpoint for training metrics

### 3. Frontend (templates/index.html)
Interactive visualization interface featuring:
- Combined plot with dual y-axes:
  - Left axis: Accuracy (90-100% range)
  - Right axis: Loss
- Real-time metrics display
- Color-coded metrics:
  - Blue: Training Loss
  - Green: Training Accuracy
  - Orange: Test Accuracy (dashed line)
- Auto-refresh every 2 seconds
- Test results visualization

## Data Flow
1. CNN training process generates metrics
2. Metrics saved to static/training_data.json
3. Flask server serves data via REST endpoint
4. Frontend fetches and visualizes data
5. Plot updates automatically every 2 seconds

## Setup and Installation

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (recommended)
- Modern web browser

### Installation
1. Clone the repository:
```bash
git clone [repository-url]
cd mnist-cnn-visualizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application
1. Start the visualization server:
```bash
python server.py
```

2. In a new terminal, start the training:
```bash
python cnn.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## Project Structure
```
project_root/
├── static/                  # Static files and training data
├── templates/              
│   └── index.html          # Visualization frontend
├── server.py               # Flask server
├── cnn.py                  # CNN implementation
├── requirements.txt        # Dependencies
├── README.md              
└── HowTo.md               # Quick start guide
```

## Key Features
- Real-time training visualization
- Dual-axis plotting (loss and accuracy)
- CUDA GPU acceleration
- Automatic best model checkpointing
- Interactive web interface
- Test set evaluation with visual results
- Progress bars with live metrics
- Focused accuracy visualization (90-100% range)

## Technical Details

### Training Parameters
- Batch Size: 512
- Learning Rate: Adam optimizer defaults
- Epochs: 5 (configurable)
- Dataset Split: 60,000 training, 10,000 test images

### Data Processing
- Image normalization: mean=0.1307, std=0.3081
- Input size: 28x28 grayscale images
- Data augmentation: None (base implementation)

### Monitoring
- Training Loss
- Training Accuracy
- Test Accuracy (per epoch)
- CUDA memory usage
- Best model tracking

## Performance
- Typical accuracy: >98% on test set
- Training time: ~5-10 minutes on modern GPU
- Memory usage: ~1-2GB GPU memory

## Future Improvements
- Add data augmentation
- Implement learning rate scheduling
- Add model architecture visualization
- Include confusion matrix
- Add export functionality for trained models
- Implement early stopping

## License
[Your License Here]

## Contributors
[Your Name/Organization] 

## Available Models

1. **Simple CNN**
   - 2 convolutional layers
   - 2 fully connected layers
   - Basic architecture for MNIST

2. **Deep CNN**
   - Deeper architecture with batch normalization
   - Dropout for regularization
   - More filters and neurons

3. **Residual CNN**
   - Residual connections
   - Batch normalization
   - Optimized for training stability

## Model Comparison
- Side-by-side training visualization
- Configurable parameters:
  - Learning rate
  - Batch size
  - Number of epochs
- Real-time performance metrics
- Interactive model selection