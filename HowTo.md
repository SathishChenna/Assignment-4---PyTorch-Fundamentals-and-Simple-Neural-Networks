# CNN MNIST Training with Live Visualization

This project implements a 4-layer CNN for MNIST digit classification with real-time training visualization.

## Requirements
- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- flask

## Setup
1. Install required packages using requirements.txt:
```
pip install -r requirements.txt
```

2. Start the visualization server:
```
python server.py
```

3. In a new terminal, start the training:
```
python cnn.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

You will see live training progress and loss curves. After training completes, the page will show predictions on 10 random test images.
