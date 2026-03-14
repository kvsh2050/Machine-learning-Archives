# Image Denoising for Microscopic Images using DnCNN (ARCHIVED)

![Microscopic Image Denoising Example](https://via.placeholder.com/800x400?text=Microscopic+Image+Denoising+Example)

## Table of Contents

1. [Introduction to Image Denoising](#introduction-to-image-denoising)
2. [Understanding DnCNN and Denoising Techniques](#understanding-dncnn-and-denoising-techniques)
3. [Dataset Preparation Guide](#dataset-preparation-guide)
4. [Model Training Pipeline](#model-training-pipeline)
5. [Testing & Deployment](#testing--deployment)

## Introduction to Image Denoising

### What is Image Denoising?

Image denoising is a fundamental image restoration task that aims to remove noise from images while preserving details and structures.

**Common Sources of Noise:**

* Gaussian noise
* Poisson noise
* Salt-and-pepper noise
* Speckle noise (common in microscopic images)

### Applications in Microscopy

* Enhancing low-contrast biological structures
* Improving automated cell/nuclei segmentation
* Noise-resilient microscopy workflows in research and diagnostics

## Understanding DnCNN and Denoising Techniques

### DnCNN (Denoising Convolutional Neural Network)

DnCNN is a deep CNN architecture designed specifically for image denoising:

* **Architecture**:

  * Multiple convolutional layers with ReLU and batch normalization
  * Learns a residual mapping from noisy image to noise
* **Advantages**:

  * Effective on Gaussian noise
  * Extensible to blind/noise-agnostic denoising

```python
from dncnn import DnCNN  # Custom implementation
model = DnCNN(depth=17, channels=1)  # For grayscale microscopic images
```

### Alternative Denoising Techniques

* **Traditional Filters**:

  * Gaussian Blur
  * Median Filter
  * Non-Local Means (NLM)
* **DL-Based Models**:

  * UNet for denoising
  * Autoencoders
  * Noise2Noise / Noise2Void (self-supervised)

## Dataset Preparation Guide

### Step 1: Image Collection

* **Sources**:

  * Microscopy datasets (BBBC, Allen Institute, DeepImageJ)
  * Lab-generated images using fluorescence or brightfield microscopy
* **Noise Injection (for supervised learning)**:

  * Simulate noise via:

    ```python
    noisy = clean + np.random.normal(0, sigma, clean.shape)
    ```

### Step 2: Preprocessing

* Grayscale conversion (if applicable)
* Normalize to `[0, 1]` or `[-1, 1]`
* Patch extraction (e.g., 64x64 or 128x128)

### Folder Structure

```
microscopy_dataset/
├── train/
│   ├── clean/
│   └── noisy/
├── val/
│   ├── clean/
│   └── noisy/
└── test/
    ├── clean/
    └── noisy/
```

## Model Training Pipeline

### 1. Environment Setup

```bash
pip install torch torchvision numpy matplotlib
```

### 2. DnCNN Model Training

```python
for epoch in range(num_epochs):
    for data in dataloader:
        noisy_imgs, clean_imgs = data
        optimizer.zero_grad()
        output = model(noisy_imgs)
        loss = criterion(output, clean_imgs)
        loss.backward()
        optimizer.step()
```

**Key Parameters:**

* `depth`: Number of convolutional layers (typically 17)
* `criterion`: MSELoss or Charbonnier Loss
* `optimizer`: Adam (lr=1e-3)

### 3. Data Augmentation (Optional)

* Horizontal/vertical flips
* Random cropping
* Intensity shifts

## Testing & Deployment

### Model Evaluation

```python
from skimage.metrics import peak_signal_noise_ratio as psnr

denoised = model(noisy_image.unsqueeze(0))
score = psnr(clean_image.numpy(), denoised.detach().numpy())
```

### Visual Comparison

```python
import matplotlib.pyplot as plt

plt.subplot(1, 3, 1)
plt.title("Noisy")
plt.imshow(noisy_image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Denoised")
plt.imshow(denoised, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Ground Truth")
plt.imshow(clean_image, cmap='gray')
```

### Export Model

```python
torch.save(model.state_dict(), 'dncnn_microscopy.pth')
```

### Deployment Options

1. **Integration with ImageJ plugins**
2. **Web App for Batch Processing**: Streamlit/FastAPI + PyTorch
3. **Microscope Integration**: Edge inference using ONNX or TensorRT

## Contributors

kvsh2050

## Images of the Denoising Project 
![Screenshot 2024-05-20 161231](https://github.com/user-attachments/assets/99874346-58fc-4c87-8fac-a4ab9c860674)

![Screenshot 2024-05-20 161330](https://github.com/user-attachments/assets/024bfe33-0d30-4b63-bdb7-2861ddc56b7d)





# Wildlife Object Detection with YOLOv8

![Wildlife Detection Example](https://via.placeholder.com/800x400?text=Wildlife+YOLOv8+Detection+Example)

## Table of Contents
1. [Introduction to Machine Learning](#introduction-to-machine-learning)
2. [Understanding YOLO and Ultralytics](#understanding-yolo-and-ultralytics)
3. [Dataset Collection Guide](#dataset-collection-guide)
4. [Model Training Pipeline](#model-training-pipeline)
5. [Testing & Deployment](#testing--deployment)

## Introduction to Machine Learning

### What is Machine Learning?
Machine Learning (ML) is a subset of artificial intelligence that enables systems to learn patterns from data without being explicitly programmed.

**Types of ML:**
- **Supervised Learning**: Labeled data (e.g., classification, regression)
- **Unsupervised Learning**: Unlabeled data (e.g., clustering)
- **Reinforcement Learning**: Reward-based systems

### Deep Learning (DL)
DL uses neural networks with multiple layers to learn hierarchical representations:
- **Computer Vision**: CNNs (YOLO, ResNet)
- **Natural Language Processing**: Transformers
- **Generative Models**: GANs, Diffusion Models

## Understanding YOLO and Ultralytics

### YOLO (You Only Look Once)
YOLO is a real-time object detection system:
- **Versions**: YOLOv3 → YOLOv8 (latest)
- **Key Features**:
  - Single-stage detector (fast)
  - Predicts bounding boxes & class probabilities
  - Processes entire image in one pass

### Ultralytics Ecosystem
Ultralytics provides the Python implementation of YOLOv8:
- **Features**:
  - Easy-to-use API
  - Pre-trained models
  - Training/validation pipelines
  - Multiple export formats

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Load pretrained model
```

## Dataset Collection Guide

### Step 1: Image Acquisition
- **Sources**:
  - Camera traps
  - Wildlife documentaries (with permissions)
  - Public datasets (GBIF, iNaturalist)
- **Requirements**:
  - Minimum 1,000 images per class
  - Diverse lighting/angles
  - 50-50% foreground/background ratio

### Step 2: Annotation
1. Use labeling tools:
   - LabelImg
   - CVAT
   - Roboflow
2. YOLO format:
   ```
   <class_id> <x_center> <y_center> <width> <height>
   ```
3. Split dataset:
   - Train (70%)
   - Val (20%)
   - Test (10%)

### Folder Structure
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── dataset.yaml
```

## Model Training Pipeline

### 1. Environment Setup
```bash
pip install ultralytics torch torchvision
```

### 2. Configuration (`dataset.yaml`)
```yaml
path: /path/to/dataset
train: train/images
val: val/images

names:
  0: elephant
  1: lion
  2: zebra
```

### 3. Start Training
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load model
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0  # GPU
)
```

**Key Parameters:**
- `imgsz`: Input size (640 recommended)
- `batch`: Adjust based on GPU memory
- `patience`: Early stopping

## Testing & Deployment

### Model Validation
```python
metrics = model.val()  # Evaluate on validation set
```

### Inference
```python
results = model.predict('test.jpg', conf=0.5)
results[0].show()  # Display results
```

### Export Options
```python
model.export(format='onnx')  # For production
```

### Deployment Options
1. **Edge Devices**: TensorRT, OpenVINO
2. **Web API**: FastAPI + ONNX runtime
3. **Mobile**: TFLite conversion

## Contributors
kvsh2050


