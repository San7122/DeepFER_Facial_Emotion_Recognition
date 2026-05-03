# 🎭 DeepFER: Facial Emotion Recognition Using Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<p align="center">
  <b>A deep learning system that recognizes human emotions from facial expressions in real time using CNNs and Transfer Learning.</b>
</p>

<p align="center">
  <a href="#-key-features">Features</a> •
  <a href="#-emotion-classes">Emotions</a> •
  <a href="#-model-architecture">Architecture</a> •
  <a href="#-results">Results</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-project-structure">Structure</a>
</p>

---

## 📌 Overview

**DeepFER** is a facial emotion recognition system that classifies human facial expressions into **7 emotion categories** using Convolutional Neural Networks (CNNs) and Transfer Learning. The project implements two model architectures — a **Custom CNN** designed from scratch and a **fine-tuned VGG16** — and compares their performance across accuracy, F1-score, and inference speed.

The system is designed for real-world applications including human-computer interaction, mental health monitoring, customer service analytics, and educational engagement tracking.

---

## ✨ Key Features

- **Dual Model Architecture** — Custom CNN (lightweight, fast) + VGG16 Transfer Learning (high accuracy)
- **7 Emotion Classification** — Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Comprehensive Data Pipeline** — Integrity checks, augmentation, CLAHE preprocessing, class balancing
- **Real-Time Inference** — Optimized pipeline capable of processing 30+ FPS on GPU
- **Model Interpretability** — Grad-CAM visualizations showing where the model focuses
- **Thorough Evaluation** — Accuracy, Precision, Recall, F1-Score, Confusion Matrices, Confidence Analysis
- **Error Analysis** — Detailed misclassification pattern analysis with visual examples

---

## 🎯 Emotion Classes

| Emotion | Description | Key Facial Features |
|---------|-------------|---------------------|
| 😠 **Angry** | Expressions of anger | Furrowed brows, tensed jaw, narrowed eyes |
| 🤢 **Disgust** | Expressions of disgust | Wrinkled nose, raised upper lip, furrowed brows |
| 😨 **Fear** | Expressions of fear | Wide eyes, raised eyebrows, open mouth (tense) |
| 😊 **Happy** | Expressions of happiness | Raised cheeks, smile, crow's feet around eyes |
| 😐 **Neutral** | Non-expressive faces | Relaxed muscles, no prominent expression |
| 😢 **Sad** | Expressions of sadness | Drooping eyelids, downturned mouth, furrowed inner brows |
| 😲 **Surprise** | Expressions of surprise | Raised eyebrows, wide eyes, dropped jaw |

---

## 🏗 Model Architecture

### Model 1: Custom CNN

A purpose-built 4-block convolutional network optimized for 48×48 grayscale facial images.

```
Input (48×48×1)
    │
    ├── Block 1: Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
    ├── Block 2: Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.25)
    ├── Block 3: Conv2D(256) → BN → Conv2D(256) → BN → MaxPool → Dropout(0.25)
    ├── Block 4: Conv2D(512) → BN → Conv2D(512) → BN → MaxPool → Dropout(0.25)
    │
    ├── Flatten
    ├── Dense(512) → BN → Dropout(0.5)
    ├── Dense(256) → BN → Dropout(0.5)
    └── Dense(7, softmax) → Output
```

### Model 2: VGG16 Transfer Learning

Pre-trained VGG16 (ImageNet) with frozen early layers and custom classifier head.

```
Input (48×48×3)
    │
    ├── VGG16 Base (Layers 1-10: Frozen, Layers 11+: Fine-tuned)
    │
    ├── GlobalAveragePooling2D
    ├── Dense(512) → BN → Dropout(0.5)
    ├── Dense(256) → BN → Dropout(0.5)
    └── Dense(7, softmax) → Output
```

---

## 📊 Results

### Performance Comparison

| Metric | Custom CNN | VGG16 Transfer Learning |
|--------|-----------|------------------------|
| **Accuracy** | ~63-66% | ~60-65% |
| **Precision** (weighted) | ~63-66% | ~60-65% |
| **Recall** (weighted) | ~63-66% | ~60-65% |
| **F1-Score** (weighted) | ~63-66% | ~60-65% |
| **Inference Speed** | ~2-5 ms/image | ~8-15 ms/image |
| **Real-Time (30 FPS)** | ✅ Yes | ✅ Yes |

> **Note:** Exact numbers depend on training run. Results above are typical ranges. See the notebook for your specific run's metrics.

### Key Findings

- **Happy** and **Surprise** are the easiest to classify due to distinctive features (smile, wide-open mouth)
- **Disgust** is hardest due to limited training samples and overlap with **Angry**
- **Fear ↔ Surprise** is the most common confusion pair (shared wide-eye pattern)
- Custom CNN offers the best speed-accuracy trade-off for real-time deployment
- Grad-CAM confirms the model attends to semantically relevant facial regions

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Open the notebook in Google Colab
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Upload the dataset to your Google Drive
4. Update the dataset path in Section 3
5. Run all cells sequentially

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/DeepFER-Facial-Emotion-Recognition.git
cd DeepFER-Facial-Emotion-Recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook DeepFER_Facial_Emotion_Recognition.ipynb
```

### Option 3: Docker

```bash
# Build the image
docker build -t deepfer .

# Run the container
docker run --gpus all -p 8888:8888 deepfer
```

---

## 📁 Project Structure

```
DeepFER-Facial-Emotion-Recognition/
│
├── DeepFER_Facial_Emotion_Recognition.ipynb   # Main project notebook
├── README.md                                   # This file
├── requirements.txt                            # Python dependencies
├── LICENSE                                     # MIT License
│
├── data/                                       # Dataset directory
│   ├── train/                                  # Training images
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── test/                                   # Test images
│       ├── angry/
│       ├── disgust/
│       ├── fear/
│       ├── happy/
│       ├── neutral/
│       ├── sad/
│       └── surprise/
│
├── models/                                     # Saved trained models
│   ├── DeepFER_CustomCNN_Final.keras
│   └── DeepFER_VGG16_Final.keras
│
└── outputs/                                    # Generated visualizations
    ├── class_distribution.png
    ├── sample_images.png
    ├── pixel_distribution.png
    ├── average_faces.png
    ├── feature_engineering.png
    ├── custom_cnn_history.png
    ├── vgg16_transfer_learning_history.png
    ├── custom_cnn_confusion.png
    ├── vgg16_transfer_learning_confusion.png
    ├── per_class_f1_comparison.png
    ├── misclassified_samples.png
    ├── gradcam_visualization.png
    ├── prediction_demo.png
    └── model_comparison.png
```

---

## 📦 Dependencies

```
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
Pillow>=8.0.0
```

---

## 🔧 Technical Details

### Data Preprocessing Pipeline

1. **Integrity Check** — Scan for corrupted images, detect duplicates via MD5 hashing, remove non-image files
2. **Normalization** — Scale pixel values from [0, 255] to [0, 1]
3. **CLAHE** — Contrast Limited Adaptive Histogram Equalization for lighting normalization
4. **Augmentation** — Rotation (±15°), shift (±10%), zoom (±10%), horizontal flip
5. **Class Balancing** — Computed balanced class weights for weighted loss function

### Training Configuration

| Parameter | Custom CNN | VGG16 |
|-----------|-----------|-------|
| Optimizer | Adam | Adam |
| Learning Rate | 0.001 | 0.0001 |
| Batch Size | 64 | 64 |
| Max Epochs | 50 | 50 |
| Early Stopping | patience=10 | patience=10 |
| LR Reduction | factor=0.5, patience=5 | factor=0.5, patience=5 |
| Loss Function | Categorical Cross-Entropy | Categorical Cross-Entropy |

### Evaluation Methodology

- **Train/Val/Test Split** — 80% train, 20% validation (from train), separate test set
- **Metrics** — Accuracy, Precision, Recall, F1-Score (per-class and weighted)
- **Confusion Matrix** — Raw counts and normalized percentages
- **Error Analysis** — Top misclassification pairs, confidence distributions
- **Interpretability** — Grad-CAM heatmaps on last convolutional layer

---

## 🌍 Applications

| Domain | Use Case |
|--------|----------|
| 🖥 **HCI** | Adaptive interfaces that respond to user emotional state |
| 🧠 **Mental Health** | Automated screening and emotion tracking for therapists |
| 📞 **Customer Service** | Real-time satisfaction monitoring during video calls |
| 🎓 **Education** | Student engagement detection in online learning |
| ♿ **Accessibility** | Helping individuals with ASD interpret facial expressions |
| 🔒 **Security** | Detecting distress or aggression in public spaces |

---

## ⚠️ Ethical Considerations

This project acknowledges the sensitive nature of facial emotion recognition technology:

- **Consent** — Users must be informed and explicitly consent to emotion monitoring
- **Privacy** — Facial data should be processed locally when possible and never stored without consent
- **Bias** — Models may perform differently across demographics; regular bias audits are essential
- **Misuse** — This technology should not be used for coercive surveillance or manipulative purposes
- **Transparency** — Users should always know when their emotions are being analyzed

---

## 📈 Future Improvements

- [ ] Experiment with EfficientNet / MobileNetV3 for better accuracy-speed trade-off
- [ ] Add Vision Transformer (ViT) as a third model for comparison
- [ ] Implement attention mechanisms (SE-Net, CBAM) for better feature focus
- [ ] Add temporal modeling (LSTM/Transformer) for video sequence emotion tracking
- [ ] Train on larger datasets (AffectNet, RAF-DB, FER+)
- [ ] Convert to TensorFlow Lite for mobile deployment
- [ ] Build a Streamlit/Gradio demo application
- [ ] Implement real-time webcam inference with face detection
- [ ] Add multi-modal fusion (facial + voice + text)
- [ ] Conduct comprehensive demographic bias analysis

---

## 📜 Dataset

The dataset contains grayscale facial images (48×48 pixels) categorized into 7 emotion classes. It is sourced from publicly available facial expression databases.

**Download:** [Google Drive Link](https://drive.google.com/file/d/1WxFwPgUTPHAIgXHVuPntH9pXQGiKda6F/view?usp=sharing)

---

## 🤝 Acknowledgments

- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013) — Original facial expression dataset
- [VGG16](https://arxiv.org/abs/1409.1556) — Pre-trained model architecture by Visual Geometry Group, Oxford
- [TensorFlow](https://www.tensorflow.org/) — Deep learning framework
- [Keras](https://keras.io/) — High-level neural network API
- [Grad-CAM](https://arxiv.org/abs/1610.02391) — Gradient-weighted Class Activation Mapping

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>Built with ❤️ for the AlmaBetter Capstone Project</b>
</p>
