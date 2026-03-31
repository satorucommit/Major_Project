<<<<<<< HEAD
# Major_Project


## AI-Powered Sign Language Translation System with Real time Gesture Recognition

### Project Overview

An AI-based assistive technology designed to translate sign language gestures into text and text into sign animations in real time. The system helps bridge the communication gap between deaf and hearing individuals using computer vision and deep learning.

The project uses hand landmark detection and machine learning models to recognize gestures from live webcam input and convert them into readable text.

### Objectives

Detect hand landmarks using MediaPipe to extract 21 key points from live webcam input.

Develop a sign-to-text gesture recognition system using deep learning.

Create a text-to-sign animation generator to display sign language through 3D avatars.

Build a searchable dictionary containing 500–1000 common sign gestures with video demonstrations.

Design an interactive learning module with practice mode and real-time feedback.

Implement user authentication and progress tracking for personalized learning.
=======
# Text-to-Sign Translation Model

A complete system for converting text input into skeleton-based sign language animation videos. Optimized for low-resource hardware (4GB VRAM + 16GB RAM) with checkpoint-based training for overnight training sessions.

## 🎯 Features

- **5-Stage Pipeline**: Text → Gloss → Pose → Refinement → Skeleton Video
- **Memory Optimized**: Works on 4GB VRAM with gradient accumulation
- **Night Training**: Automatic training schedule (10 PM - 5 AM)
- **Auto-Resume**: Checkpoints every 30 minutes with automatic resumption
- **Mixed Precision**: FP16 training for memory efficiency
- **Multi-format Output**: NPY skeleton data, MP4 video, optional GIF

## 📁 Project Structure

```
sign_language_project/
├── configs/
│   └── config.py           # All configuration settings
├── models/
│   └── text_to_sign_model.py  # 5-stage model architecture
├── data/
│   ├── dataset.py          # Memory-efficient dataset
│   └── organized_classes/  # Your data goes here
├── utils/
│   ├── checkpoint.py       # Checkpoint & resume system
│   ├── training.py         # Training loop with mixed precision
│   ├── inference.py        # Inference pipeline
│   ├── visualization.py    # Skeleton video rendering
│   └── helpers.py          # Utility functions
├── train.py                # Main training script
├── inference.py            # Inference script
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Sample Data

```bash
python train.py --setup_sample_data --num_sample_classes 20
```

### 3. Train the Model

```bash
# Start training
python train.py --epochs 100 --batch_size 2

# Resume from checkpoint
python train.py --resume
```

### 4. Run Inference

```bash
# Single text translation
python inference.py --text "Hello, how are you?"

# Interactive mode
python inference.py --interactive

# Batch mode
python inference.py --batch texts.txt outputs/
```

## 📊 Model Architecture

### Stage 1: Text Encoder
- Lightweight transformer encoder
- Multilingual support (English, Hindi, Spanish)
- DistilBERT-style architecture for memory efficiency

### Stage 2: Text-to-Gloss Translation
- Seq2Seq transformer
- Handles sign language grammar differences
- Beam search for inference

### Stage 3: Gloss-to-Pose Generation
- Motion transformer
- Generates 543 keypoints (body, hands, face)
- Temporal attention for smooth motion

### Stage 4: Pose Refinement
- Spatio-Temporal Graph Convolutional Network (ST-GCN)
- Applies kinematic constraints
- Temporal smoothing

### Stage 5: Skeleton Video Output
- OpenCV-based rendering
- MediaPipe skeleton structure
- MP4 video output

## ⚙️ Configuration

Key training parameters in `configs/config.py`:

```python
TRAINING_CONFIG = {
    "batch_size": 2,                    # Small batch for 4GB VRAM
    "gradient_accumulation_steps": 8,   # Effective batch = 16
    "mixed_precision": True,            # FP16 training
    "max_frames": 60,                   # 2.4 seconds at 25fps
    
    # Checkpointing
    "checkpoint_frequency": 30,         # Every 30 minutes
    "training_start_time": "22:00",     # 10 PM
    "training_end_time": "05:00",       # 5 AM
}
```

## 📈 Training Schedule

The system is designed for overnight training:

| Time | Action |
|------|--------|
| 10:00 PM | Training starts automatically |
| Every 30 min | Checkpoint saved |
| 4:55 AM | Graceful shutdown, final checkpoint saved |
| Next night | Auto-resume from last checkpoint |

## 📝 Data Format

### Directory Structure
```
data/organized_classes/
├── class_001_hello/
│   ├── regular/
│   │   ├── video_001.npy
│   │   └── video_002.npy
│   ├── augmented/
│   │   └── aug_video_001.npy
│   └── metadata.json
├── class_002_thank_you/
│   └── ...
```

### Metadata Format
```json
{
  "class_name": "hello",
  "class_id": 1,
  "text_labels": ["hello", "hi", "greetings"],
  "gloss": "HELLO",
  "videos": [
    {
      "filename": "video_001.npy",
      "duration_frames": 60,
      "fps": 25
    }
  ]
}
```

### Skeleton Data Format
- Shape: `[frames, 543, 3]`
- 543 keypoints: 33 body + 21×2 hands + 468 face
- 3 coordinates: x, y, z (normalized 0-1)

## 🔧 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4 GB | 8 GB |
| RAM | 16 GB | 32 GB |
| Storage | 10 GB | 50 GB |
| Python | 3.8+ | 3.10+ |

## 📦 Output

The system generates:

1. **Skeleton Data (.npy)**: Raw keypoint coordinates
2. **Video (.mp4)**: Animated skeleton visualization
3. **Optional GIF**: For quick preview

Example usage:
```python
from utils.inference import TextToSignInference
from utils.visualization import create_skeleton_video

# Initialize
inference = TextToSignInference(model_path="checkpoints/best_model.pth")

# Translate
result = inference.translate("Hello, how are you?")

# Save video
create_skeleton_video(result['skeleton'], "output.mp4")
```

## 🎓 Training Tips

1. **Start Small**: Use 20 classes, 10 videos each for initial training
2. **Monitor VRAM**: Check `torch.cuda.memory_allocated()` during training
3. **Gradient Accumulation**: Increase `gradient_accumulation_steps` for stability
4. **Early Stopping**: Patience of 15 epochs prevents overfitting
5. **Learning Rate**: Default 3e-4 works well; reduce if unstable

## 🐛 Troubleshooting

### Out of Memory
- Reduce `batch_size` to 1
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing`

### Slow Training
- Reduce `num_workers` in DataLoader
- Use smaller `max_frames`
- Disable augmentation

### Poor Quality Output
- Train for more epochs
- Increase model capacity
- Add more training data

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- MediaPipe for skeleton structure
- PyTorch for deep learning framework
- OpenCV for video processing
>>>>>>> 7dfeda3 (Initial code)
