# 🎨 ControlNet Face Generator

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Diffusers](https://img.shields.io/badge/🤗-Diffusers-yellow.svg)

*A powerful ControlNet-based system for parametric face generation using semantic segmentation masks*

[Features](#-features) • [Quick Start](#-quick-start-workflow) • [Installation](#-installation) • [Dataset Preparation](#-dataset-preparation) • [Training](#-training) • [Interface](#-interface) • [Architecture](#-architecture)

</div>

---

## 📋 Overview

This project implements a **ControlNet model** fine-tuned on facial semantic segmentation masks, enabling precise control over individual facial features during face generation. By combining multiple facial components (eyes, nose, lips, hair, etc.), you can create entirely new synthetic faces with parametric control over position, scale, and composition.

### Complete Workflow

```
Face Images → BiSeNet Segmentation → Feature Masks → ControlNet Training → Interactive Generation
```

1. **Start with face images** from any source (CelebA, FFHQ, custom dataset)
2. **Generate masks** using [BiSeNet Face Parsing](https://github.com/yakhyo/face-parsing)
3. **Train ControlNet** on face-mask pairs
4. **Generate new faces** by composing features interactively

### What Makes This Special?

- 🎯 **Precise Feature Control**: Manipulate 8 distinct facial features independently
- 🔄 **Real-time Generation**: Interactive GUI with instant visual feedback
- 🎨 **Parametric Design**: Adjust position, scale, and aspect ratio of each feature
- 🚀 **Production Ready**: Built on Stable Diffusion v1.5 with proven architecture
- 📊 **Complete Pipeline**: From raw images to trained model with dataset creation guide

---

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **8 Facial Features** | Right/Left Eyebrow, Right/Left Eye, Nose, Upper/Lower Lip, Hair |
| **Interactive GUI** | Gradio-based interface for intuitive feature composition |
| **Real-time Preview** | See your composition instantly before generation |
| **Flexible Training** | Support for EMA, gradient checkpointing, mixed precision |
| **Dataset Scanning** | Automatic indexing of available facial feature masks |
| **Checkpoint Management** | Regular saving and resumption of training |

### Facial Features Supported

```
Feature ID  │  Name           │  Control
────────────┼─────────────────┼──────────────────────
    2       │  Right Eyebrow  │  Position, Scale, Aspect
    3       │  Left Eyebrow   │  Position, Scale, Aspect
    4       │  Right Eye      │  Position, Scale, Aspect
    5       │  Left Eye       │  Position, Scale, Aspect
    10      │  Nose           │  Position, Scale, Aspect
    11      │  Lower Lip      │  Position, Scale, Aspect
    12      │  Upper Lip      │  Position, Scale, Aspect
    17      │  Hair           │  Position, Scale, Aspect
```

---

## 🚀 Quick Start Workflow

Follow these three simple steps to go from raw face images to generating new faces:

### Step 1: Prepare Your Dataset
```bash
# Clone the face parsing repository
git clone https://github.com/yakhyo/face-parsing.git
cd face-parsing

# Download the pre-trained BiSeNet model
wget https://github.com/yakhyo/face-parsing/releases/download/v0.0.1/resnet34.pt

# Generate segmentation masks from your face images
python inference.py \
  --model resnet34 \
  --weight ./resnet34.pt \
  --input /path/to/your/face/images \
  --output /path/to/output/masks
```

See the [Dataset Preparation](#-dataset-preparation) section for detailed instructions on organizing the generated masks.

### Step 2: Train ControlNet
```bash
cd /path/to/controlnet-face-generator
accelerate launch train.py \
  --data_root ./your_dataset \
  --output_dir ./trained_model \
  --batch_size 4 \
  --max_steps 50000 \
  --validation_every 1000
```

### Step 3: Generate Faces Interactively
```bash
python gui-inference.py \
  --controlnet_dir ./trained_model/final_model \
  --share  # Optional: create public link
```

Open your browser and start composing faces!

---

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 16GB+ VRAM for training, 8GB+ for inference

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/controlnet-face-generator.git
cd controlnet-face-generator
```

2. **Install dependencies**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate safetensors
pip install gradio opencv-python pillow matplotlib tqdm
pip install bitsandbytes  # Optional: for 8-bit Adam optimizer
```

3. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 📂 Dataset Preparation

### Creating Your Dataset

This project requires a dataset of face images with corresponding semantic segmentation masks. We used the **[BiSeNet Face Parsing](https://github.com/yakhyo/face-parsing)** model to automatically generate facial feature masks from a collection of face images.

#### Step 1: Generate Masks from Face Images

1. **Obtain a face image dataset** (e.g., CelebA, FFHQ, or your custom collection)
2. **Use BiSeNet to generate masks**:
   ```bash
   # Clone the face parsing repository
   git clone https://github.com/yakhyo/face-parsing.git
   cd face-parsing
   
   # Download the pre-trained model
   wget https://github.com/yakhyo/face-parsing/releases/download/v0.0.1/resnet34.pt
   
   # Run inference on your face images
   python inference.py \
     --model resnet34 \
     --weight ./resnet34.pt \
     --input /path/to/your/face/images \
     --output /path/to/output/masks
   ```

3. **Extract individual feature masks**: The BiSeNet model outputs 19 different facial regions. You'll need to extract the 8 specific features used in this project.

#### Step 2: Dataset Structure

Organize your generated dataset as follows:

```
your_dataset/
├── input/
│   ├── person1.png
│   ├── person2.png
│   └── ...
└── output/
    └── resnet34/
        ├── person1/
        │   ├── 2.png    # Right eyebrow mask
        │   ├── 3.png    # Left eyebrow mask
        │   ├── 4.png    # Right eye mask
        │   ├── 5.png    # Left eye mask
        │   ├── 10.png   # Nose mask
        │   ├── 11.png   # Lower lip mask
        │   ├── 12.png   # Upper lip mask
        │   └── 17.png   # Hair mask
        └── person2/
            ├── 2.png
            └── ...
```

**Requirements:**
- Input images: RGB face images (`.png`)
- Masks: Single-channel grayscale images (`.png`)
- All 8 required masks must be present for each face
- Images are automatically resized to 512×512 during training

#### BiSeNet Face Parsing Label Map

The BiSeNet model segments faces into 19 classes. Here's the mapping:

| Label ID | Feature | Used in Training |
|----------|---------|------------------|
| 0 | Background | ❌ |
| 1 | Skin | ❌ |
| 2 | Left Eyebrow | ✅ (Feature 3) |
| 3 | Right Eyebrow | ✅ (Feature 2) |
| 4 | Left Eye | ✅ (Feature 5) |
| 5 | Right Eye | ✅ (Feature 4) |
| 6 | Glasses | ❌ |
| 7 | Left Ear | ❌ |
| 8 | Right Ear | ❌ |
| 9 | Earring | ❌ |
| 10 | Nose | ✅ (Feature 10) |
| 11 | Mouth Interior | ❌ |
| 12 | Upper Lip | ✅ (Feature 12) |
| 13 | Lower Lip | ✅ (Feature 11) |
| 14 | Neck | ❌ |
| 15 | Necklace | ❌ |
| 16 | Cloth | ❌ |
| 17 | Hair | ✅ (Feature 17) |
| 18 | Hat | ❌ |

---

## 🎓 Training

### Basic Training

```bash
accelerate launch train.py \
  --data_root ./your_dataset \
  --output_dir ./trained_model \
  --batch_size 4 \
  --max_steps 50000 \
  --validation_every 1000
```

### Advanced Training Options

```bash
accelerate launch train.py \
  --data_root ./your_dataset \
  --output_dir ./trained_model \
  --base_model runwayml/stable-diffusion-v1-5 \
  --batch_size 8 \
  --num_workers 4 \
  --max_steps 100000 \
  --lr 1e-5 \
  --lr_scheduler cosine \
  --warmup_steps 1000 \
  --use_ema \
  --ema_decay 0.9999 \
  --use_8bit_adam \
  --use_gradient_checkpointing \
  --mixed_precision fp16 \
  --save_checkpoint_every 2500 \
  --validation_every 500
```

### Resume Training

```bash
accelerate launch train.py \
  --data_root ./your_dataset \
  --output_dir ./trained_model \
  --resume_from_checkpoint ./trained_model/checkpoint_step_10000 \
  --batch_size 4
```

### Continue Training from Existing ControlNet

```bash
accelerate launch train.py \
  --data_root ./your_dataset \
  --output_dir ./fine_tuned_model \
  --load_controlnet_from ./trained_model/final_model \
  --batch_size 4
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_root` | *required* | Root directory of your dataset |
| `--output_dir` | *required* | Where to save checkpoints and logs |
| `--base_model` | `runwayml/stable-diffusion-v1-5` | Base Stable Diffusion model |
| `--batch_size` | `4` | Training batch size |
| `--max_steps` | `50000` | Total training steps |
| `--lr` | `1e-5` | Learning rate |
| `--use_ema` | `False` | Use Exponential Moving Average |
| `--use_8bit_adam` | `False` | Use memory-efficient 8-bit Adam |
| `--mixed_precision` | `fp16` | Mixed precision: `no`, `fp16`, or `bf16` |

---

## 🖥️ Interactive Interface

### Launch the GUI

```bash
python gui-inference.py \
  --controlnet_dir ./trained_model/final_model \
  --share  # Optional: create public link
```

### GUI Features

The interface provides three main sections:

#### 1️⃣ Dataset Loading
- Specify your dataset path
- Automatically scans and indexes all available facial feature masks
- Displays up to 50 random samples per feature (configurable)

#### 2️⃣ Feature Selection & Composition
For each facial feature:
- **Gallery**: Browse available masks from your dataset
- **Preview**: See the selected mask
- **X/Y Offset**: Position the feature (-200 to +200 pixels)
- **Scale**: Resize the feature (0.1× to 3.0×)
- **Aspect Ratio**: Adjust width/height ratio (0.2 to 2.0)

#### 3️⃣ Generation
- **Composition Canvas**: Real-time preview of combined masks
- **Generated Face**: Final synthesized face
- **Settings**:
  - Real-time inference toggle
  - Seed control (reproducibility)
  - Inference steps (1-100)
  - ControlNet scale (0.1-2.0)

### Usage Tips

1. **Start Simple**: Select one feature at a time to understand its impact
2. **Use Real-time Mode**: For instant feedback (requires good GPU)
3. **Experiment with Scale**: Small adjustments can create dramatic differences
4. **Seed Control**: Use same seed for reproducible results
5. **ControlNet Scale**: Higher values = stronger adherence to masks

---

## 🏗️ Architecture

### Training Pipeline

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Face RGB  │────────▶│  VAE Encoder │────────▶│   Latents   │
└─────────────┘         └──────────────┘         └─────────────┘
                                                         │
                                                         ▼
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│8-ch Masks   │────────▶│  ControlNet  │────────▶│ Conditioning│
└─────────────┘         └──────────────┘         └─────────────┘
                                                         │
                                                         ▼
                                              ┌──────────────────┐
                                              │   UNet + Noise   │
                                              └──────────────────┘
                                                         │
                                                         ▼
                                              ┌──────────────────┐
                                              │   MSE Loss       │
                                              └──────────────────┘
```

### Inference Pipeline

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│User Selects │────────▶│  Transform & │────────▶│8-ch Control │
│  Features   │         │  Composite   │         │   Tensor    │
└─────────────┘         └──────────────┘         └─────────────┘
                                                         │
                                                         ▼
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│Random Noise │────────▶│  ControlNet  │────────▶│Conditioned  │
│  Latents    │         │   + UNet     │         │  Latents    │
└─────────────┘         └──────────────┘         └─────────────┘
                                                         │
                                                         ▼
                                              ┌──────────────────┐
                                              │  VAE Decoder     │
                                              └──────────────────┘
                                                         │
                                                         ▼
                                              ┌──────────────────┐
                                              │ Generated Face   │
                                              └──────────────────┘
```

### Technical Details

- **Base Model**: Stable Diffusion v1.5 (runwayml)
- **ControlNet**: 8-channel input (one per facial feature)
- **Resolution**: 512×512 pixels
- **Latent Space**: 4-channel, 64×64
- **Noise Scheduler**: DDPM (Denoising Diffusion Probabilistic Models)
- **Loss Function**: Mean Squared Error (MSE) in latent space

---

## 📊 Monitoring Training

### TensorBoard

Training automatically logs to TensorBoard:

```bash
tensorboard --logdir ./trained_model/logs
```

### Loss Plot

A loss plot is automatically generated at:
```
./trained_model/loss_plot.png
```

The plot shows:
- **Training Loss**: Exponentially smoothed (α=0.9)
- **Validation Loss**: Evaluated every N steps
- Both plotted against training steps

### Checkpoint Structure

```
trained_model/
├── checkpoint_step_2500/
│   ├── optimizer.bin
│   ├── scheduler.bin
│   └── pytorch_model.bin
├── checkpoint_step_5000/
├── logs/
│   └── tensorboard events
├── loss_plot.png
└── final_model/
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

---

## 🔧 Troubleshooting

### Common Issues

**Issue**: Out of memory during training
```bash
# Solutions:
--batch_size 1
--use_gradient_checkpointing
--mixed_precision fp16
--use_8bit_adam
```

**Issue**: Gallery selection not working in GUI
- This version includes fixes for Gradio's gallery selection
- Ensure you're using Gradio 3.x or 4.x

**Issue**: Models not loading
- Verify ControlNet path is correct
- Check CUDA/GPU availability
- Ensure all dependencies are installed

**Issue**: No valid image-mask pairs found
- Verify dataset structure matches the expected format
- Ensure all 8 mask files exist for each face
- Check file naming: masks should be `{index}.png`

### Performance Optimization

| Goal | Strategy |
|------|----------|
| Faster Training | Increase `batch_size`, use `fp16`, add `num_workers` |
| Less Memory | Enable `gradient_checkpointing`, use `8bit_adam` |
| Better Quality | Increase `max_steps`, enable `use_ema` |
| Faster Convergence | Use `cosine` scheduler, tune `lr` |

---

## 📖 Code Structure

### `train.py` (Training Script)

**Key Components:**
- `FaceMaskDataset`: Custom dataset loader with train/val split
- `encode_latents()`: VAE encoding for diffusion training
- `train()`: Main training loop with validation and checkpointing
- Multi-channel mask stacking (8 features → 8 channels)
- EMA support for stable convergence

**Training Features:**
- Automatic train/validation split (90/10)
- Exponential moving average (EMA) of weights
- Gradient checkpointing for memory efficiency
- 8-bit Adam optimizer support
- Mixed precision training (fp16/bf16)
- Regular checkpointing and validation

### `gui-inference.py` (Interactive Interface)

**Key Components:**
- `load_models()`: One-time model loading to VRAM
- `scan_dataset()`: Indexes available masks from dataset
- `transform_mask()`: Applies user transformations to masks
- `update_and_infer()`: Unified preview + generation function
- `on_gallery_select()`: Robust gallery selection handler

**GUI Features:**
- Lazy model loading (loads in background)
- Real-time inference toggle
- Per-feature transformation controls
- Composition canvas preview
- Reproducible generation with seed control

---

## 🎯 Use Cases

### Creative Applications
- 🎭 **Character Design**: Create unique faces for games, animation, or storytelling
- 🖼️ **Digital Art**: Generate base faces for digital painting and illustration
- 📱 **Avatar Generation**: Create personalized avatars with specific features

### Research & Development
- 🔬 **Face Synthesis Research**: Study controllable face generation
- 📊 **Dataset Augmentation**: Generate synthetic training data
- 🧪 **Feature Analysis**: Understand facial feature contributions

### Education & Training
- 📚 **ML Demonstrations**: Teach ControlNet and diffusion models
- 🎓 **Face Perception Studies**: Create stimuli for psychology research
- 🧑‍🏫 **Computer Vision**: Demonstrate semantic segmentation applications

---

## 🛣️ Roadmap

- [ ] Support for additional facial features (ears, facial hair, accessories)
- [ ] Style transfer integration
- [ ] Age and expression control
- [ ] Batch generation mode
- [ ] Prompt-based refinement
- [ ] Export to various formats (PNG, JPEG, vector)
- [ ] API endpoint for programmatic access
- [ ] Pre-trained model release

---

## 📚 References

### Papers
- **ControlNet**: [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)
- **Stable Diffusion**: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- **DDPM**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- **BiSeNet**: [Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897)

### Libraries & Tools
- [🤗 Diffusers](https://github.com/huggingface/diffusers) - Diffusion models library
- [Accelerate](https://github.com/huggingface/accelerate) - Distributed training
- [Gradio](https://gradio.app/) - Interactive ML interfaces
- [BiSeNet Face Parsing](https://github.com/yakhyo/face-parsing) - Facial segmentation for mask generation

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report Bugs**: Open an issue with details and reproduction steps
2. **Suggest Features**: Share your ideas for improvements
3. **Submit PRs**: Fix bugs or add features (please discuss first)
4. **Share Results**: Show us what you've created!

### Development Setup

```bash
git clone https://github.com/yourusername/controlnet-face-generator.git
cd controlnet-face-generator
pip install -e .
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Model Licenses
- **Stable Diffusion v1.5**: [CreativeML Open RAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
- Please respect the licenses of any base models you use

---

## 🙏 Acknowledgments

- **Stability AI** for Stable Diffusion
- **Hugging Face** for the Diffusers library
- **Lvmin Zhang** for the ControlNet architecture
- **[yakhyo/face-parsing](https://github.com/yakhyo/face-parsing)** - BiSeNet face parsing model used for generating facial feature masks
- **The ML Community** for continuous inspiration and support

### Dataset Creation
This project uses facial segmentation masks generated by the **BiSeNet (Bilateral Segmentation Network)** model. Special thanks to [@yakhyo](https://github.com/yakhyo) for providing the well-maintained [face-parsing](https://github.com/yakhyo/face-parsing) repository with pre-trained ResNet18 and ResNet34 models, which made it easy to generate high-quality facial feature masks from our face image dataset.

**BiSeNet Paper**: [Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897)

---

## 📬 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/controlnet-face-generator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/controlnet-face-generator/discussions)

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

Made with ❤️ by [Your Name]

</div>
