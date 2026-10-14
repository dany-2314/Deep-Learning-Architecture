[readme.md](https://github.com/user-attachments/files/33436081/readme.md)
# Deep Learning Architecture

Coursework, experiments, and research materials for Deep Learning Architecture ([dany-2314/Deep-Learning-Architecture](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdany-2314%2FDeep-Learning-Architecture)). This repository combines practical PyTorch implementations of foundational neural networks with applied research on Vision-Language Models (VLMs) for optical character recognition (OCR) and theoretical LaTeX write-ups on deep learning optimization.

## Repository Contents & Architecture Summary

| Component | Path | Architecture / Focus | Dataset / Input | Key Frameworks & Techniques |
| :--- | :--- | :--- | :--- | :--- |
| **Assignment 1: MNIST DNN** | [A1\_dnn\_mnist\_pytorch.ipynb](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdany-2314%2FDeep-Learning-Architecture%2Fblob%2Fmain%2FA1_dnn_mnist_pytorch.ipynb) | Dense Neural Network (DNN) / Multi-Layer Perceptron (MLP) | MNIST (`28x28` grayscale handwritten digits, 10 classes) | `torch`, `torchvision`, linear layers, ReLU/Sigmoid/Softmax activations, confusion matrices |
| **Assignment 2: CIFAR-10 CNN** | [A2\_cnn\_cifar10\_pytorch.ipynb](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdany-2314%2FDeep-Learning-Architecture%2Fblob%2Fmain%2FA2_cnn_cifar10_pytorch.ipynb) | Convolutional Neural Network (`SimpleCNN`) | CIFAR-10 (`32x32` RGB images, 10 object classes) | `Conv2d`, `BatchNorm2d`, `MaxPool2d`, `AdamW`, `CosineAnnealingLR`, `CrossEntropyLoss`, early stopping |
| **Saved CNN Checkpoint** | [best\_cifar10\_simplecnn\_adamw.pt](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdany-2314%2FDeep-Learning-Architecture%2Fblob%2Fmain%2Fbest_cifar10_simplecnn_adamw.pt) | Pre-trained `SimpleCNN` state dictionary | CIFAR-10 validation best weights | PyTorch model serialization (`torch.save` / `torch.load`) |
| **Course Project: LightOnOCR** | [Project/lightOnOCR.ipynb](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdany-2314%2FDeep-Learning-Architecture%2Fblob%2Fmain%2FProject%2FlightOnOCR.ipynb) | `LightOnOCR` (1B Vision-Language Model) | Document and web images | `transformers` (`LightOnOcrForConditionalGeneration`, `LightOnOcrProcessor`), `easyocr`, structured Markdown/HTML table extraction |
| **Project Proposal** | [Project/topic.txt](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdany-2314%2FDeep-Learning-Architecture%2Fblob%2Fmain%2FProject%2Ftopic.txt) | Applied OCR for RAG QA & Accessibility | Research scope & problem statement | Retrieval-Augmented Generation (RAG) document ingestion and assistive tools for visually impaired users |
| **Participation 2: Neural Foundations** | [participation/p2/](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdany-2314%2FDeep-Learning-Architecture%2Ftree%2Fmain%2Fparticipation%2Fp2) | Perceptrons & Backpropagation | Theoretical derivations (`.tex`, `.pdf`, `.txt`) | Sigmoid, Tanh, ReLU activations, multi-layer forward pass, loss computation, chain rule |
| **Participation 3: Generalization** | [participation/p3/](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdany-2314%2FDeep-Learning-Architecture%2Ftree%2Fmain%2Fparticipation%2Fp3) | Regularization & Optimization | Theoretical derivations (`.tex`, `.pdf`, `.txt`) | L1/L2 weight decay, dropout, data augmentation, early stopping, hyperparameter tuning |

## Directory Structure

```text
Deep-Learning-Architecture/
├── .gitignore
├── A1_dnn_mnist_pytorch.ipynb          # Assignment 1: MLP/DNN digit classification on MNIST
├── A2_cnn_cifar10_pytorch.ipynb        # Assignment 2: SimpleCNN image classification on CIFAR-10
├── best_cifar10_simplecnn_adamw.pt     # Saved PyTorch checkpoint for the best SimpleCNN model
├── requirements.txt                    # Python package dependencies
├── Project/
│   ├── lightOnOCR.ipynb                # Evaluation of the 1B-parameter LightOnOCR VLM
│   └── topic.txt                       # Project proposal: OCR for RAG QA and accessibility
└── participation/
    ├── p2/                             # Write-up on perceptrons, activations, and backpropagation
    │   ├── p2.pdf
    │   ├── p2.tex
    │   └── p2.txt
    └── p3/                             # Write-up on regularization, dropout, and tuning
        ├── p3.txt
        ├── p3_final.pdf
        └── p3_final.tex
```

## Detailed Module Breakdown

### 1. Assignments: Foundational Deep Learning in PyTorch

* **Dense Neural Networks on MNIST ([A1\_dnn\_mnist\_pytorch.ipynb](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdany-2314%2FDeep-Learning-Architecture%2Fblob%2Fmain%2FA1_dnn_mnist_pytorch.ipynb))**: Builds and trains fully connected neural networks from scratch in PyTorch to classify `28x28` grayscale handwritten digits across 10 classes. Explores linear layer topologies, non-linear activation functions (ReLU, Sigmoid, Softmax), training/validation loss and accuracy curves, and test-set confusion matrices.
* **Convolutional Neural Networks on CIFAR-10 ([A2\_cnn\_cifar10\_pytorch.ipynb](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdany-2314%2FDeep-Learning-Architecture%2Fblob%2Fmain%2FA2_cnn_cifar10_pytorch.ipynb))**: Implements a custom `SimpleCNN` architecture featuring 2D convolutional layers (`Conv2d`), batch normalization (`BatchNorm2d`), ReLU activations, max pooling (`MaxPool2d`), and fully connected classification heads. The training pipeline incorporates `CrossEntropyLoss`, the `AdamW` optimizer, `CosineAnnealingLR` learning rate scheduling, and early stopping, exporting the top-performing model weights to [best\_cifar10\_simplecnn\_adamw.pt](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdany-2314%2FDeep-Learning-Architecture%2Fblob%2Fmain%2Fbest_cifar10_simplecnn_adamw.pt).

### 2. Course Project: Vision-Language Models for Structured OCR

* **LightOnOCR Evaluation ([Project/lightOnOCR.ipynb](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdany-2314%2FDeep-Learning-Architecture%2Fblob%2Fmain%2FProject%2FlightOnOCR.ipynb))**: Investigates end-to-end multilingual document understanding using the 1B-parameter `LightOnOCR` Vision-Language Model via Hugging Face `transformers` (`LightOnOcrForConditionalGeneration` and `LightOnOcrProcessor`). Extracts structured text, Markdown tables, and HTML representations directly from document and web images.
* **Research Objectives ([Project/topic.txt](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdany-2314%2FDeep-Learning-Architecture%2Fblob%2Fmain%2FProject%2Ftopic.txt))**: Outlines two primary downstream applications for `LightOnOCR`:
  1. Improving document parsing fidelity for Retrieval-Augmented Generation (RAG) question-answering pipelines.
  2. Supporting accessibility workflows for visually impaired users requiring accurate screen and document reading.

### 3. Theoretical Participation Write-Ups

* **Participation 2 ([participation/p2/](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdany-2314%2FDeep-Learning-Architecture%2Ftree%2Fmain%2Fparticipation%2Fp2))**: Covers single-layer perceptrons, mathematical properties of activation functions (Sigmoid, Tanh, ReLU), multi-layer network forward propagation, loss formulation, and gradient computation via the backpropagation chain rule. Available as plain text (`p2.txt`), LaTeX source (`p2.tex`), and compiled PDF (`p2.pdf`).
* **Participation 3 ([participation/p3/](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdany-2314%2FDeep-Learning-Architecture%2Ftree%2Fmain%2Fparticipation%2Fp3))**: Examines model generalization and optimization strategies, including L1/L2 norm weight decay regularization, dropout, image data augmentation, early stopping criteria, and systematic hyperparameter tuning. Available as plain text (`p3.txt`), LaTeX source (`p3_final.tex`), and compiled PDF (`p3_final.pdf`).

## Getting Started

### Prerequisites

* Python 3.9+
* PyTorch (`torch`, `torchvision`, `torchinfo`)
* Hugging Face `transformers` and `huggingface-hub`
* Scientific and OCR utilities: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `pillow`, `easyocr`
* Jupyter Notebook or JupyterLab (for running `.ipynb` files)
* A TeX distribution such as TeX Live or MiKTeX (optional, for recompiling `.tex` files)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dany-2314/Deep-Learning-Architecture.git
   cd Deep-Learning-Architecture
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required dependencies from [requirements.txt](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdany-2314%2FDeep-Learning-Architecture%2Fblob%2Fmain%2Frequirements.txt):
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Assignment Notebooks

Launch Jupyter to run the MNIST DNN or CIFAR-10 CNN experiments (datasets download automatically via `torchvision.datasets`):

```bash
jupyter notebook A1_dnn_mnist_pytorch.ipynb
jupyter notebook A2_cnn_cifar10_pytorch.ipynb
```

### Loading the Pre-Trained CIFAR-10 Checkpoint

To load the saved `SimpleCNN` weights from [best\_cifar10\_simplecnn\_adamw.pt](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdany-2314%2FDeep-Learning-Architecture%2Fblob%2Fmain%2Fbest_cifar10_simplecnn_adamw.pt) in Python:

```python
import torch

# Instantiate SimpleCNN (as defined in A2_cnn_cifar10_pytorch.ipynb)
checkpoint = torch.load("best_cifar10_simplecnn_adamw.pt", map_location="cpu")
model.load_state_dict(checkpoint)
model.eval()
```

### Running the LightOnOCR Project Notebook

Open the project notebook to test VLM-based OCR extraction on document images:

```bash
jupyter notebook Project/lightOnOCR.ipynb
```

### Compiling the LaTeX Participation Reports

Pre-compiled PDFs are already included in [participation/p2/](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdany-2314%2FDeep-Learning-Architecture%2Ftree%2Fmain%2Fparticipation%2Fp2) and [participation/p3/](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdany-2314%2FDeep-Learning-Architecture%2Ftree%2Fmain%2Fparticipation%2Fp3). To recompile from source:

```bash
cd participation/p2 && pdflatex p2.tex
cd ../p3 && pdflatex p3_final.tex
```
