

# 🧠 Federated Brain Tumor Classification at the Edge

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](YOUR_HUGGINGFACE_SPACE_LINK_HERE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Framework: Flower](https://img.shields.io/badge/Federated_Learning-Flower-orange)](https://flower.ai/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.0-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
This repository implements a **Privacy-Preserving Federated Learning (FL)** framework designed to classify brain tumors from MRI scans across decentralized hospital networks. By utilizing the **Flower (`flwr`)** framework, the system trains deep learning models on local edge devices (simulated hospital nodes) and aggregates their learning into a global model *without* ever centralizing sensitive patient data.

Through rigorous architectural testing, an optimized, fine-tuned **EfficientNetB0** emerged as the champion model, achieving **96.14% global accuracy** while operating under strict hardware constraints (NVIDIA RTX 3050 Laptop GPU).



---

## 🚀 Key Innovations & Engineering Highlights
* **Decentralized Privacy:** Successfully implemented **Federated Averaging (FedAvg)** to aggregate model weights from multiple isolated client nodes, proving the viability of zero-trust medical AI.
* **Architectural Optimization (Partial Unfreeze):** Overcame initial gradient explosion ("network blindness") by injecting a custom 0-255 scaling `Lambda` layer and unfreezing the top 20 layers of EfficientNetB0 to adapt specifically to MRI radiological textures.
* **Edge-Device Hardware Tuning:** Managed strict 4GB VRAM limitations on an RTX 3050 by optimizing batch sizes (reduced to `8`) and tuning TensorFloat-32 matrix multiplications, preventing memory allocation crashes during continuous round-training.
* **Interactive Deployment:** Packaged the final inference model into a streamlined **Streamlit** web application, deployed seamlessly to Hugging Face Spaces.

---

## 📊 Empirical Results & Model Comparison

The project benchmarked five distinct CNN architectures over 15 federated training rounds. The fine-tuned EfficientNetB0 demonstrated superior "Compound Scaling," drastically outperforming heavier traditional networks like ResNet50 which suffered from client data drift.

| Model Architecture | Final Global Accuracy | Status / Notes |
| :--- | :--- | :--- |
| **EfficientNetB0 (Fine-Tuned)** | **96.14%** 🏆 | **Champion Model** (High stability, low loss: `0.1071`) |
| **MobileNetV2** | 92.20% | Strong lightweight baseline |
| **VGG16** | 89.22% | Capable but computationally heavy |
| **Custom Micro Edge-Net** | 86.50% | Proof of concept for extreme low-resource edges |
| **ResNet50** | 70.90% | Failed to converge due to non-IID data drift |

### 🔬 Champion Model Classification Report (EfficientNetB0)
Evaluated on a completely unseen Global Test Set.

| Pathology Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Glioma** | 0.95 | 0.97 | 0.96 |
| **Meningioma** | 0.93 | 0.91 | **0.92** |
| **Pituitary** | 0.99 | 0.98 | 0.98 |
| **No Tumor** | 0.97 | 0.98 | 0.98 |

* **Global Weighted Accuracy:** **96%**
* *Note: The model successfully maintained an F1-Score > 0.90 on Meningioma, typically the hardest class to distinguish from Gliomas in clinical settings.*

---

## 📂 Final Project Structure

```text
BRAIN_TUMOR_PROJECT/
├── src/
│   ├── client_node.py         # Federated Learning Client (Hospital Node logic)
│   ├── server_node.py         # Federated Learning Global Server (FedAvg logic)
│   ├── model_architecture.py  # Centralized factory for CNN architectures
│   ├── confusion_matrix.py    # Evaluation script for global metrics
│   ├── data_prep.py           # Automated dataset partitioning for FL nodes
│   └── utils.py               # Shared helper functions
├── models/                    # Contains archived charts and baseline models
├── app.py                     # Streamlit Frontend UI
├── efficientnet_best.h5       # 🏆 Champion Model Weights (96.14%)
├── requirements.txt           # Python environment dependencies
├── packages.txt               # System-level dependencies (Debian/Hugging Face)
└── README.md                  # Project documentation

```

*(Note: Raw patient datasets and federated data splits are ignored via `.gitignore` to comply with data privacy standards).*

---

## ⚙️ How to Run Locally

### 1. Environment Setup

It is highly recommended to run this within a Conda environment to manage TensorFlow dependencies.

```bash
conda create -n fed_tumor python=3.9
conda activate fed_tumor
pip install -r requirements.txt

```

### 2. Launch the Streamlit Web App (Inference Only)

To test the 96.14% model directly without retraining:

```bash
streamlit run app.py

```

### 3. Run the Federated Network (Training)

To simulate the federated network, you must run the server and clients concurrently. Open three separate terminal windows:

* **Terminal 1 (Start the Aggregation Server):**
```bash
python src/server_node.py

```


* **Terminal 2 (Start Hospital Node A):**
```bash
python src/client_node.py 1

```


* **Terminal 3 (Start Hospital Node B):**
```bash
python src/client_node.py 2

```



---

## ⚠️ Disclaimer

This software is intended strictly for **educational and research purposes**. It demonstrates the efficacy of Federated Learning in medical contexts and is **not** an FDA-approved medical device. It should not be used for clinical diagnosis. Always consult a qualified radiologist or medical professional.
