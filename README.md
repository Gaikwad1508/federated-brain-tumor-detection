# 🧠 Federated Learning for Brain Tumor Classification

![Accuracy](https://img.shields.io/badge/Accuracy-92%%-success)
![Framework](https://img.shields.io/badge/Framework-TensorFlow_2.10-orange)
![FL](https://img.shields.io/badge/Federated_Learning-Flower-blue)

An end-to-end, privacy-preserving AI system that detects brain tumors (Glioma, Meningioma, Pituitary) across multiple simulated hospital nodes without sharing raw patient MRI data.

## 🌟 Key Features
* **Federated Architecture:** Implements a central Server and multiple Client nodes using the Flower (`flwr`) framework.
* **Privacy-Preserving:** Raw MRI scans never leave the local client machines. Only encrypted weight updates are aggregated by the global server.
* **Handling Non-IID Data:** Successfully trained on skewed, non-independent and identically distributed (Non-IID) datasets, proving the model can learn from specialized "hospitals."
* **Optimized for Consumer Hardware:** Architected using a customized **MobileNetV2** backbone with Dropout regularization and tuned learning rates to run locally on an RTX 3050 GPU without memory overflow.
* **Interactive Web App:** Includes a Streamlit GUI for real-time MRI upload and clinical prediction.

## 📊 Model Performance
The global model was evaluated on a held-out, centralized test set of 1,141 MRI images after 15 communication rounds.

* **Peak Global Accuracy:** 92.2%
* **No Tumor Detection (F1-Score):** 96% (High reliability for healthy patients)
* **Pituitary Tumor Detection (F1-Score):** 95%

## 📂 Project Structure
```text
Brain_Tumor_Project/
│
├── federated_dataset/             # Auto-partitioned Non-IID data for clients
├── src/                           # Source Code
│   ├── app.py                     # Streamlit Web GUI
│   ├── client_node.py             # Hospital training logic (Local Epochs)
│   ├── server_node.py             # Global aggregator logic (FedAvg)
│   ├── model_architecture.py      # MobileNetV2 CNN definition
│   ├── confusion_matrix.py        # Evaluation metrics generator
│   └── best_brain_tumor_model.h5  # The final compiled Global Brain
│
├── main.py                        # Automated simulation launcher
└── requirements.txt               # Environment dependencies

```

## 🚀 How to Run

**1. Install Dependencies**

```bash
pip install -r requirements.txt

```

**2. Run the Federated Simulation (Backend)**
To watch the Server and Clients negotiate and train in real-time:

```bash
python main.py

```

**3. Launch the Diagnostic Web App (Frontend)**
To use the trained model for inference:

```bash
cd src
streamlit run app.py

```

## 🛠️ Tech Stack

* **Deep Learning:** TensorFlow / Keras
* **Federated Learning:** Flower (flwr)
* **Computer Vision:** OpenCV (`cv2`)
* **Frontend:** Streamlit
* **Data Visualization:** Matplotlib, Seaborn, Scikit-learn

```

