import os
# 🛡️ Safe CPU Mode: Prevents evaluation from fighting with other open apps for GPU memory
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from model_architecture import get_model 

# 🏆 THE CHAMPION: Pointing to the final EfficientNet weights
MODEL_PATH = "../efficientnet_best.h5"

# 🏗️ Build the architecture and load the high-performance weights
print("🏗️ Building EfficientNetB0 architecture...")
model = get_model("efficientnet")

# Load weights (using load_weights because we saved in .h5 weights-only format)
model.load_weights(MODEL_PATH)

# Define our 4 medical categories
classes = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
test_dir = "../federated_dataset/global_test_set"

y_true = []
y_pred = []

print(f"🔍 Evaluating {MODEL_PATH} on the Global Test Set...")

# Loop through each folder in the test set
for true_idx, class_name in enumerate(classes):
    # Standardize folder naming for the script
    folder_name = class_name.lower().replace(" ", "")
    if not os.path.exists(os.path.join(test_dir, folder_name)):
        folder_name = "notumor" if "no" in folder_name else folder_name
    
    class_path = os.path.join(test_dir, folder_name)
    
    if os.path.exists(class_path):
        print(f"📂 Processing category: {class_name}...")
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            
            # 👁️ PREPROCESS: Resize and normalize (0.0 to 1.0)
            # The model's internal Lambda layer will rescale this back to 0-255 automatically
            img_resized = cv2.resize(img, (224, 224))
            img_array = np.array(img_resized, dtype='float32') / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            
            # 🧠 AI PREDICTION
            prediction = model.predict(img_batch, verbose=0)
            pred_idx = np.argmax(prediction)
            
            y_true.append(true_idx)
            y_pred.append(pred_idx)

# --- VISUALIZATION: CONFUSION MATRIX ---
# This shows exactly where the AI is making mistakes (e.g., confusing Glioma for Meningioma)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))

# 🎨 THE CHAMPION THEME: Professional "Blues" for the final report
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes,
            annot_kws={"size": 14, "weight": "bold"}) 

plt.title('EfficientNetB0 Federated Model - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Actual Patient Diagnosis', fontsize=12, fontweight='bold')
plt.xlabel('AI Predicted Diagnosis', fontsize=12, fontweight='bold')
plt.tight_layout()

# 💾 Save the high-resolution evidence for the research paper
plt.savefig("efficientnet_final_cm.png", dpi=300)
print("\n✅ Evidence saved as 'efficientnet_final_cm.png'")

# --- STATISTICAL ANALYSIS ---
print("\n📊 Detailed Classification Report (The Final Results):")
# This provides Precision, Recall, and F1-Score for every class
print(classification_report(y_true, y_pred, target_names=classes))

plt.show()