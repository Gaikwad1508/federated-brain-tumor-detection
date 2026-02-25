import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Load the newly trained smart model
MODEL_PATH = "best_brain_tumor_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
classes = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
test_dir = "../federated_dataset/global_test_set"

y_true = []
y_pred = []

print("🔍 Evaluating the Global Test Set... (This takes about 30-60 seconds)")

for true_idx, true_label in enumerate(classes):
    # Handle folder naming
    folder_name = true_label.lower().replace(" ", "")
    if not os.path.exists(os.path.join(test_dir, folder_name)):
        folder_name = "notumor" if "no" in folder_name else folder_name
    
    class_path = os.path.join(test_dir, folder_name)
    
    if os.path.exists(class_path):
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Preprocess
            img_resized = cv2.resize(img, (224, 224))
            img_array = np.array(img_resized, dtype='float32') / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            
            # Predict
            prediction = model.predict(img_batch, verbose=0)
            pred_idx = np.argmax(prediction)
            
            y_true.append(true_idx)
            y_pred.append(pred_idx)

# Generate Plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Federated Model - Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual True Diagnosis', fontsize=12)
plt.xlabel('AI Predicted Diagnosis', fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
print("\n✅ Matrix saved as 'confusion_matrix.png'")

print("\n📊 Detailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=classes))

plt.show()