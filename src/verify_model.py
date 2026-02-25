import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # CPU Mode
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Load the Model
MODEL_PATH = "best_brain_tumor_model.h5"
if not os.path.exists(MODEL_PATH):
    print("❌ Error: Model file not found!")
    exit()

print(f"📂 Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
classes = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
test_dir = "../federated_dataset/global_test_set"

# 2. Loop through EACH class to test them all
print("\n🔍 STARTING COMPREHENSIVE TEST (One of each type)...\n")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for i, true_label in enumerate(classes):
    # Find the folder for this specific tumor type
    # (Handles "no tumor" vs "notumor" folder naming)
    folder_name = true_label.lower().replace(" ", "")
    if not os.path.exists(os.path.join(test_dir, folder_name)):
        folder_name = "notumor" if "no" in folder_name else folder_name
    
    class_path = os.path.join(test_dir, folder_name)
    
    # Pick a random image from this folder
    if os.path.exists(class_path) and len(os.listdir(class_path)) > 0:
        img_name = np.random.choice(os.listdir(class_path))
        img_path = os.path.join(class_path, img_name)
        
        # Read & Preprocess (Standard OpenCV BGR)
        img = cv2.imread(img_path)
        img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # For Matplotlib only
        img_resized = cv2.resize(img, (224, 224))
        img_array = np.array(img_resized, dtype='float32') / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_batch, verbose=0)
        pred_idx = np.argmax(prediction)
        pred_label = classes[pred_idx]
        confidence = prediction[0][pred_idx] * 100
        
        # Print Result
        status = "✅ PASS" if pred_label == true_label else "❌ FAIL"
        print(f"Test {i+1}: True=[{true_label}] --> Pred=[{pred_label}] ({confidence:.2f}%) {status}")
        
        # Show image in the plot
        axes[i].imshow(img_display)
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", color=("green" if status == "✅ PASS" else "red"))
        axes[i].axis('off')

print("\n--- End of Test ---")
plt.tight_layout()
plt.show()