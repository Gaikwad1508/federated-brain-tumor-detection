import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Run prediction on CPU
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Load the Trained Model
MODEL_PATH = "best_brain_tumor_model.h5"
if not os.path.exists(MODEL_PATH):
    print("❌ Error: Model file not found. Did you run the training?")
    exit()

print(f"📂 Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
classes = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

# 2. Pick a Random Image from the Test Set
test_dir = "../federated_dataset/global_test_set"
random_class = np.random.choice(classes)
class_path = os.path.join(test_dir, random_class.lower().replace(" ", ""))

# Handle folder naming differences (notumor vs no_tumor)
if not os.path.exists(class_path):
    class_path = os.path.join(test_dir, "notumor")

images = os.listdir(class_path)
random_image = np.random.choice(images)
img_path = os.path.join(class_path, random_image)

# 3. Preprocess the Image
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (224, 224))
img_array = np.array(img_resized, dtype='float32') / 255.0
img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

# 4. Predict
prediction = model.predict(img_array)
predicted_class_idx = np.argmax(prediction)
predicted_label = classes[predicted_class_idx]
confidence = prediction[0][predicted_class_idx] * 100

# 5. Show Result
print(f"\n🔍 True Label: {random_class}")
print(f"🤖 Model Prediction: {predicted_label} ({confidence:.2f}%)")

plt.imshow(img_rgb)
plt.title(f"True: {random_class} | Pred: {predicted_label} ({confidence:.1f}%)")
plt.axis('off')
plt.show()