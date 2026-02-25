import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import cv2
import tensorflow as tf

def plot_training_history(history_data, save_path="federated_results.png"):
    """
    Generates and saves the Accuracy/Loss graph.
    """
    df = pd.DataFrame(history_data)
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 6))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    sns.lineplot(x='Round', y='Accuracy', data=df, marker='o', linewidth=3, color='#2ecc71')
    plt.title('Global Model Accuracy', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Round')
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    sns.lineplot(x='Round', y='Loss', data=df, marker='o', linewidth=3, color='#e74c3c')
    plt.title('Global Model Loss', fontsize=14, fontweight='bold')
    plt.ylabel('Loss')
    plt.xlabel('Round')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Graph saved to {save_path}")

def predict_single_image(model_path, test_dir):
    """
    Loads the model and predicts a random image from the test set.
    """
    if not os.path.exists(model_path):
        print("❌ Model not found!")
        return

    model = tf.keras.models.load_model(model_path)
    classes = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
    
    # Pick random image
    random_class = np.random.choice(classes)
    class_path = os.path.join(test_dir, random_class.lower().replace(" ", ""))
    if not os.path.exists(class_path): class_path = os.path.join(test_dir, "notumor")
    
    img_name = np.random.choice(os.listdir(class_path))
    img_path = os.path.join(class_path, img_name)
    
    # Predict
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.array(img_resized, dtype='float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)
    idx = np.argmax(pred)
    
    print(f"🔍 Actual: {random_class} | 🤖 Predicted: {classes[idx]} ({pred[0][idx]*100:.2f}%)")
    return img, classes[idx]