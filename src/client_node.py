import os
import sys
import cv2
import numpy as np
import flwr as fl
import tensorflow as tf
from model_architecture import get_model

# 🛑 CRITICAL: Prevent GPU Memory Hoarding on the RTX 3050
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Tell TensorFlow to only use the GPU memory it strictly needs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def load_data(client_id, base_dir="../federated_dataset"):
    """Loads and preprocesses the MRI images for a specific hospital (client)."""
    client_path = os.path.join(base_dir, client_id)
    classes = ['glioma', 'meningioma', 'pituitary', 'notumor']
    
    images = []
    labels = []
    
    print(f"📂 Loading data for {client_id}...")
    
    for label_idx, cls in enumerate(classes):
        cls_path = os.path.join(client_path, cls)
        if not os.path.exists(cls_path):
            continue
            
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            try:
                # Read, resize to 224x224, and store
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))
                images.append(img)
                labels.append(label_idx)
            except Exception as e:
                print(f"⚠️ Error loading {img_name}: {e}")

    # Convert to Numpy Arrays and Normalize pixel values between 0 and 1
    x_train = np.array(images, dtype='float32') / 255.0
    y_train = np.array(labels, dtype='int32')
    
    print(f"✅ Loaded {len(x_train)} images for {client_id}.")
    return x_train, y_train

class TumorClient(fl.client.NumPyClient):
    """The Federated Learning Client that represents a single Hospital."""
    def __init__(self, model_name, client_id):
        self.model = get_model(model_name)
        self.x_train, self.y_train = load_data(client_id)

    def get_parameters(self, config):
        """Sends the local model weights to the central server."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Receives global weights, trains on local MRI data, and returns updated weights."""
        self.model.set_weights(parameters)
        
        # Train locally for 3 epochs
        history = self.model.fit(
            self.x_train, self.y_train, 
            epochs=3, 
            batch_size=16, # 🚀 Increased to 16 for EfficientNet to speed up training
            shuffle=True,  # 🔀 Mix up the images so the model doesn't memorize the order
            verbose=1
        )
        
        weights_prime = [np.array(w) for w in self.model.get_weights()]
        num_examples_train = len(self.x_train)
        results = {
            "loss": float(history.history["loss"][0]),
            "accuracy": float(history.history["accuracy"][0]),
        }

        # 🧹 CLEANUP: Force flush the GPU memory to prevent crashes in the next round
        tf.keras.backend.clear_session() 
        import gc
        gc.collect()

        return weights_prime, num_examples_train, results
    
    def evaluate(self, parameters, config):
        """Validates the model on local data (Handled centrally by the server in our setup)."""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        return float(loss), len(self.x_train), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    # Read client ID from command line (e.g., "python client_node.py 1")
    if len(sys.argv) > 1:
        c_id = sys.argv[1] 
        client_id = f"client_{c_id}"
    else:
        client_id = "client_1"

    print(f"🏥 Starting Hospital Node: {client_id}")
    
    # 🔄 UPDATED FOR EFFICIENTNET
    client = TumorClient(model_name="efficientnet", client_id=client_id)
    
    # Connect to the local server
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080", 
        client=client
    )