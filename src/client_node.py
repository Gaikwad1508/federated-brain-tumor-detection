import sys
import os
import cv2
import numpy as np
import flwr as fl
import tensorflow as tf
from model_architecture import get_model

# 🛑 CRITICAL: Prevent GPU Memory Hoarding
# This ensures one client doesn't eat 100% of VRAM, allowing others to run.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def load_data(client_id, base_dir="../federated_dataset"):
    """
    Loads images for a specific client (e.g., 'client_1').
    Returns numpy arrays: x_train, y_train
    """
    client_path = os.path.join(base_dir, client_id)
    classes = ['glioma', 'meningioma', 'pituitary', 'notumor']
    
    images = []
    labels = []
    
    print(f"📂 Loading data for {client_id}...")
    
    for label_idx, cls in enumerate(classes):
        cls_path = os.path.join(client_path, cls)
        if not os.path.exists(cls_path):
            continue
            
        # Iterate over images
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            try:
                # Read and Resize
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))
                images.append(img)
                labels.append(label_idx)
            except Exception as e:
                print(f"⚠️ Error loading {img_name}: {e}")

    # Convert to Numpy Arrays and Normalize (0-1)
    x_train = np.array(images, dtype='float32') / 255.0
    y_train = np.array(labels, dtype='int32')
    
    print(f"✅ Loaded {len(x_train)} images for {client_id}.")
    return x_train, y_train

class TumorClient(fl.client.NumPyClient):
    def __init__(self, model_name, client_id):
        self.model = get_model(model_name)
        self.x_train, self.y_train = load_data(client_id)

    def get_parameters(self, config):
        """Returns the current model weights."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """
        1. Receive global weights.
        2. Train on local data.
        3. Return updated weights.
        """
        self.model.set_weights(parameters)
        
        # Train carefully
        history = self.model.fit(
            self.x_train, self.y_train, 
            epochs=3, 
            batch_size=8, # Keep this strictly at 8
            shuffle=True, # 🔀 THE FIX: Force it to shuffle images every epoch
            verbose=1
        )
        
        # Convert weights
        weights_prime = [np.array(w) for w in self.model.get_weights()]
        num_examples_train = len(self.x_train)
        results = {
            "loss": float(history.history["loss"][0]),
            "accuracy": float(history.history["accuracy"][0]),
        }

        # 🧹 CLEANUP MEMORY TO PREVENT CRASH IN NEXT ROUND
        tf.keras.backend.clear_session() 
        import gc
        gc.collect()

        return weights_prime, num_examples_train, results
    
    def evaluate(self, parameters, config):
        """
        Validate the model on local data.
        """
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        return loss, len(self.x_train), {"accuracy": accuracy}

# This allows us to run a client manually if needed
if __name__ == "__main__":
    # Read client ID from command line (e.g., "python client_node.py 2")
    if len(sys.argv) > 1:
        c_id = sys.argv[1] # "1", "2", or "3"
        client_id = f"client_{c_id}"
    else:
        client_id = "client_1" # Default

    print(f"🏥 Starting Hospital Node: {client_id}")
    
    # Start the client
    client = TumorClient(model_name="mobilenet", client_id=client_id)
    
    # Connect to the server
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080", 
        client=client
    )