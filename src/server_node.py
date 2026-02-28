import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # 🛡️ Server stays on CPU to save GPU memory for clients

import flwr as fl
import tensorflow as tf
import numpy as np
import cv2
from model_architecture import get_model

def load_global_test_data():
    """Loads the global testing dataset to evaluate the model after each round."""
    base_dir = "../federated_dataset/global_test_set"
    classes = ['glioma', 'meningioma', 'pituitary', 'notumor']
    images, labels = [], []
    
    print("📂 Loading Global Test Set...")
    for label_idx, cls in enumerate(classes):
        cls_path = os.path.join(base_dir, cls)
        if not os.path.exists(cls_path): continue
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))
                images.append(img)
                labels.append(label_idx)
            except: pass
    
    return np.array(images, dtype='float32') / 255.0, np.array(labels, dtype='int32')

# Load the test data once when the server starts
X_TEST, Y_TEST = load_global_test_data()
best_accuracy = 0.0

def get_evaluate_fn(model_name):
    """Creates an evaluation function that the server uses to test the merged global model."""
    model = get_model(model_name)
    
    def evaluate(server_round, parameters, config):
        global best_accuracy
        # Inject the newly merged weights from the hospitals into the server's model
        model.set_weights(parameters)
        
        # Test the model on the unseen global data
        loss, accuracy = model.evaluate(X_TEST, Y_TEST, verbose=0)
        
        print(f"📊 ROUND {server_round} GLOBAL RESULT: Accuracy = {accuracy:.4f}")
        
        # 💾 SAVE MECHANISM: Only save the model if it beats the previous all-time high
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.save_weights("efficientnet_best.h5") # 🔄 UPDATED FOR EFFICIENTNET
            print(f"💾 New Record! Model saved with accuracy: {accuracy:.4f}")
            
        return float(loss), {"accuracy": float(accuracy)}
    return evaluate

if __name__ == "__main__":
    # -------------------------------------------------------
    # 🛡️ THE RESUME FIX: Load existing weights if they exist
    # -------------------------------------------------------
    initial_weights = None
    model_path = "efficientnet_best.h5" # 🔄 UPDATED FOR EFFICIENTNET
    
    if os.path.exists(model_path):
        print(f"🔄 Found saved weights '{model_path}'. Resuming training from previous state!")
        # 🔄 THE FIX: Build the blueprint first, then load the weights into it
        saved_model = get_model("efficientnet")
        saved_model.load_weights(model_path)
        initial_weights = fl.common.ndarrays_to_parameters(saved_model.get_weights())
    else:
        print("🆕 No existing model found. Starting fresh from Round 0.")

    # Define the Federated Averaging Strategy (How the server merges weights)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,             # Train on 100% of connected clients
        fraction_evaluate=0.0,        # We evaluate on the server, not the clients
        min_fit_clients=2,            # Wait for at least 2 hospitals
        min_available_clients=2,      
        evaluate_fn=get_evaluate_fn("efficientnet"), # 🔄 UPDATED FOR EFFICIENTNET
        initial_parameters=initial_weights, 
    )

    print("🚀 Server Online. Waiting for Hospitals...")
    
    # Start the server and run for 3 rounds per cycle
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3), 
        strategy=strategy
    )