import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Server stays on CPU

import flwr as fl
import tensorflow as tf
import numpy as np
import cv2
from model_architecture import get_model

def load_global_test_data():
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

X_TEST, Y_TEST = load_global_test_data()

best_accuracy = 0.0

def get_evaluate_fn(model_name):
    model = get_model(model_name)
    
    def evaluate(server_round, parameters, config):
        global best_accuracy
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_TEST, Y_TEST, verbose=0)
        
        print(f"📊 ROUND {server_round} GLOBAL RESULT: Accuracy = {accuracy:.4f}")
        
        # Save the model if it's the best one so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.save("best_brain_tumor_model.h5")
            print(f"💾 New Record! Model saved with accuracy: {accuracy:.4f}")
            
        return loss, {"accuracy": accuracy}
    return evaluate

if __name__ == "__main__":
    # The Strategy handles the averaging of weights automatically
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,             # Sample 100% of clients for training
        fraction_evaluate=0.0,        # <--- CHANGE THIS TO 0.0 (Disable client-side testing)
        min_fit_clients=2,            
        min_evaluate_clients=0,       # <--- CHANGE THIS TO 0 (Don't wait for client tests)
        min_available_clients=2,      
        evaluate_fn=get_evaluate_fn("mobilenet"), # Keep Server-side testing (The important one)
    )

    print("🚀 Server Online. Waiting for Hospitals...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=15),
        strategy=strategy
    )