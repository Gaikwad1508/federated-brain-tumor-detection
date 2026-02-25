import os
import shutil
import random
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_DIR = "../data_raw/Training"  # Path to Kaggle folder
OUTPUT_DIR = "../federated_dataset"
CLASSES = ['glioma', 'meningioma', 'pituitary', 'notumor']

def setup_federated_data():
    # 1. Clean previous runs
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    # Create Client and Test directories
    for split in ['client_1', 'client_2', 'client_3', 'global_test_set']:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

    print(f"🚀 Starting Non-IID Data Distribution from {SOURCE_DIR}...")

    # 2. Process each class
    for cls in CLASSES:
        src_path = os.path.join(SOURCE_DIR, cls)
        # Get all valid images
        images = [os.path.join(src_path, f) for f in os.listdir(src_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        
        total = len(images)
        # Separate 20% for Global Testing (The Server's secret set)
        test_split = int(total * 0.2)
        test_imgs = images[:test_split]
        train_imgs = images[test_split:] # Remaining 80% for training
        
        # Save Test Images
        for img in test_imgs: 
            shutil.copy(img, os.path.join(OUTPUT_DIR, 'global_test_set', cls))

        # 3. NON-IID LOGIC (The "Specialist" Simulation)
        if cls == 'glioma':
            # Client 1 is the Glioma Specialist
            c1_end = int(len(train_imgs) * 0.7) # 70% to Client 1
            c2_end = c1_end + int(len(train_imgs) * 0.1) # 10% to Client 2
            # Remaining 20% to Client 3
        
        elif cls == 'meningioma':
            # Client 2 is the Meningioma Specialist
            c1_end = int(len(train_imgs) * 0.1) # 10% to Client 1
            c2_end = c1_end + int(len(train_imgs) * 0.7) # 70% to Client 2
        
        else:
            # Pituitary & No Tumor are split equally (General/IID)
            c1_end = int(len(train_imgs) * 0.33)
            c2_end = c1_end + int(len(train_imgs) * 0.33)

        # Slice the lists
        c1_imgs = train_imgs[:c1_end]
        c2_imgs = train_imgs[c1_end:c2_end]
        c3_imgs = train_imgs[c2_end:]

        # Copy to Clients
        for img in c1_imgs: shutil.copy(img, os.path.join(OUTPUT_DIR, 'client_1', cls))
        for img in c2_imgs: shutil.copy(img, os.path.join(OUTPUT_DIR, 'client_2', cls))
        for img in c3_imgs: shutil.copy(img, os.path.join(OUTPUT_DIR, 'client_3', cls))
        
        print(f"   Processed {cls}: C1={len(c1_imgs)}, C2={len(c2_imgs)}, C3={len(c3_imgs)}")

    print(f"\n✅ Federated Data Ready at: {OUTPUT_DIR}")
    print("   - global_test_set: Use this to evaluate the Global Model")
    print("   - client_1/2/3: Use these to train Local Models")

if __name__ == "__main__":
    setup_federated_data()