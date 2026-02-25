import subprocess
import time
import sys
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

def run_simulation():
    print("🚀 Initializing Federated Learning Simulation...")
    print(f"📂 Working Directory: {SRC_DIR}")

    # 1. Start Server (New Window)
    print("   - Launching Server...")
    server_cmd = f'start cmd /k "cd /d {SRC_DIR} && conda activate fed_tumor && python server_node.py"'
    subprocess.Popen(server_cmd, shell=True)
    
    time.sleep(5) # Give server time to warm up

    # 2. Start Client 1 (New Window)
    print("   - Launching Hospital A (Client 1)...")
    client1_cmd = f'start cmd /k "cd /d {SRC_DIR} && conda activate fed_tumor && python client_node.py 1"'
    subprocess.Popen(client1_cmd, shell=True)

    # 3. Start Client 2 (New Window)
    print("   - Launching Hospital B (Client 2)...")
    client2_cmd = f'start cmd /k "cd /d {SRC_DIR} && conda activate fed_tumor && python client_node.py 2"'
    subprocess.Popen(client2_cmd, shell=True)

    print("\n✅ Simulation Running! Check the pop-up windows.")
    print("   (Close the new windows to stop the simulation)")

if __name__ == "__main__":
    run_simulation()