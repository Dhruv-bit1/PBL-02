import kagglehub
import shutil
import os

# Always build paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Download dataset (this downloads FULL 12GB)
dataset_path = kagglehub.dataset_download("vulamnguyen/rwf2000")

print("Downloaded to:", dataset_path)

# Target directory inside project
target_dir = os.path.join(BASE_DIR, "data", "raw", "RWF-2000")

# Create target directory
os.makedirs(target_dir, exist_ok=True)

# Copy dataset into project folder
for item in os.listdir(dataset_path):
    src = os.path.join(dataset_path, item)
    dst = os.path.join(target_dir, item)

    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print("Dataset copied to:", target_dir)
