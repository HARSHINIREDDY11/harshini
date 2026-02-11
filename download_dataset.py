import os
import requests
import zipfile
import shutil

def download_trashnet(target_dir="./trashnet_raw"):
    """Downloads the official TrashNet dataset from GitHub."""
    target_repo = os.path.join(target_dir, "trashnet_repo")
    
    if os.path.exists(target_repo):
        shutil.rmtree(target_repo)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        
    os.makedirs(target_dir)
    
    print(f"Cloning TrashNet repo from https://github.com/garythung/trashnet.git...")
    try:
        import subprocess
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/garythung/trashnet.git", target_repo], check=True)
        print("Clone complete.")
    except Exception as e:
        print(f"Git clone failed: {e}")
        return

    # structure matches: trashnet_repo/data/dataset-resized.zip
    target_dir = target_repo # Adjust logic below to look inside here
    zip_path = "trashnet_repo" # Dummy to verify logic flow if needed, but below code needs adjustment

    
    # The real data is in a nested zip: trashnet-master/data/dataset-resized.zip
    # The real data is in: trashnet_repo/data/dataset-resized.zip
    inner_zip = os.path.join(target_repo, "data", "dataset-resized.zip")
    
    if os.path.exists(inner_zip):
        print(f"Extracting inner dataset {inner_zip}...")
        with zipfile.ZipFile(inner_zip, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
        source_folder = os.path.join(target_dir, "dataset-resized")
        final_dest = "./trashnet"
        
        print(f"Organizing files into {final_dest}...")
        if os.path.exists(final_dest):
            shutil.rmtree(final_dest)
        shutil.copytree(source_folder, final_dest)
        print("Dataset ready.")
        
        # Cleanup
        os.remove("trashnet.zip")
        shutil.rmtree(target_dir)
    else:
        print(f"Error: Could not find inner zip at {inner_zip}")

if __name__ == "__main__":
    download_trashnet()
