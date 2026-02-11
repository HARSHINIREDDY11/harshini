import subprocess
import sys
import os

def run_script(script_name, description):
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"{'='*50}")
    try:
        # We use sys.executable to ensure we use the same python environment
        result = subprocess.run([sys.executable, script_name], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        return False

def main():
    print("Starting Multi-View Waste Segmentation Pipeline...")
    
    # 1. Data Setup
    if not run_script("data_setup.py", "Generating Dataset Structure and Mock Data"):
        sys.exit(1)
        
    # 2. Training (Limited to 2 epochs for demo run)
    # We modify the script content inline temporarily or just override sys.argv if we supported it
    # For simplicity, we just run train.py which I've already optimized for a demo feel
    print("\nStarting Training (this might take a few minutes on CPU)...")
    if not run_script("train.py", "Training Multi-View Swin Transformer + 3D CNN"):
        sys.exit(1)
        
    # 3. Evaluation
    if not run_script("evaluate.py", "Benchmarking and Metrics Visualization"):
        sys.exit(1)
        
    # 4. Inference Performance
    if not run_script("inference.py", "Deployment and FPS measurement"):
        sys.exit(1)
        
    print("\n" + "="*50)
    print("PIPELINE EXECUTION COMPLETE!")
    print("Check the 'plots/' directory for confusion matrices.")
    print("="*50)

if __name__ == "__main__":
    main()
