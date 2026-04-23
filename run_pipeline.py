import os
import subprocess
import sys



def run_step(command, step_name):
    """
    Execute a subprocess command and stream output.
    Stop pipeline on error.
    """
    print(f"\n{'='*50}")
    print(f"🚀 [START] Step: {step_name}")
    print(f"💻 Command: {command}")
    print(f"{'='*50}\n")
    
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"\n✅ [SUCCESS] Step '{step_name}' completed!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ [FAILED] Step '{step_name}' failed with exit code: {e.returncode}")
        print("Pipeline stopped.")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⚠️ [INTERRUPT] User interrupted step '{step_name}'.")
        sys.exit(1)


if __name__ == "__main__":
    print("🌟 SECOND-DeepOps Training & Evaluation Pipeline 🌟")
    print("This pipeline will automatically execute training and evaluation.\n")
    
    # Get Python executable
    python_exe = sys.executable

    # Step 1: Training
    print("\n" + "="*60)
    print("STEP 1: Model Training")
    print("="*60)
    train_cmd = f'"{python_exe}" train.py'
    run_step(train_cmd, "Model Training (train.py)")
    
    # Step 2: Evaluation
    print("\n" + "="*60)
    print("STEP 2: Model Evaluation")
    print("="*60)
    eval_cmd = f'"{python_exe}" evaluate.py'
    run_step(eval_cmd, "Model Evaluation (evaluate.py)")
    
    print("\n" + "🎉" + "="*50)
    print("Pipeline completed successfully!")
    print("="*50)
    print("\nModel checkpoints saved in: output/ckpt/")
    print("Final model: output/ckpt/final_model.pth")
