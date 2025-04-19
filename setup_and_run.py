import os
import subprocess
import sys

def run_command(command):
    """Run a command and print output"""
    print(f"Running: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False

def setup_project():
    """Setup the project environment"""
    print("Setting up Exoplanet Analysis Project...")

    # Create virtual environment
    print("\n1. Creating virtual environment...")
    if os.name == 'nt':  # Windows
        run_command("python -m venv venv")
        run_command(".\\venv\\Scripts\\activate")
    else:  # Linux/Mac
        run_command("python3 -m venv venv")
        run_command("source venv/bin/activate")

    # Install requirements
    print("\n2. Installing requirements...")
    run_command("pip install -r requirements.txt")

    # Create necessary directories
    print("\n3. Creating model directory...")
    os.makedirs("models", exist_ok=True)

    # Train and save models
    print("\n4. Training and saving models...")
    if run_command("python train_save_models.py"):
        print("Models trained successfully!")
    else:
        print("Error training models!")
        return False

    # Test Streamlit app
    print("\n5. Testing Streamlit app...")
    print("Starting Streamlit app... (Press Ctrl+C to exit)")
    run_command("cd streamlit_app && streamlit run app.py")

    return True

if __name__ == "__main__":
    setup_project()