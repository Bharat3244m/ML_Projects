#!/usr/bin/env python3
"""
Script to fix scikit-learn version compatibility issues
"""
import subprocess
import sys
import os

def check_scikit_version():
    """Check current scikit-learn version"""
    try:
        import sklearn
        print(f"Current scikit-learn version: {sklearn.__version__}")
        return sklearn.__version__
    except ImportError:
        print("scikit-learn not installed")
        return None

def install_compatible_version():
    """Install a compatible scikit-learn version"""
    print("Installing scikit-learn 1.3.2 for compatibility...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.3.2"])
        print("✓ scikit-learn 1.3.2 installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing scikit-learn: {e}")
        return False

def test_model_loading():
    """Test if model and preprocessor can be loaded"""
    try:
        from src.utils import load_object
        
        print("Testing model loading...")
        model = load_object('artifacts/model.pkl')
        print("✓ Model loaded successfully")
        
        print("Testing preprocessor loading...")
        preprocessor = load_object('artifacts/preprocessor.pkl')
        print("✓ Preprocessor loaded successfully")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model/preprocessor: {e}")
        return False

def main():
    print("🔧 Scikit-learn Compatibility Fixer")
    print("=" * 40)
    
    current_version = check_scikit_version()
    
    if current_version != "1.3.2":
        print(f"⚠️  Incompatible version detected: {current_version}")
        print("Installing compatible version...")
        
        if install_compatible_version():
            print("\n✅ Compatibility issue fixed!")
            print("You can now run your application without version errors.")
        else:
            print("\n❌ Failed to fix compatibility issue")
            return False
    else:
        print("✅ Compatible version already installed")
    
    print("\n🧪 Testing model loading...")
    if test_model_loading():
        print("\n🎉 Everything is working correctly!")
        print("You can now run: streamlit run streamlit_app.py")
    else:
        print("\n❌ Model loading failed. You may need to retrain the model.")
    
    return True

if __name__ == "__main__":
    main()
