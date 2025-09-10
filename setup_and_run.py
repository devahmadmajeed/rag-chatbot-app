#!/usr/bin/env python3
"""
Setup and run script for RAG Chatbot - Fixed version
"""

import os
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install required packages with error handling"""
    print("📦 Installing required packages...")
    
    # First, upgrade pip
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    except subprocess.CalledProcessError:
        print("⚠️  Warning: Could not upgrade pip")
    
    # Install packages one by one for better error handling
    packages = [
        "streamlit>=1.28.0",
        "numpy>=1.21.0", 
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "torch>=1.12.0",
        "transformers>=4.21.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.2",
        "python-dotenv>=0.19.0"
    ]
    
    failed_packages = []
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                package, "--no-cache-dir"
            ])
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️  Some packages failed to install: {failed_packages}")
        print("The chatbot might still work with basic functionality.")
        return False
    
    print("✅ All packages installed successfully!")
    return True

def install_requirements_fallback():
    """Fallback installation method"""
    print("🔄 Trying alternative installation method...")
    
    try:
        # Try installing from requirements.txt if it exists
        if os.path.exists("requirements.txt"):
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "-r", "requirements.txt", "--no-cache-dir"
            ])
        else:
            # Install minimal requirements
            minimal_packages = [
                "streamlit",
                "numpy", 
                "scikit-learn",
                "transformers",
                "torch"
            ]
            
            for package in minimal_packages:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ])
        
        print("✅ Fallback installation completed!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Fallback installation also failed: {e}")
        return False

def create_data_directory():
    """Create data directory if it doesn't exist"""
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir()
        print("📁 Created data directory")
    
    # Create sample Q&A file if none exists
    sample_file = data_dir / "sample_qa.txt"
    if not sample_file.exists():
        sample_content = """Q: How do I create an account?
A: To create an account, click the 'Sign Up' button on the homepage, fill in your details, and verify your email address.

Q: How do I reset my password?
A: Go to the login page, click 'Forgot Password', enter your email address, and follow the instructions sent to your email.

Q: What are the pricing plans?
A: We offer three plans: Basic ($9/month), Pro ($19/month), and Enterprise ($49/month). Each includes different features and usage limits.

Q: How do I contact support?
A: You can contact our support team through the 'Help' section in your dashboard, or email us at support@yourapp.com.

Q: Is there a mobile app?
A: Yes! Our mobile app is available for both iOS and Android. Download it from the App Store or Google Play Store.

Q: How do I delete my account?
A: To delete your account, go to Settings > Account > Delete Account. Note that this action is irreversible.

Q: What payment methods do you accept?
A: We accept all major credit cards (Visa, MasterCard, American Express), PayPal, and bank transfers for enterprise plans.

Q: Can I export my data?
A: Yes, you can export your data anytime by going to Settings > Data Export. We support CSV, JSON, and PDF formats.

Q: Do you offer refunds?
A: We offer a 30-day money-back guarantee for all paid plans. Contact support to request a refund.

Q: How secure is my data?
A: We use enterprise-grade encryption, regular security audits, and comply with GDPR and SOC 2 standards to protect your data."""
        
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        print("📝 Created sample Q&A file at data/sample_qa.txt")

def test_imports():
    """Test if critical imports work"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import numpy
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    # Test optional imports
    try:
        import sentence_transformers
        print("✅ Sentence transformers available")
    except ImportError:
        print("⚠️  Sentence transformers not available (fallback mode will be used)")
    
    try:
        import faiss
        print("✅ FAISS available")
    except ImportError:
        print("⚠️  FAISS not available (numpy similarity search will be used)")
    
    return True

def run_streamlit_app():
    """Run the Streamlit application"""
    print("🚀 Starting the RAG Chatbot application...")
    try:
        # Check if app.py exists
        if not os.path.exists("app.py"):
            print("❌ app.py not found. Please ensure all files are in the correct directory.")
            return
        
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.address", "localhost"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        print("Try running manually: streamlit run app.py")

def main():
    """Main setup and run function"""
    print("🤖 RAG Chatbot Setup (Fixed Version)")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install requirements
    installation_success = install_requirements()
    
    if not installation_success:
        print("\n🔄 Trying fallback installation...")
        if not install_requirements_fallback():
            print("\n❌ Installation failed. You can try:")
            print("1. pip install streamlit numpy scikit-learn transformers")
            print("2. Then run: streamlit run app.py")
            return
    
    # Test imports
    if not test_imports():
        print("\n⚠️  Some imports failed, but the app might still work with reduced functionality.")
    
    # Create directories and sample data
    create_data_directory()
    
    print("\n" + "=" * 60)
    print("✅ Setup complete!")
    print("\n📋 Next steps:")
    print("1. The application will start automatically")
    print("2. Upload your Q&A text file using the sidebar")
    print("3. Start chatting with your bot!")
    print("4. Use the sample file at 'data/sample_qa.txt' for testing")
    print("\n💡 If you encounter issues:")
    print("- Try the sample data first")
    print("- Check that your Q&A file follows the supported formats")
    print("- Restart the app if needed")
    print("\n" + "=" * 60)
    
    input("Press Enter to start the application...")
    
    # Run the application
    run_streamlit_app()

if __name__ == "__main__":
    main()