#!/usr/bin/env python3
"""
Troubleshooting script for RAG Chatbot
"""

import sys
import os
import importlib.util

def check_python_version():
    """Check Python version"""
    print("🐍 Python Version Check:")
    print(f"   Current version: {sys.version}")
    
    if sys.version_info >= (3, 8):
        print("   ✅ Python version is compatible")
        return True
    else:
        print("   ❌ Python 3.8+ required")
        return False

def check_package(package_name, import_name=None, required=True):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    status = "🔍" if required else "🔧"
    print(f"{status} Checking {package_name}...")
    
    # Check if package is installed
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        if required:
            print(f"   ❌ {package_name} not installed")
            return False
        else:
            print(f"   ⚠️  {package_name} not available (optional)")
            return False
    
    # Try importing
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"   ✅ {package_name} v{version} - OK")
        return True
    except ImportError as e:
        print(f"   ❌ {package_name} import failed: {e}")
        return False
    except Exception as e:
        print(f"   ⚠️  {package_name} warning: {e}")
        return True

def check_files():
    """Check if required files exist"""
    print("📁 File Check:")
    
    required_files = [
        "app.py",
        "vector_store.py", 
        "data_processor.py",
        "chatbot.py"
    ]
    
    optional_files = [
        "requirements.txt",
        "setup_and_run.py",
        "train_model.py"
    ]
    
    all_good = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - MISSING!")
            all_good = False
    
    for file in optional_files:
        if os.path.exists(file):
            print(f"   ✅ {file} (optional)")
        else:
            print(f"   ⚠️  {file} (optional) - missing")
    
    return all_good

def test_vector_store():
    """Test vector store functionality"""
    print("🔧 Testing Vector Store:")
    
    try:
        from vector_store import VectorStoreManager
        print("   ✅ vector_store.py imports successfully")
        
        # Test initialization
        vm = VectorStoreManager()
        print("   ✅ VectorStoreManager initializes")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Initialization error: {e}")
        return False

def test_data_processor():
    """Test data processor"""
    print("🔧 Testing Data Processor:")
    
    try:
        from data_processor import QADataProcessor
        print("   ✅ data_processor.py imports successfully")
        
        processor = QADataProcessor()
        print("   ✅ QADataProcessor initializes")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_chatbot():
    """Test chatbot functionality"""
    print("🔧 Testing Chatbot:")
    
    try:
        from chatbot import SimpleBERTChatbot
        print("   ✅ chatbot.py imports successfully")
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def run_quick_test():
    """Run a quick end-to-end test"""
    print("🧪 Quick Functionality Test:")
    
    try:
        # Test the complete pipeline
        from data_processor import QADataProcessor
        from vector_store import VectorStoreManager
        from chatbot import SimpleBERTChatbot
        
        # Create sample data
        sample_qa = """Q: What is a test?
A: A test is a procedure to check if something works correctly."""
        
        with open("test_qa.txt", "w") as f:
            f.write(sample_qa)
        
        # Test processing
        processor = QADataProcessor()
        documents = processor.load_qa_file("test_qa.txt")
        print(f"   ✅ Processed {len(documents)} documents")
        
        # Test vector store (this might take a moment)
        print("   🔄 Creating vector embeddings (this may take a minute)...")
        vector_manager = VectorStoreManager()
        vector_manager.create_vector_store(documents)
        print("   ✅ Vector store created")
        
        # Test chatbot
        chatbot = SimpleBERTChatbot(vector_manager)
        response = chatbot.chat("What is a test?")
        print(f"   ✅ Chatbot response: {response['answer'][:50]}...")
        
        # Cleanup
        os.remove("test_qa.txt")
        
        print("   ✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        if os.path.exists("test_qa.txt"):
            os.remove("test_qa.txt")
        return False

def provide_solutions():
    """Provide common solutions"""
    print("\n💡 Common Solutions:")
    print("=" * 50)
    
    print("\n1. If packages are missing:")
    print("   pip install streamlit numpy scikit-learn transformers torch")
    print("   pip install sentence-transformers faiss-cpu")
    
    print("\n2. If sentence-transformers fails:")
    print("   pip install --upgrade sentence-transformers")
    print("   or pip install transformers torch (chatbot will use fallback)")
    
    print("\n3. If FAISS fails:")
    print("   pip install faiss-cpu")
    print("   (chatbot will work without FAISS using numpy)")
    
    print("\n4. If import errors persist:")
    print("   Try creating a fresh virtual environment:")
    print("   python -m venv myenv")
    print("   myenv\\Scripts\\activate  (Windows)")
    print("   source myenv/bin/activate  (Linux/Mac)")
    
    print("\n5. Memory issues:")
    print("   Use a smaller model or reduce batch size")
    print("   Close other applications to free up RAM")

def main():
    """Main troubleshooting function"""
    print("🔍 RAG Chatbot Troubleshooting")
    print("=" * 60)
    
    issues = []
    
    # Check Python version
    if not check_python_version():
        issues.append("Python version")
    
    print()
    
    # Check required packages
    required_packages = [
        ("streamlit", "streamlit"),
        ("numpy", "numpy"), 
        ("scikit-learn", "sklearn"),
        ("transformers", "transformers")
    ]
    
    for pkg_name, import_name in required_packages:
        if not check_package(pkg_name, import_name, required=True):
            issues.append(f"Missing {pkg_name}")
    
    # Check optional packages
    optional_packages = [
        ("sentence-transformers", "sentence_transformers"),
        ("faiss-cpu", "faiss"),
        ("torch", "torch"),
        ("pandas", "pandas")
    ]
    
    for pkg_name, import_name in optional_packages:
        check_package(pkg_name, import_name, required=False)
    
    print()
    
    # Check files
    if not check_files():
        issues.append("Missing files")
    
    print()
    
    # Test components
    if not test_data_processor():
        issues.append("Data processor")
    
    if not test_vector_store():
        issues.append("Vector store")
        
    if not test_chatbot():
        issues.append("Chatbot")
    
    print()
    
    # Summary
    if issues:
        print("❌ Issues Found:")
        for issue in issues:
            print(f"   - {issue}")
        
        provide_solutions()
    else:
        print("✅ Basic checks passed! Testing full functionality...")
        if run_quick_test():
            print("\n🎉 Everything looks good! Your chatbot should work.")
            print("   Run: streamlit run app.py")
        else:
            provide_solutions()

if __name__ == "__main__":
    main()