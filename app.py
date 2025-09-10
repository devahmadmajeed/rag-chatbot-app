import streamlit as st
import os
from data_processor import QADataProcessor
from vector_store import VectorStoreManager
from chatbot import SimpleBERTChatbot
import time

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Ensure data directory exists (for Streamlit Cloud)
if not os.path.exists("data"):
    os.makedirs("data", exist_ok=True)

# Create sample data if none exists (for Streamlit Cloud)
sample_data_path = "data/sample_qa.txt"
if not os.path.exists(sample_data_path):
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
    
    with open(sample_data_path, 'w', encoding='utf-8') as f:
        f.write(sample_content)

@st.cache_resource
def initialize_chatbot():
    """Initialize the chatbot components with caching for better performance."""
    processor = QADataProcessor()
    vector_manager = VectorStoreManager()
    return processor, vector_manager

@st.cache_resource
def load_and_process_data(_processor, _vector_manager, file_content, file_name):
    """Load and process the Q&A data with caching."""
    try:
        # Save content to temp file
        temp_path = f"temp_{file_name}"
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(file_content)
        
        # Process the data
        documents = _processor.load_qa_file(temp_path)
        
        if not documents:
            return None, "No documents were processed. Please check your file format."
        
        # Create vector store
        vector_store = _vector_manager.create_vector_store(documents)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Get statistics
        stats = _processor.get_statistics()
        
        return SimpleBERTChatbot(_vector_manager), stats
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None, f"Error processing data: {str(e)}"

def main():
    st.title("🤖 RAG Chatbot with BERT")
    st.markdown("Ask questions about your app based on uploaded Q&A data!")
    
    # Initialize components
    processor, vector_manager = initialize_chatbot()
    
    # Sidebar for file upload and configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Option to use sample data
        use_sample = st.checkbox("Use Sample Data", value=True, help="Use built-in sample Q&A data to try the chatbot")
        
        if use_sample:
            # Load sample data
            with open(sample_data_path, 'r', encoding='utf-8') as f:
                sample_content = f.read()
            
            if 'sample_chatbot' not in st.session_state:
                with st.spinner("Loading sample data..."):
                    chatbot, stats = load_and_process_data(processor, vector_manager, sample_content, "sample_qa.txt")
                    if chatbot:
                        st.session_state.sample_chatbot = chatbot
                        st.session_state.sample_stats = stats
                        st.success("Sample data loaded!")
                    else:
                        st.error("Failed to load sample data")
            
            if 'sample_stats' in st.session_state:
                stats = st.session_state.sample_stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Documents", stats['total_documents'])
                with col2:
                    st.metric("Q&A Pairs", stats['qa_pairs'])
                with col3:
                    st.metric("Avg Length", f"{stats['avg_content_length']:.0f}")
        else:
            # File upload option
            uploaded_file = st.file_uploader(
                "Upload Q&A Text File",
                type=['txt'],
                help="Upload a text file containing questions and answers about your app"
            )
            
            if uploaded_file is not None:
                # Process uploaded file
                file_content = uploaded_file.read().decode('utf-8')
                
                if 'uploaded_chatbot' not in st.session_state or st.session_state.get('current_file') != uploaded_file.name:
                    with st.spinner("Processing uploaded data..."):
                        chatbot, result = load_and_process_data(processor, vector_manager, file_content, uploaded_file.name)
                        if chatbot:
                            st.session_state.uploaded_chatbot = chatbot
                            st.session_state.uploaded_stats = result
                            st.session_state.current_file = uploaded_file.name
                            st.success(f"Successfully processed {result['total_documents']} documents!")
                        else:
                            st.error(result)
                
                if 'uploaded_stats' in st.session_state:
                    stats = st.session_state.uploaded_stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Documents", stats['total_documents'])
                    with col2:
                        st.metric("Q&A Pairs", stats['qa_pairs'])
                    with col3:
                        st.metric("Avg Length", f"{stats['avg_content_length']:.0f}")
        
        # Model configuration
        st.markdown("---")
        model_options = {
            "all-MiniLM-L6-v2 (Fast)": "sentence-transformers/all-MiniLM-L6-v2",
            "all-mpnet-base-v2 (Better Quality)": "sentence-transformers/all-mpnet-base-v2",
        }
        
        selected_model = st.selectbox(
            "Choose BERT Model",
            options=list(model_options.keys()),
            help="Select the embedding model for better performance"
        )
        
        k_docs = st.slider("Number of context documents", 1, 5, 3)
        
        st.markdown("---")
        st.info(f"Using: {model_options[selected_model]}")
    
    # Determine which chatbot to use
    current_chatbot = None
    if use_sample and 'sample_chatbot' in st.session_state:
        current_chatbot = st.session_state.sample_chatbot
    elif not use_sample and 'uploaded_chatbot' in st.session_state:
        current_chatbot = st.session_state.uploaded_chatbot
    
    # Chat interface
    if current_chatbot:
        st.markdown("---")
        st.header("💬 Chat with your bot")
        
        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm ready to answer questions about your app. What would you like to know?"}
            ]
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your app..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get bot response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = current_chatbot.chat(prompt, k=k_docs)
                
                # Display answer
                st.markdown(response["answer"])
                
                # Display confidence and sources in expandable section
                with st.expander(f"📊 Details (Confidence: {response['confidence']:.2f})"):
                    st.markdown("**Sources used:**")
                    for i, source in enumerate(response["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(source["content"][:200] + "..." if len(source["content"]) > 200 else source["content"])
                        if "score" in source:
                            st.caption(f"Similarity score: {source['score']:.3f}")
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
        
        # Suggested questions
        with st.container():
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔍 Get Similar Questions"):
                    if st.session_state.messages:
                        last_user_msg = None
                        for msg in reversed(st.session_state.messages):
                            if msg["role"] == "user":
                                last_user_msg = msg["content"]
                                break
                        
                        if last_user_msg:
                            similar_q = current_chatbot.get_similar_questions(last_user_msg, k=3)
                            if similar_q:
                                st.markdown("**Similar questions:**")
                                for q in similar_q:
                                    st.write(f"❓ {q}")
            
            with col2:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.messages = [
                        {"role": "assistant", "content": "Hello! I'm ready to answer questions about your app. What would you like to know?"}
                    ]
                    current_chatbot.clear_history()
                    st.rerun()
    
    else:
        # Welcome screen
        st.markdown("""
        ### Welcome to your RAG Chatbot! 🚀
        
        **Quick Start:**
        1. ✅ **Use Sample Data** (checkbox in sidebar) to try the chatbot immediately
        2. 📁 **Or upload your own Q&A text file** using the sidebar
        3. 💬 **Start chatting** with your bot!
        
        #### Supported file formats:
        - `Q: ... A: ...` format
        - Question-Answer pairs separated by lines
        - JSON-like structured format
        
        #### Example Q&A format:
        ```
        Q: How do I reset my password?
        A: Go to the login page and click 'Forgot Password'. Enter your email to receive a reset link.
        
        Q: What are the pricing plans?
        A: We offer Basic ($9/month), Pro ($19/month), and Enterprise ($49/month) plans.
        ```
        
        **Try the sample data first to see how it works!**
        """)

if __name__ == "__main__":
    main()