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

def initialize_chatbot():
    """Initialize the chatbot components."""
    if 'chatbot_initialized' not in st.session_state:
        with st.spinner("Initializing chatbot..."):
            # Initialize components
            st.session_state.processor = QADataProcessor()
            st.session_state.vector_manager = VectorStoreManager()
            st.session_state.chatbot_initialized = True
    
    return st.session_state.processor, st.session_state.vector_manager

def load_and_process_data(file_path, processor, vector_manager):
    """Load and process the Q&A data."""
    try:
        # Process the data
        with st.spinner("Processing Q&A data..."):
            documents = processor.load_qa_file(file_path)
            
        if not documents:
            st.error("No documents were processed. Please check your file format.")
            return None
        
        # Create vector store
        with st.spinner("Creating vector embeddings..."):
            vector_store = vector_manager.create_vector_store(documents)
        
        # Display statistics
        stats = processor.get_statistics()
        st.success(f"Successfully processed {stats['total_documents']} documents!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", stats['total_documents'])
        with col2:
            st.metric("Q&A Pairs", stats['qa_pairs'])
        with col3:
            st.metric("Avg Length", f"{stats['avg_content_length']:.0f} chars")
        
        return SimpleBERTChatbot(vector_manager)
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

def main():
    st.title("🤖 RAG Chatbot with BERT")
    st.markdown("Ask questions about your app based on the uploaded Q&A data!")
    
    # Initialize components
    processor, vector_manager = initialize_chatbot()
    
    # Sidebar for file upload and configuration
    with st.sidebar:
        st.header("Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Q&A Text File",
            type=['txt'],
            help="Upload a text file containing questions and answers about your app"
        )
        
        # Model selection
        model_options = {
            "all-MiniLM-L6-v2 (Fast)": "sentence-transformers/all-MiniLM-L6-v2",
            "all-mpnet-base-v2 (Better Quality)": "sentence-transformers/all-mpnet-base-v2",
            "paraphrase-distilroberta-base-v1": "sentence-transformers/paraphrase-distilroberta-base-v1"
        }
        
        selected_model = st.selectbox(
            "Choose BERT Model",
            options=list(model_options.keys()),
            help="Select the embedding model for better performance"
        )
        
        # Number of context documents
        k_docs = st.slider("Number of context documents", 1, 10, 3)
        
        st.markdown("---")
        st.markdown("### Model Info")
        st.info(f"Using: {model_options[selected_model]}")
    
    # Main content area
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_qa_data.txt", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Initialize chatbot if not already done for this file
        if 'chatbot' not in st.session_state or st.session_state.get('current_file') != uploaded_file.name:
            # Update model if changed
            if st.session_state.get('current_model') != selected_model:
                vector_manager.model_name = model_options[selected_model]
                vector_manager.embeddings = None  # Reset embeddings
                st.session_state.current_model = selected_model
            
            chatbot = load_and_process_data("temp_qa_data.txt", processor, vector_manager)
            if chatbot:
                st.session_state.chatbot = chatbot
                st.session_state.current_file = uploaded_file.name
        
        # Chat interface
        if 'chatbot' in st.session_state:
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
                        response = st.session_state.chatbot.chat(prompt, k=k_docs)
                    
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
            if st.button("🔍 Get Similar Questions"):
                if st.session_state.messages:
                    last_user_msg = None
                    for msg in reversed(st.session_state.messages):
                        if msg["role"] == "user":
                            last_user_msg = msg["content"]
                            break
                    
                    if last_user_msg:
                        similar_q = st.session_state.chatbot.vector_store_manager.similarity_search(last_user_msg, k=5)
                        if similar_q:
                            st.markdown("**Similar questions you might ask:**")
                            for doc in similar_q:
                                if doc.metadata.get("question"):
                                    if st.button(f"❓ {doc.metadata['question'][:80]}...", key=f"sim_{hash(doc.metadata['question'])}"):
                                        st.rerun()
        
        # Clean up temp file
        if os.path.exists("temp_qa_data.txt"):
            os.remove("temp_qa_data.txt")
    
    else:
        # Welcome screen
        st.markdown("""
        ### Welcome to your RAG Chatbot! 🚀
        
        To get started:
        1. **Upload your Q&A text file** using the sidebar
        2. **Choose a BERT model** for embeddings (all-MiniLM-L6-v2 is recommended for speed)
        3. **Start chatting** with your bot!
        
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
        """)

if __name__ == "__main__":
    main()