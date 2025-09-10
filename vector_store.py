import os
import pickle
from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available, using fallback method")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available, using numpy-based similarity search")

class VectorStoreManager:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize vector store with BERT-based embeddings.
        
        Args:
            model_name: Name of the sentence transformer model to use.
        """
        self.model_name = model_name
        self.model = None
        self.documents = []
        self.embeddings = None
        self.index = None
        
        # Initialize model
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print(f"Loading model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading sentence-transformers model: {e}")
                print("Falling back to transformers library...")
                self._load_transformers_model()
        else:
            self._load_transformers_model()
    
    def _load_transformers_model(self):
        """Fallback to transformers library."""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.transformer_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.model = "transformers"
            print("Using transformers library as fallback")
        except Exception as e:
            print(f"Error loading transformers: {e}")
            raise ImportError("Could not load any embedding model. Please install sentence-transformers or transformers.")
    
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        if self.model == "transformers":
            return self._encode_with_transformers(texts)
        else:
            return self.model.encode(texts, show_progress_bar=True)
    
    def _encode_with_transformers(self, texts: List[str]) -> np.ndarray:
        """Encode using transformers library."""
        import torch
        
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt',
                max_length=512
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.transformer_model(**encoded)
                # Mean pooling
                attention_mask = encoded['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings.extend(batch_embeddings.cpu().numpy())
        
        return np.array(embeddings)
    
    def create_vector_store(self, documents: List[Document]) -> 'VectorStoreManager':
        """
        Create vector store from documents.
        """
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        print(f"Creating vector store with {len(documents)} documents...")
        
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        
        # Create embeddings
        print("Generating embeddings...")
        self.embeddings = self._encode_texts(texts)
        
        # Create index
        if FAISS_AVAILABLE:
            self._create_faiss_index()
        else:
            print("Using numpy-based similarity search")
        
        print("Vector store created successfully!")
        return self
    
    def _create_faiss_index(self):
        """Create FAISS index."""
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings.astype(np.float32))
        self.index.add(self.embeddings.astype(np.float32))
        print(f"FAISS index created with {self.index.ntotal} vectors")
    
    def save_vector_store(self, path: str = "vector_store"):
        """Save vector store to disk."""
        if not self.documents:
            raise ValueError("No vector store to save. Create one first.")
        
        os.makedirs(path, exist_ok=True)
        
        # Save documents and embeddings
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        
        np.save(os.path.join(path, "embeddings.npy"), self.embeddings)
        
        # Save FAISS index if available
        if self.index is not None and FAISS_AVAILABLE:
            faiss.write_index(self.index, os.path.join(path, "faiss_index.bin"))
        
        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "num_documents": len(self.documents),
            "embedding_dim": self.embeddings.shape[1]
        }
        
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)
        
        print(f"Vector store saved to {path}")
    
    def load_vector_store(self, path: str = "vector_store") -> 'VectorStoreManager':
        """Load vector store from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vector store not found at {path}")
        
        # Load documents
        with open(os.path.join(path, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)
        
        # Load embeddings
        self.embeddings = np.load(os.path.join(path, "embeddings.npy"))
        
        # Load FAISS index if available
        faiss_path = os.path.join(path, "faiss_index.bin")
        if os.path.exists(faiss_path) and FAISS_AVAILABLE:
            self.index = faiss.read_index(faiss_path)
        
        print(f"Vector store loaded from {path}")
        return self
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search to find relevant documents.
        """
        if not self.documents:
            raise ValueError("No vector store available. Create or load one first.")
        
        # Encode query
        query_embedding = self._encode_texts([query])[0]
        
        if self.index is not None and FAISS_AVAILABLE:
            # Use FAISS search
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    results.append(self.documents[idx])
        else:
            # Use numpy-based similarity
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = [self.documents[idx] for idx in top_indices]
        
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with similarity scores.
        """
        if not self.documents:
            raise ValueError("No vector store available. Create or load one first.")
        
        # Encode query
        query_embedding = self._encode_texts([query])[0]
        
        if self.index is not None and FAISS_AVAILABLE:
            # Use FAISS search
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    # Convert distance to similarity (FAISS returns distances)
                    similarity = float(score)
                    results.append((self.documents[idx], similarity))
        else:
            # Use numpy-based similarity
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = [(self.documents[idx], similarities[idx]) for idx in top_indices]
        
        return results
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to existing vector store."""
        if not self.documents:
            raise ValueError("No vector store available. Create one first.")
        
        # Add documents
        self.documents.extend(documents)
        
        # Generate embeddings for new documents
        new_texts = [doc.page_content for doc in documents]
        new_embeddings = self._encode_texts(new_texts)
        
        # Update embeddings
        if self.embeddings is not None:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        else:
            self.embeddings = new_embeddings
        
        # Update FAISS index
        if self.index is not None and FAISS_AVAILABLE:
            faiss.normalize_L2(new_embeddings.astype(np.float32))
            self.index.add(new_embeddings.astype(np.float32))
        
        print(f"Added {len(documents)} documents to vector store")
    
    def get_retriever(self, k: int = 4):
        """
        Get a simple retriever function.
        """
        def retrieve(query: str) -> List[Document]:
            return self.similarity_search(query, k=k)
        
        return retrieve