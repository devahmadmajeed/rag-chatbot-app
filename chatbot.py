import os
from typing import List, Dict, Any, Optional
from vector_store import VectorStoreManager
from data_processor import QADataProcessor
import re
from datetime import datetime

class SimpleBERTChatbot:
    def __init__(self, vector_store_manager: VectorStoreManager):
        """
        Simplified chatbot that uses BERT embeddings for retrieval
        and template-based responses.
        """
        self.vector_store_manager = vector_store_manager
        self.conversation_history = []
    
    def chat(self, question: str, k: int = 3) -> Dict[str, Any]:
        """
        Simple chat function using similarity search.
        """
        try:
            # Get similar documents with scores
            results = self.vector_store_manager.similarity_search_with_score(question, k=k)
            
            if not results:
                return {
                    "answer": "I don't have information about that topic in my knowledge base.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Get the best match
            best_doc, best_score = results[0]
            
            # Extract answer based on document type
            answer = self._extract_answer(best_doc, question)
            
            # Calculate confidence
            confidence = self._calculate_confidence(best_score, results)
            
            # Add to history
            self.conversation_history.append({
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "timestamp": self._get_timestamp()
            })
            
            return {
                "answer": answer,
                "sources": [
                    {
                        "content": doc.page_content,
                        "score": float(score) if hasattr(score, 'item') else float(score),
                        "metadata": doc.metadata
                    }
                    for doc, score in results
                ],
                "confidence": round(confidence, 2)
            }
            
        except Exception as e:
            print(f"Error in chat: {e}")  # For debugging
            return {
                "answer": f"I encountered an error while processing your question. Please try rephrasing it.",
                "sources": [],
                "confidence": 0.0
            }
    
    def _extract_answer(self, document, question: str) -> str:
        """
        Extract the most relevant answer from a document.
        """
        # If it's a Q&A pair, return the answer
        if document.metadata.get("type") == "qa_pair":
            answer = document.metadata.get("answer", "")
            if answer:
                return answer.strip()
        
        # If it's regular content, try to find the most relevant part
        content = document.page_content
        
        # Look for answer patterns in the content
        answer_patterns = [
            r'(?:Answer|A):\s*(.+?)(?=Question|Q:|$)',
            r'(?:Response|Reply):\s*(.+?)(?=Question|Q:|$)',
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        # If no specific answer pattern found, return the content
        # but try to make it more conversational
        if len(content) > 300:
            # Try to find the most relevant sentence
            sentences = content.split('.')
            question_words = set(question.lower().split())
            
            best_sentence = ""
            max_overlap = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # Ignore very short sentences
                    sentence_words = set(sentence.lower().split())
                    overlap = len(question_words.intersection(sentence_words))
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_sentence = sentence
            
            if best_sentence:
                return best_sentence + "."
        
        return content.strip()
    
    def _calculate_confidence(self, best_score: float, all_results: List) -> float:
        """
        Calculate confidence score based on similarity scores and result quality.
        """
        try:
            # Convert score to float if needed
            if hasattr(best_score, 'item'):
                score = float(best_score.item())
            else:
                score = float(best_score)
            
            # For cosine similarity, scores are typically between -1 and 1
            # Higher scores mean better similarity
            if score < 0:
                confidence = max(0.0, (score + 1) / 2)  # Convert -1,1 to 0,1
            else:
                confidence = min(1.0, score)  # Cap at 1.0
            
            # Adjust confidence based on number of good results
            good_results = sum(1 for _, s in all_results if (float(s.item()) if hasattr(s, 'item') else float(s)) > 0.3)
            if good_results >= 2:
                confidence = min(confidence + 0.1, 1.0)  # Bonus for multiple good matches
            
            return max(0.0, min(1.0, confidence))  # Ensure it's between 0 and 1
            
        except Exception:
            # Fallback confidence calculation
            return 0.5
    
    def get_similar_questions(self, question: str, k: int = 3) -> List[str]:
        """Get similar questions from the knowledge base."""
        try:
            similar_docs = self.vector_store_manager.similarity_search(question, k=k)
            
            questions = []
            for doc in similar_docs:
                if doc.metadata.get("type") == "qa_pair":
                    q = doc.metadata.get("question", "")
                    if q and q not in questions and q.lower() != question.lower():
                        questions.append(q)
            
            return questions[:k]
        except:
            return []
    
    def _get_timestamp(self):
        """Get current timestamp."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

class RAGChatbot:
    """
    Advanced RAG chatbot using language models (optional, for when LangChain works properly)
    """
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        self.conversation_history = []
    
    def chat(self, question: str, k: int = 3) -> Dict[str, Any]:
        """
        Process user question using retrieval and simple generation.
        """
        try:
            # Get relevant documents
            results = self.vector_store_manager.similarity_search_with_score(question, k=k)
            
            if not results:
                return {
                    "answer": "I don't have information about that topic in my knowledge base.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Combine context from multiple sources
            context_parts = []
            for doc, score in results:
                if doc.metadata.get("type") == "qa_pair":
                    answer = doc.metadata.get("answer", "")
                    if answer:
                        context_parts.append(answer)
                else:
                    context_parts.append(doc.page_content)
            
            # Generate response based on context
            context = " ".join(context_parts[:3])  # Limit context length
            answer = self._generate_response(question, context)
            
            # Calculate confidence
            best_score = results[0][1]
            confidence = self._calculate_confidence(best_score)
            
            # Add to conversation history
            self.conversation_history.append({
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            return {
                "answer": answer,
                "sources": [
                    {
                        "content": doc.page_content,
                        "score": float(score) if hasattr(score, 'item') else float(score),
                        "metadata": doc.metadata
                    }
                    for doc, score in results
                ],
                "confidence": confidence
            }
            
        except Exception as e:
            print(f"Error in RAG chat: {e}")  # For debugging
            return {
                "answer": "I encountered an error while processing your question. Please try rephrasing it.",
                "sources": [],
                "confidence": 0.0
            }
    
    def _generate_response(self, question: str, context: str) -> str:
        """
        Generate response based on question and context.
        This is a simple template-based approach.
        """
        # Simple rule-based response generation
        question_lower = question.lower()
        
        # Check for common question types
        if any(word in question_lower for word in ['how', 'what', 'where', 'when', 'why', 'who']):
            # For specific questions, try to find direct answers
            sentences = context.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10 and any(word in sentence.lower() for word in question_lower.split()):
                    return sentence + "."
        
        # For other questions, return the most relevant context
        if len(context) > 500:
            return context[:500] + "..."
        
        return context if context else "I found some relevant information, but I'm not sure how to answer your specific question."
    
    def _calculate_confidence(self, score: float) -> float:
        """Calculate confidence from similarity score."""
        try:
            if hasattr(score, 'item'):
                score = float(score.item())
            else:
                score = float(score)
            
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

# Factory function to create appropriate chatbot
def create_chatbot(vector_store_manager: VectorStoreManager, chatbot_type: str = "simple") -> Any:
    """
    Factory function to create the appropriate chatbot.
    
    Args:
        vector_store_manager: The vector store manager
        chatbot_type: "simple" or "rag"
    
    Returns:
        Appropriate chatbot instance
    """
    if chatbot_type.lower() == "rag":
        return RAGChatbot(vector_store_manager)
    else:
        return SimpleBERTChatbot(vector_store_manager)