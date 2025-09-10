import re
import pandas as pd
from typing import List, Dict
from langchain.schema import Document

class QADataProcessor:
    def __init__(self):
        self.documents = []
    
    def load_qa_file(self, file_path: str) -> List[Document]:
        """
        Load Q&A data from text file and convert to LangChain documents.
        Supports various Q&A formats.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Try different parsing methods
        documents = []
        
        # Method 1: Q: ... A: ... format
        qa_pairs = self._parse_qa_format(content)
        if qa_pairs:
            for qa in qa_pairs:
                doc = Document(
                    page_content=f"Question: {qa['question']}\nAnswer: {qa['answer']}",
                    metadata={
                        "type": "qa_pair",
                        "question": qa['question'],
                        "answer": qa['answer']
                    }
                )
                documents.append(doc)
        
        # Method 2: JSON-like format
        if not documents:
            documents = self._parse_json_like_format(content)
        
        # Method 3: Simple line-by-line format
        if not documents:
            documents = self._parse_line_format(content)
        
        self.documents = documents
        return documents
    
    def _parse_qa_format(self, content: str) -> List[Dict]:
        """Parse Q: ... A: ... format"""
        qa_pairs = []
        
        # Pattern to match Q: ... A: ... format
        pattern = r'Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)'
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        
        for question, answer in matches:
            qa_pairs.append({
                'question': question.strip(),
                'answer': answer.strip()
            })
        
        return qa_pairs
    
    def _parse_json_like_format(self, content: str) -> List[Document]:
        """Parse JSON-like or structured format"""
        documents = []
        
        # Try to find question-answer patterns
        lines = content.split('\n')
        current_question = None
        current_answer = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line contains question indicators
            if any(indicator in line.lower() for indicator in ['question:', 'q:', '?']):
                if current_question and current_answer:
                    # Save previous Q&A pair
                    doc = Document(
                        page_content=f"Question: {current_question}\nAnswer: {current_answer}",
                        metadata={
                            "type": "qa_pair",
                            "question": current_question,
                            "answer": current_answer
                        }
                    )
                    documents.append(doc)
                
                current_question = line
                current_answer = None
            
            # Check if line contains answer indicators
            elif any(indicator in line.lower() for indicator in ['answer:', 'a:', 'response:']):
                current_answer = line
            
            # If we have a question but no answer yet, this might be the answer
            elif current_question and not current_answer:
                current_answer = line
        
        # Don't forget the last pair
        if current_question and current_answer:
            doc = Document(
                page_content=f"Question: {current_question}\nAnswer: {current_answer}",
                metadata={
                    "type": "qa_pair",
                    "question": current_question,
                    "answer": current_answer
                }
            )
            documents.append(doc)
        
        return documents
    
    def _parse_line_format(self, content: str) -> List[Document]:
        """Parse simple line-by-line format"""
        documents = []
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # If odd number of lines, treat each line as a separate document
        if len(lines) % 2 != 0:
            for i, line in enumerate(lines):
                doc = Document(
                    page_content=line,
                    metadata={
                        "type": "text_chunk",
                        "line_number": i + 1
                    }
                )
                documents.append(doc)
        else:
            # Treat pairs of lines as Q&A
            for i in range(0, len(lines), 2):
                question = lines[i]
                answer = lines[i + 1] if i + 1 < len(lines) else ""
                
                doc = Document(
                    page_content=f"Question: {question}\nAnswer: {answer}",
                    metadata={
                        "type": "qa_pair",
                        "question": question,
                        "answer": answer
                    }
                )
                documents.append(doc)
        
        return documents
    
    def get_statistics(self) -> Dict:
        """Get statistics about the processed data"""
        if not self.documents:
            return {"error": "No documents processed"}
        
        stats = {
            "total_documents": len(self.documents),
            "qa_pairs": sum(1 for doc in self.documents if doc.metadata.get("type") == "qa_pair"),
            "text_chunks": sum(1 for doc in self.documents if doc.metadata.get("type") == "text_chunk"),
            "avg_content_length": sum(len(doc.page_content) for doc in self.documents) / len(self.documents)
        }
        
        return stats