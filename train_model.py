#!/usr/bin/env python3
"""
Optional training script for fine-tuning BERT embeddings on your specific Q&A data
"""

import os
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import json
from data_processor import QADataProcessor
from typing import List, Tuple
import random

class QAModelTrainer:
    def __init__(self, base_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the trainer with a base model.
        """
        self.base_model = base_model
        self.model = None
        self.train_examples = []
        self.eval_examples = []
    
    def prepare_training_data(self, qa_file_path: str) -> List[InputExample]:
        """
        Prepare training data from Q&A file.
        Creates positive pairs (question-answer) and negative pairs.
        """
        processor = QADataProcessor()
        documents = processor.load_qa_file(qa_file_path)
        
        examples = []
        qa_pairs = []
        
        # Extract Q&A pairs
        for doc in documents:
            if doc.metadata.get("type") == "qa_pair":
                question = doc.metadata.get("question", "")
                answer = doc.metadata.get("answer", "")
                if question and answer:
                    qa_pairs.append((question, answer))
        
        print(f"Found {len(qa_pairs)} Q&A pairs for training")
        
        # Create positive examples (question-answer pairs should be similar)
        for question, answer in qa_pairs:
            examples.append(InputExample(
                texts=[question, answer], 
                label=1.0  # High similarity
            ))
        
        # Create negative examples (questions with wrong answers)
        for i, (question, _) in enumerate(qa_pairs):
            # Get random answer from different Q&A pair
            wrong_answers = [answer for j, (_, answer) in enumerate(qa_pairs) if j != i]
            if wrong_answers:
                wrong_answer = random.choice(wrong_answers)
                examples.append(InputExample(
                    texts=[question, wrong_answer], 
                    label=0.0  # Low similarity
                ))
        
        # Create question-question similarities
        # Similar questions should have medium-high similarity
        for i in range(len(qa_pairs)):
            for j in range(i + 1, min(i + 3, len(qa_pairs))):  # Compare with next few questions
                question1 = qa_pairs[i][0]
                question2 = qa_pairs[j][0]
                
                # Calculate simple text similarity as label
                similarity = self._calculate_text_similarity(question1, question2)
                examples.append(InputExample(
                    texts=[question1, question2], 
                    label=similarity
                ))
        
        return examples
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Simple text similarity calculation based on common words.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def train_model(self, qa_file_path: str, output_path: str = "fine_tuned_model", 
                   num_epochs: int = 4, batch_size: int = 16):
        """
        Train the sentence transformer model on Q&A data.
        """
        print("🚀 Starting model training...")
        
        # Load base model
        self.model = SentenceTransformer(self.base_model)
        
        # Prepare training data
        all_examples = self.prepare_training_data(qa_file_path)
        
        # Split train/eval (80/20)
        random.shuffle(all_examples)
        split_idx = int(0.8 * len(all_examples))
        self.train_examples = all_examples[:split_idx]
        self.eval_examples = all_examples[split_idx:]
        
        print(f"Training examples: {len(self.train_examples)}")
        print(f"Evaluation examples: {len(self.eval_examples)}")
        
        # Create data loaders
        train_dataloader = DataLoader(self.train_examples, shuffle=True, batch_size=batch_size)
        
        # Define loss function
        train_loss = losses.CosineSimilarityLoss(model=self.model)
        
        # Create evaluator
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            self.eval_examples, 
            name='qa_eval'
        )
        
        # Training arguments
        warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data
        
        print(f"Training for {num_epochs} epochs with {warmup_steps} warmup steps...")
        
        # Train the model
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            evaluation_steps=500,
            save_best_model=True
        )
        
        print(f"✅ Training completed! Model saved to {output_path}")
        return output_path
    
    def evaluate_model(self, model_path: str, test_questions: List[str] = None):
        """
        Evaluate the trained model on sample questions.
        """
        if not test_questions:
            test_questions = [
                "How do I reset my password?",
                "What are the pricing plans?",
                "How to create an account?",
                "Contact support information"
            ]
        
        print("🔍 Evaluating model performance...")
        
        # Load trained model
        model = SentenceTransformer(model_path)
        
        # Encode test questions
        question_embeddings = model.encode(test_questions)
        
        print("Sample embeddings shape:", question_embeddings.shape)
        print("Model evaluation completed!")
        
        # Test similarity between questions
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity_matrix = cosine_similarity(question_embeddings)
        print("\nSimilarity matrix between test questions:")
        for i, q1 in enumerate(test_questions):
            print(f"\n{q1}:")
            for j, q2 in enumerate(test_questions):
                if i != j:
                    print(f"  vs '{q2}': {similarity_matrix[i][j]:.3f}")

def main():
    """Main training function"""
    print("🤖 BERT Model Training for RAG Chatbot")
    print("=" * 60)
    
    # Configuration
    qa_file = input("Enter path to your Q&A file (or press Enter for data/sample_qa.txt): ").strip()
    if not qa_file:
        qa_file = "data/sample_qa.txt"
    
    if not os.path.exists(qa_file):
        print(f"❌ File {qa_file} not found!")
        return
    
    base_model = input("Enter base model name (or press Enter for all-MiniLM-L6-v2): ").strip()
    if not base_model:
        base_model = "sentence-transformers/all-MiniLM-L6-v2"
    
    epochs = input("Number of training epochs (default 4): ").strip()
    epochs = int(epochs) if epochs.isdigit() else 4
    
    output_dir = input("Output directory (default: fine_tuned_model): ").strip()
    if not output_dir:
        output_dir = "fine_tuned_model"
    
    # Initialize trainer
    trainer = QAModelTrainer(base_model=base_model)
    
    try:
        # Train model
        model_path = trainer.train_model(
            qa_file_path=qa_file,
            output_path=output_dir,
            num_epochs=epochs
        )
        
        # Evaluate model
        trainer.evaluate_model(model_path)
        
        print("\n" + "=" * 60)
        print("✅ Training completed successfully!")
        print(f"📁 Fine-tuned model saved to: {model_path}")
        print("\n📋 To use the fine-tuned model:")
        print(f"1. Update vector_store.py to use model_name='{model_path}'")
        print("2. Or pass the path when initializing VectorStoreManager")
        print("3. Restart your chatbot application")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        print("Please check your data file and try again.")

if __name__ == "__main__":
    main()