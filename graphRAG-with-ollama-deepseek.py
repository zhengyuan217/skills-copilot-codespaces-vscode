# graphrag_deepseek.py
# GraphRAG with DeepSeek LLM via Ollama - Local Document Query System

import os
import json
from typing import List, Dict, Optional
import networkx as nx
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DocumentProcessor:
    """Processes documents into chunks and extracts key information"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def chunk_documents(self, documents: List[str]) -> List[Dict]:
        """Split documents into manageable chunks with metadata"""
        chunks = []
        for doc_id, doc_text in enumerate(documents):
            start = 0
            while start < len(doc_text):
                end = min(start + self.chunk_size, len(doc_text))
                chunk = doc_text[start:end]
                
                chunks.append({
                    'id': f"doc_{doc_id}_chunk_{len(chunks)}",
                    'text': chunk,
                    'doc_id': doc_id,
                    'start_pos': start,
                    'end_pos': end
                })
                
                if end == len(doc_text):
                    break
                start = end - self.chunk_overlap
                
        return chunks
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for document chunks"""
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
            
        return chunks

class KnowledgeGraph:
    """Builds and manages a knowledge graph from document chunks"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.entity_index = {}
        
    def build_graph(self, chunks: List[Dict], similarity_threshold: float = 0.7):
        """Construct a knowledge graph from document chunks"""
        # Add document chunks as nodes
        for chunk in chunks:
            self.graph.add_node(chunk['id'], type='chunk', **chunk)
            
        # Calculate similarities between chunks
        embeddings = np.array([chunk['embedding'] for chunk in chunks])
        sim_matrix = cosine_similarity(embeddings)
        
        # Add edges between similar chunks
        for i in range(len(chunks)):
            for j in range(i+1, len(chunks)):
                if sim_matrix[i][j] > similarity_threshold:
                    self.graph.add_edge(
                        chunks[i]['id'], 
                        chunks[j]['id'],
                        weight=sim_matrix[i][j]
                    )
                    
    def query_related_chunks(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Find the most relevant chunks to a query"""
        chunk_nodes = [
            node for node, data in self.graph.nodes(data=True) 
            if data['type'] == 'chunk'
        ]
        
        chunk_embeddings = np.array([
            self.graph.nodes[node]['embedding'] for node in chunk_nodes
        ])
        
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), 
            chunk_embeddings
        )[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.graph.nodes[chunk_nodes[i]] for i in top_indices]

class GraphRAGSystem:
    """End-to-end system combining GraphRAG with DeepSeek LLM"""
    
    def __init__(self, ollama_model: str = "deepseek"):
        self.processor = DocumentProcessor()
        self.knowledge_graph = KnowledgeGraph()
        self.ollama_model = ollama_model
        
    def ingest_documents(self, documents: List[str]):
        """Process and index documents into the knowledge graph"""
        print("Processing documents...")
        chunks = self.processor.chunk_documents(documents)
        chunks_with_embeddings = self.processor.embed_chunks(chunks)
        
        print("Building knowledge graph...")
        self.knowledge_graph.build_graph(chunks_with_embeddings)
        print(f"Graph built with {len(chunks)} nodes and {self.knowledge_graph.graph.number_of_edges()} edges")
        
    def query(self, question: str, max_context_chunks: int = 3) -> str:
        """Query the system with a question"""
        # Embed the question
        question_embedding = self.processor.embedding_model.encode([question])[0]
        
        # Retrieve relevant chunks
        context_chunks = self.knowledge_graph.query_related_chunks(
            question_embedding, 
            top_k=max_context_chunks
        )
        
        # Prepare context for LLM
        context = "\n\n".join([chunk['text'] for chunk in context_chunks])
        
        # Generate prompt
        prompt = f"""You are a helpful AI assistant answering questions based on the provided context.
        Context:
        {context}

        Question: {question}
        Answer: """
        
        # Query DeepSeek LLM via Ollama
        response = ollama.generate(
            model=self.ollama_model,
            prompt=prompt,
            options={
                'temperature': 0.3,
                'num_ctx': 4096  # Context window size
            }
        )
        
        return response['response']

# Example Usage
if __name__ == "__main__":
    # Initialize the system
    rag_system = GraphRAGSystem()
    
    # Load your private documents (replace with your actual documents)
    documents = [
        "This is the content of your first private document...",
        "Another document with sensitive information...",
        # Add more documents as needed
    ]
    
    # Ingest documents into the system
    rag_system.ingest_documents(documents)
    
    # Example query
    question = "What information is contained in the documents about X?"
    answer = rag_system.query(question)
    
    print("\nQuestion:", question)
    print("Answer:", answer)