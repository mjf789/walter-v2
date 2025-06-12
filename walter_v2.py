# walter_v2.py
"""
Core RAG model logic - handles all LlamaIndex operations
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class PsychologyRAG:
    """Main RAG model for psychology research papers"""
    
    def __init__(self, persist_dir: str = "./storage", data_dir: str = "./psych_pdfs"):
        self.persist_dir = persist_dir
        self.data_dir = data_dir
        self.index = None
        
    def load_or_create_index(self) -> Optional[VectorStoreIndex]:
        """Load existing index or create new one"""
        
        # Check if we have a persisted index
        if os.path.exists(self.persist_dir):
            print("Loading existing index...")
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(storage_context)
            print("✅ Loaded existing index!")
            return self.index
        
        # Create new index
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory '{self.data_dir}' not found!")
        
        files = os.listdir(self.data_dir)
        if not files:
            raise ValueError(f"No files found in '{self.data_dir}'!")
        
        print(f"Creating new index from {len(files)} files...")
        
        # The famous 5 lines!
        documents = SimpleDirectoryReader(self.data_dir).load_data()
        self.index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True
        )
        
        # Persist for next time
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        print(f"✅ Created index from {len(documents)} documents!")
        
        return self.index
    
    def query(self, 
              question: str, 
              model_name: str = "gpt-3.5-turbo",
              temperature: float = 0.1,
              top_k: int = 3,
              streaming: bool = True) -> Dict[str, Any]:
        """Query the index with a question"""
        
        if not self.index:
            raise ValueError("Index not loaded! Call load_or_create_index() first.")
        
        # Configure LLM
        llm = OpenAI(model=model_name, temperature=temperature)
        
        # Create query engine
        query_engine = self.index.as_query_engine(
            llm=llm,
            similarity_top_k=top_k,
            streaming=streaming
        )
        
        # Execute query
        response = query_engine.query(question)
        
        # Package response
        result = {
            "response": response,
            "streaming": streaming and hasattr(response, 'response_gen')
        }
        
        return result
    
    def rebuild_index(self) -> None:
        """Force rebuild the index by deleting storage"""
        import shutil
        
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
            print("Deleted existing index")
        
        self.index = None
        self.load_or_create_index()
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        if not self.index:
            return {"loaded": False}
        
        stats = {
            "loaded": True,
            "document_count": len(self.index.docstore.docs) if hasattr(self.index, 'docstore') else "Unknown",
            "persist_dir": self.persist_dir,
            "data_dir": self.data_dir
        }
        
        return stats


# Convenience functions for simple usage
def create_rag_model(persist_dir: str = "./storage", data_dir: str = "./psych_pdfs") -> PsychologyRAG:
    """Create and initialize a RAG model"""
    rag = PsychologyRAG(persist_dir, data_dir)
    rag.load_or_create_index()
    return rag


def quick_query(question: str, rag_model: Optional[PsychologyRAG] = None) -> str:
    """Quick query function for testing"""
    if not rag_model:
        rag_model = create_rag_model()
    
    result = rag_model.query(question)
    response = result["response"]
    
    if result["streaming"] and hasattr(response, 'response_gen'):
        # Collect streaming response
        full_response = ""
        for text in response.response_gen:
            full_response += text
        return full_response
    else:
        return str(response)