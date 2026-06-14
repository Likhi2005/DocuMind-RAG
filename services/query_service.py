import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import dotenv

dotenv.load_dotenv()

class QueryService:
    
    def __init__(self):
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        
        self.vector_store = Chroma(
            collection_name="documind_rag",
            embedding_function=embeddings,
            persist_directory="chroma_db",
        )
        
        # Debugging: Print number of documents in vector store
        print("Documents in DB:", self.vector_store._collection.count())
    
    def retrieve_relevant_context(self, query: str):
        relevant_docs = self.vector_store.similarity_search(query, k=5)
        
        return relevant_docs
    