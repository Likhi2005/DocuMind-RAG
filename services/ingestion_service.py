import os
from langchain_docling.loader import DoclingLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
import hashlib
import dotenv
    
dotenv.load_dotenv()

UPLOAD_DIR = os.getenv("UPLOAD_DIR") or "uploads"

class IngestionService:
    
    def load_documents(self):
        # TODO: Add support for multiple file formats (PDF, DOCX, TXT, etc.)
        documents = []
        for filename in os.listdir(UPLOAD_DIR):
            filepath = os.path.join(UPLOAD_DIR,filename)

            loader = DoclingLoader(file_path=filepath)
            docs = loader.load()
            
            documents.extend(docs)
        return documents
    
    def split_documents(self, documents):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
            
            chunks = splitter.split_documents(documents)
            return chunks
    
    def create_embeddings(self):
        return HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        
    def store_in_vectordb(self, chunks):
        embeddings = self.create_embeddings()
        
        vector_store = Chroma(
            collection_name="documind_rag",
            embedding_function=embeddings,
            persist_directory="chroma_db"
        )
        
        cleaned_chunks = filter_complex_metadata(chunks)
        
        ids = [hashlib.sha256(chunk.page_content.encode()).hexdigest() for chunk in cleaned_chunks]
        
        
        
        vector_store.add_documents(
            chunks,
            ids=ids)
        return vector_store
        
    def ingest(self):
        # skip if vector database already exists
        if os.path.exists("chroma_db"):
            print("Vector database already exists. skipping ingestion.")
            return
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        vector_store = self.store_in_vectordb(chunks)
        
        return vector_store

if __name__ == "__main__":
    ingestion_service = IngestionService()
    ingestion_service.ingest()