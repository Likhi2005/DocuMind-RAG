# def main():
#     print("Hello from documind-rag!")


# if __name__ == "__main__":
#     main()

from services.ingestion_service import IngestionService
from services.chat_service import ChatService

if __name__ == "__main__":
    # Step 1: Ingest documents and create vector database
    ingestion_service = IngestionService()
    ingestion_service.ingest()
    
    # Step 2: Start chat service
    chat_service = ChatService()
    
    # Example Query
    user_query = "what is my name?"
    response = chat_service.chat(user_query)
    print(response)