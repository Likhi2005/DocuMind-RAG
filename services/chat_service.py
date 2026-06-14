import os

from services.query_service import QueryService
from langchain_groq import ChatGroq
import dotenv
    
dotenv.load_dotenv()

class ChatService:
    def __init__(self):
        self.query_service = QueryService()
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )

    def _prompt_template(self, context: str, query: str):
        template = f"""
        You are a helpful AI assistant for question answering.

        Your task is to answer the user's question using ONLY the provided context.

        Instructions:
        1. Read the context carefully before answering.
        2. Use only the information present in the context.
        3. Do not make up facts or use outside knowledge.
        4. If the answer cannot be found in the context, say:
        "I couldn't find enough information in the provided documents to answer this question."
        5. Keep the answer clear, accurate, and concise.
        6. If relevant, present the answer using bullet points.
        7. Preserve important names, dates, numbers, and technical terms exactly as they appear in the context.
        
        Context:
        {context}
        
        Question:
        {query}
        
        Answer:
        """
        return template
        
    def chat(self, query: str):
        

        relevant_docs = self.query_service.retrieve_relevant_context(query)
        # Debugging: Print retrieved documents
        print("\nRetrieved Chunks:")
        for i,doc in enumerate(relevant_docs):
            print(f"\nChunk {i}:")
            print(doc.page_content[:500])  # Print first 500 characters of each chunk for brevity
        
        if not relevant_docs:
            return (
                "I couldn't find enough information in the "
                "provided documents to answer this question."
            )
        
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = self._prompt_template(context, query)
        
        response = self.llm.invoke(prompt)
        
        return response

if __name__ == "__main__":
    chat_service = ChatService()
    
    # Example Query
    # user_query = "what is AI"
    # response = chat_service.chat(user_query)
    # print(response)