from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from src.logger import logger

class RetrievalModule:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", persist_dir="./chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = None
        self.persist_dir = persist_dir  # Persistent storage

    def build_vector_store(self, texts):
        """Build Chroma vector store with better logging."""
        if not texts:
            logger.warning("No texts provided. Skipping vector store creation.")
            return
        
        self.vector_store = Chroma.from_texts(
            texts, self.embeddings, persist_directory=self.persist_dir
        )
        self.vector_store.persist()
        logger.info("Chroma vector store successfully built.")

    def retrieve_relevant(self, query, k=2):
        """Fetch top-k relevant documents, logging warnings if store is empty."""
        if not self.vector_store:
            logger.warning("Vector store is empty. Run `build_vector_store` first.")
            return []
        
        top_docs = self.vector_store.similarity_search(query, k=k)
        retrieved = [doc.page_content for doc in top_docs] if top_docs else []
        
        logger.info(f"Retrieved {len(retrieved)} relevant papers for query: '{query}'.")
        return retrieved
