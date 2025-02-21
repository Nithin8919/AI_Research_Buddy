# RAG.py
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

class RAGModule:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = None

    def build_vector_store(self, texts):
        """Build FAISS vector store from a list of texts."""
        if not texts:
            print("No texts? RAG’s got nothing to chew on!")
            return
        self.vector_store = FAISS.from_texts(texts, self.embeddings)
        print("Vector store built—ready to hunt for gold!")

    def retrieve_relevant(self, query, k=2):
        """Retrieve top k relevant documents for the query."""
        if not self.vector_store:
            print("Vector store’s empty—did you forget to build it?")
            return []
        top_docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in top_docs]

# Test it standalone (optional)
if __name__ == "__main__":
    rag = RAGModule()
    rag.build_vector_store(["AI is cool.", "Deep learning rocks."])
    results = rag.retrieve_relevant("best AI stuff")
    print(results)