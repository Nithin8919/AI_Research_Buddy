import gradio as gr
import requests
import xml.etree.ElementTree as ET
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from loguru import logger
import numpy as np

# --- DataIngestion Class with Query Expansion ---
class DataIngestion:
    def __init__(self, api_url="http://export.arxiv.org/api/query"):
        self.api_url = api_url
        self.synonyms = {
            "RAG": "Retrieval-Augmented Generation",
            "AI": "Artificial Intelligence",
            "ML": "Machine Learning"
        }

    def expand_query(self, query):
        expanded = query
        for key, value in self.synonyms.items():
            if key.lower() in query.lower():
                expanded += f" OR {value}"
        logger.info(f"Expanded query: {expanded}")
        return expanded

    def fetch_papers(self, topic, max_results=5):
        expanded_query = self.expand_query(topic)
        url = f"{self.api_url}?search_query=ti:{expanded_query}+OR+ab:{expanded_query}&start=0&max_results={max_results}"
        logger.info(f"Fetching papers from: {url}")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching papers: {e}")
            return [], [], []
        root = ET.fromstring(response.text)
        titles, abstracts, paper_ids = [], [], []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
            paper_id_elem = entry.find("{http://www.w3.org/2005/Atom}id")
            paper_id = paper_id_elem.text.split("abs/")[-1].strip() if paper_id_elem is not None else "unknown"
            titles.append(title)
            abstracts.append(abstract)
            paper_ids.append(paper_id)
        logger.info(f"Fetched {len(abstracts)} papers.")
        return titles, abstracts, paper_ids

# --- RetrievalModule Class with Reranking ---
class RetrievalModule:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", persist_dir="./chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = None
        self.persist_dir = persist_dir
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.reranker_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def build_vector_store(self, abstracts, titles, paper_ids):
        if not abstracts:
            logger.warning("No abstracts provided. Skipping vector store creation.")
            return
        metadatas = [{"title": title, "paper_id": pid} for title, pid in zip(titles, paper_ids)]
        self.vector_store = Chroma.from_texts(
            texts=abstracts, embedding=self.embeddings, metadatas=metadatas, persist_directory=self.persist_dir
        )
        self.vector_store.persist()
        logger.info("Chroma vector store built.")

    def rerank(self, query, retrieved):
        if not retrieved:
            return retrieved
        inputs = [f"{query} [SEP] {doc[0]}" for doc in retrieved]
        tokenized = self.reranker_tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
        scores = self.reranker_model(**tokenized).logits.squeeze().detach().numpy()
        ranked_indices = np.argsort(scores)[::-1]
        return [retrieved[i] for i in ranked_indices[:3]]

    def retrieve_relevant(self, query, k=5):
        if not self.vector_store:
            logger.warning("Vector store empty. Run `build_vector_store` first.")
            return []
        top_docs = self.vector_store.similarity_search(query, k=k)
        retrieved = [(doc.page_content, doc.metadata) for doc in top_docs]
        reranked = self.rerank(query, retrieved)
        logger.info(f"Retrieved and reranked {len(reranked)} papers for query: '{query}'.")
        return reranked

# --- Main Application Logic ---
data_ingestion = DataIngestion()
retrieval_module = RetrievalModule()
generator = pipeline("text-generation", model="distilgpt2")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def process_query(query):
    """Retrieve and summarize the best papers with their sources."""
    try:
        # Check chat history for follow-up context
        history = memory.load_memory_variables({})["chat_history"]
        if history and "more" in query.lower():
            last_output = history[-1]["content"] if history else ""
            context = "\n".join([line for line in last_output.split("\n") if "Summary" in line])
        else:
            # Fetch and retrieve papers for new query
            titles, abstracts, paper_ids = data_ingestion.fetch_papers(query)
            if not abstracts:
                return "No papers found after query expansion."
            retrieval_module.build_vector_store(abstracts, titles, paper_ids)
            retrieved = retrieval_module.retrieve_relevant(query)
            if not retrieved:
                return "No relevant papers retrieved."
            retrieved_abstracts = [item[0] for item in retrieved]
            retrieved_metadata = [item[1] for item in retrieved]
            context = "\n".join(retrieved_abstracts)
            memory.save_context({"input": "Retrieved papers"}, {"output": context})

        # Generate a concise summary of the best papers
        prompt = f"Summarize the best research papers on {query} based on these abstracts:\n{context}"
        summary = generator(prompt, max_new_tokens=100, num_return_sequences=1, truncation=True)[0]["generated_text"]
        
        # Include sources if not a follow-up
        if "more" not in query.lower():
            papers_ref = "\n".join([f"- {m['title']} ([link](https://export.arxiv.org/abs/{m['paper_id']}))" for m in retrieved_metadata])
            full_output = f"📜 **Summary of Best Papers on {query}:**\n{summary}\n\n**Sources:**\n{papers_ref}"
        else:
            full_output = f"📜 **More on {query}:**\n{summary}"

        memory.save_context({"input": query}, {"output": full_output})
        return full_output

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return f"Error: {str(e)}"

# --- Gradio Interface ---
demo = gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(label="Enter your research query (e.g., 'RAG' or 'Tell me more')"),
    outputs=gr.Textbox(label="Result"),
    title="Conversational RAG Demo",
    description="Retrieve summaries of the best papers on your topic with their sources. Ask follow-ups like 'Tell me more.'"
)

demo.launch()