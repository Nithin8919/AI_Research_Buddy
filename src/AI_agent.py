# AI_agent.py
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from src.data_ingestion import DataIngestion
from src.RAG import RetrievalModule
from transformers import pipeline
import sqlite3
import time
from src.logger import logger

# Load LLaMA with llama.cpp—simple chatter
llm = LlamaCpp(
    model_path="/Users/nitin/Downloads/llama-2-7b-chat.Q4_0.gguf",  # Update this!
    n_ctx=512,  # Fits 8 GB
    n_threads=4,  # Fast on M3 Pro
    temperature=0.7,
    max_tokens=150,
    verbose=True
)

# Instances
data_ingestion = DataIngestion()
retrieval_module = RetrievalModule()

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Database Setup
conn = sqlite3.connect("research_data.db")
cursor = conn.cursor()
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS papers (
        query TEXT,
        retrieved_papers TEXT,
        summary TEXT,
        evaluation TEXT
    )
    """
)
conn.commit()

# Tools (just functions now)
def retrieve_relevant_papers(topic: str) -> str:
    """Fetch and retrieve relevant papers."""
    titles, abstracts = data_ingestion.fetch_papers(topic)
    if not abstracts:
        logger.warning(f"No papers retrieved for topic: {topic}")
        return "Could not retrieve papers."
    retrieval_module.build_vector_store(abstracts)
    relevant_sections = retrieval_module.retrieve_relevant(topic)
    logger.info(f"Retrieved {len(relevant_sections)} relevant papers for {topic}")
    return "\n".join(relevant_sections)

def summarize_text(text: str) -> str:
    """Summarize text using DistilBART."""
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", device="mps")
    text = text[:500]  # Keep it short
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
    logger.info("Generated summary for retrieved papers")
    return summary

def evaluate_summary(summary: str) -> str:
    """Evaluate summary quality with LLaMA."""
    prompt = f"Evaluate this summary for accuracy, completeness, and clarity: {summary[:200]}"
    evaluation = llm(prompt)
    logger.info("Evaluated summary quality")
    return evaluation

# Simple Conversational Chain—no retriever needed
class ResearchAssistant:
    def __init__(self):
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "query"],
            template="You are a research assistant. Based on the chat history and query, provide a helpful response.\n\nChat History: {chat_history}\nQuery: {query}\n\nResponse: "
        )
        self.chain = LLMChain(llm=llm, prompt=self.prompt, memory=memory)

    def process_query(self, query: str) -> tuple:
        """Process query with retries—no ReAct mess."""
        retries = 0
        max_retries = 3

        while retries < max_retries:
            try:
                # Step 1: Retrieve papers
                retrieved_papers = retrieve_relevant_papers(query)
                if "Could not retrieve papers" in retrieved_papers:
                    query = f"more detailed {query}"
                    retries += 1
                    time.sleep(2)
                    continue

                # Step 2: Summarize
                summary = summarize_text(retrieved_papers)
                if len(summary.split()) < 10:
                    retries += 1
                    time.sleep(2)
                    continue

                # Step 3: Evaluate
                evaluation = evaluate_summary(summary)

                # Save to memory and DB
                memory.save_context(
                    {"input": query},
                    {"output": f"Summary: {summary}\nEvaluation: {evaluation}\nAsk me anything about these findings!"}
                )
                cursor.execute(
                    "INSERT INTO papers (query, retrieved_papers, summary, evaluation) VALUES (?, ?, ?, ?)",
                    (query, retrieved_papers, summary, evaluation)
                )
                conn.commit()
                return summary, evaluation

            except Exception as e:
                logger.error(f"Error in processing: {str(e)}")
                retries += 1
                time.sleep(2)

        logger.error("Max retries reached—task failed.")
        return "Failed after retries.", "N/A"

    def chat(self, user_input: str) -> str:
        """Handle follow-up chats."""
        if not memory.chat_memory.messages:
            return "Please start with a research query like 'large language model memory optimization'."
        return self.chain.run(query=user_input)

if __name__ == "__main__":
    assistant = ResearchAssistant()
    query = "large language model memory optimization"
    summary, evaluation = assistant.process_query(query)
    print("Summary:", summary)
    print("Evaluation:", evaluation)
    # Test follow-up
    follow_up = "Tell me more about memory optimization."
    print("Follow-up:", assistant.chat(follow_up))
    conn.close()