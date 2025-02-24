import random
from src.data_ingestion import DataIngestion
from src.RAG import RetrievalModule
from src.AI_agent import ResearchAgent
from src.logger import logger

class ResearchBuddyPipeline:
    def __init__(self, config):
        self.config = config
        self.ingestor = DataIngestion(self.config["api_url"])
        self.retriever = RetrievalModule(self.config["embedding_model"], self.config["persist_dir"])
        self.agent = ResearchAgent(self.config["summarizer_model"])
        self.openers = [
            "Hold my coffee, I’m diving into this!",
            "Time to unleash my inner paper monster!",
            "Buckle up, we’re raiding the research jungle!",
            "Let’s crank this up to eleven—here we go!"
        ]

    def process_query(self, topic, query):
        opener = random.choice(self.openers)
        logger.info(f"Processing query for topic: {topic}")

        titles, abstracts = self.ingestor.fetch_papers(topic, self.config["max_results"])
        if not abstracts:
            return f"{opener}\n\nNo research found for '{topic}'. Try a different topic?"
        
        summaries = self.agent.summarize_papers(abstracts)
        self.retriever.build_vector_store(summaries)
        relevant_papers = self.retriever.retrieve_relevant(query, k=self.config["top_k"])

        if not relevant_papers:
            return f"{opener}\n\nNo relevant results for '{query}'. Try refining your query?"
        
        return f"{opener}\n\n" + self.agent.chat_response(None, relevant_papers, topic, query)
