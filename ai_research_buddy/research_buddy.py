# research_buddy.py
from src.data_ingestion import DataIngestion
from src.RAG import RAGModule
from src.AI_agent import ResearchAgent

class ResearchBuddyPipeline:
    def __init__(self, config=None):
        # Default config—customizable via a dict if you want
        self.config = config or {
            "api_url": "http://export.arxiv.org/api/query",
            "embedding_model": "all-MiniLM-L6-v2",
            "summarizer_model": "facebook/bart-large-cnn",
            "max_results": 5,
            "top_k": 2
        }
        # Initialize modules with config
        self.ingestor = DataIngestion(api_url=self.config["api_url"])
        self.rag = RAGModule(embedding_model=self.config["embedding_model"])
        self.agent = ResearchAgent(summarizer_model=self.config["summarizer_model"])
        self.titles = []  # Store titles for reference
        self.abstracts = []  # Store abstracts for RAG

    def run(self, topic, query):
        """Run the full pipeline: fetch → retrieve → summarize → pick."""
        print("Research Buddy: Let’s crank up the knowledge machine!")
        
        # Step 1: Fetch papers
        self.titles, self.abstracts = self.ingestor.fetch_papers(topic, self.config["max_results"])
        if not self.abstracts:
            print("No papers fetched—time to cry or try again!")
            return

        # Step 2: Build RAG and retrieve
        self.rag.build_vector_store(self.abstracts)
        relevant_papers = self.rag.retrieve_relevant(query, k=self.config["top_k"])
        print(f"Top papers retrieved: {relevant_papers}")

        # Step 3: Summarize and pick best
        summaries = self.agent.summarize_papers(relevant_papers)
        best_summary = self.agent.pick_best(summaries)

        # Step 4: Show off the results
        print("\nResults:")
        for i, summary in enumerate(summaries, 1):
            print(f"Paper {i}: {summary}")
        print(f"Best Pick: {best_summary}")
        print("Research Buddy: Mission accomplished—go be brilliant!")

# Run the pipeline
if __name__ == "__main__":
    pipeline = ResearchBuddyPipeline()
    pipeline.run(topic="deep learning", query="best deep learning advancements")