# AI_agent.py
from transformers import pipeline

class ResearchAgent:
    def __init__(self, summarizer_model="facebook/bart-large-cnn"):
        self.summarizer = pipeline("summarization", model=summarizer_model)

    def summarize_papers(self, papers):
        """Summarize a list of papers into short snippets."""
        if not papers:
            print("No papers? I’m not summarizing thin air!")
            return []
        summaries = []
        for paper in papers:
            summary = self.summarizer(paper, max_length=30, min_length=15)[0]["summary_text"]
            summaries.append(summary)
        print(f"Summarized {len(summaries)} papers—short and sweet!")
        return summaries

    def pick_best(self, summaries):
        """Pick the 'best' summary—shortest wins for now."""
        if not summaries:
            print("No summaries to judge—everyone’s a loser!")
            return None
        best = min(summaries, key=len)
        print(f"Best pick chosen: {best}")
        return best

# Test it standalone (optional)
if __name__ == "__main__":
    agent = ResearchAgent()
    summaries = agent.summarize_papers(["AI is advancing fast with new tech.", "Deep learning is cool."])
    best = agent.pick_best(summaries)
    print(best)