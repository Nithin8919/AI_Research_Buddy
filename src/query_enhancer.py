# query_enhancer.py
from llama_cpp import Llama

class QueryEnhancer:
    def __init__(self, model_path="TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q4_0.gguf"):
        """Load LLaMA model with llama.cpp for query enhancement."""
        try:
            self.model = Llama(
                model_path=f"{model_path}/{model_file}",  # Full path or download manually
                n_ctx=512,  # Context length—keep it small for 8 GB
                n_threads=4  # Use 4 CPU threads—fast on M3 Pro
            )
            print("LLaMA-2-7B loaded successfully with llama.cpp.")
        except Exception as e:
            raise RuntimeError(f"Failed to load LLaMA-2-7B: {str(e)}")
    
    def enhance_query(self, user_query):
        """Refine user queries for arXiv search."""
        prompt = (
            f"You are a research assistant. Improve this search query for better research paper results:\n"
            f"Original: {user_query}\n"
            f"Refined: "
        )
        result = self.model(
            prompt,
            max_tokens=50,
            temperature=0.7,
            stop=["\n"]  # Stop at newline for clean output
        )
        refined_query = result["choices"][0]["text"].strip()
        return refined_query

if __name__ == "__main__":
    # Manually download model to local path if needed
    enhancer = QueryEnhancer(model_path="Downloads/llama-2-7b-chat.Q4_0.gguf ~/models/", model_file="llama-2-7b-chat.Q4_0.gguf")
    print("Enhanced Query:", enhancer.enhance_query("AI in healthcare"))