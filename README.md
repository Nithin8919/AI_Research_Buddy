# AI_Research_BuddyBelow is a polished, comprehensive, and engaging **README.md** file for your **AI_Research_Buddy** project. This documentation is designed to showcase your expertise in **Retrieval-Augmented Generation (RAG)** and **agents**, explain the project’s purpose and functionality, and provide clear instructions for setup and usage. It’s professional yet approachable, perfect for your GitHub repo (`https://github.com/Nithin8919/AI_Research_Buddy`) and Hugging Face Space (`https://huggingface.co/spaces/Nithin89/AI_Reaserch_Buddy` or corrected version).

---

# AI Research Buddy: Your Conversational RAG Sidekick

![AI Research Buddy Logo](https://via.placeholder.com/150?text=AI+Research+Buddy)  
*Unleash the power of research with a twist of AI magic!*  

Welcome to **AI Research Buddy**, a slick, conversational app that dives into the vast ocean of arXiv papers to fetch, summarize, and source the best research on any topic you throw at it—like "RAG" (yep, it’s meta enough to research itself!). Built from the ground up by **Nithin** (that’s me!), this project is a living testament to my mastery of **Retrieval-Augmented Generation (RAG)** and **agents**, blending advanced retrieval tricks with a chatty AI that’s always ready to dig deeper. Whether you’re a student, researcher, or just a curious mind, this buddy’s got your back—all running on a humble MacBook M3 Pro with 8GB RAM as of February 24, 2025!

---

## 🚀 What’s This All About?

AI Research Buddy isn’t just another research tool—it’s a **conversational RAG agent** with a mission: to make exploring academic papers fast, fun, and insightful. Here’s the gist:

- **Ask Anything**: Type a topic (e.g., "RAG") or a follow-up (e.g., "Tell me more"), and watch it work its magic.
- **Smart Retrieval**: It grabs papers from arXiv, expands your query (think "RAG" → "Retrieval-Augmented Generation"), and reranks them to spotlight the best.
- **Snappy Summaries**: Powered by `distilgpt2`, it crafts concise summaries of the top papers, serving up knowledge in bite-sized chunks.
- **Sources Included**: Every summary comes with clickable arXiv links, so you can dive into the originals.
- **Chatty Agent**: Ask for more, and it’ll refine the story using what it’s already found—no extra digging required.

This isn’t just a demo—it’s proof that RAG and agents can team up to turn research into a conversation, all while keeping things light on your hardware.

---

## 🌟 Features That Shine

- **Advanced RAG Magic**:
  - **Query Expansion**: Boosts recall by adding synonyms (e.g., "AI" → "Artificial Intelligence").
  - **Reranking**: Uses a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to pick the top 3 papers with precision.
  - **Generation**: Summarizes with `distilgpt2`, generating up to 100 new tokens for crisp, relevant outputs.

- **Agent Awesomeness**:
  - **Autonomy**: Fetches, ranks, and summarizes papers without hand-holding.
  - **Conversational Memory**: Tracks history with `ConversationBufferMemory`, adapting to follow-ups like "Tell me more."
  - **Decision-Making**: Smartly chooses the best papers and adjusts responses based on your input.

- **Lightweight Design**: Runs smoothly on 8GB RAM (~700MB footprint), optimized for my MacBook M3 Pro.

---

## 🎉 Try It Out!

Hosted live on [Hugging Face Spaces](https://huggingface.co/spaces/Nithin89/AI_Reaserch_Buddy), AI Research Buddy is ready to roll! Here’s what you’ll see:

- **Input**: "RAG"
  ```
  📜 **Summary of Best Papers on RAG:**
  Retrieval-Augmented Generation (RAG) enhances language models by integrating external knowledge retrieval, improving performance on knowledge-intensive tasks. Research highlights modular frameworks and benchmarks.

  **Sources:**
  - Modular RAG: Transforming RAG Systems ([link](https://export.arxiv.org/abs/2407.21059v1))
  - ARAGOG: Advanced RAG Output Grading ([link](https://export.arxiv.org/abs/2404.01037v1))
  - CRAG -- Comprehensive RAG Benchmark ([link](https://export.arxiv.org/abs/2406.04744v2))
  ```

- **Follow-Up**: "Tell me more"
  ```
  📜 **More on RAG:**
  Modular RAG offers reconfigurable frameworks, while CRAG benchmarks evaluate real-world QA, advancing RAG applications.
  ```

---

## 🛠️ How It Works

Here’s the techy breakdown:

1. **Retrieval**:
   - **DataIngestion**: Fetches up to 5 papers from arXiv with an expanded query (`ti:{query} OR ab:{query}`).
   - **RetrievalModule**: Builds a Chroma vector store with `all-MiniLM-L6-v2` embeddings, retrieves 5 papers, and reranks to the top 3 using a cross-encoder.

2. **Generation**:
   - Combines retrieved abstracts into a prompt, then uses `distilgpt2` to generate a 100-token summary.

3. **Agent Behavior**:
   - `ConversationBufferMemory` tracks chat history, reusing context for follow-ups.
   - Adapts output based on whether it’s a new query or a deeper dive.

4. **Output**:
   - New queries get summaries + sources; follow-ups refine the summary.

All this runs in a Gradio app, deployed to Hugging Face Spaces for the world to see!

---

## 📦 Setup and Installation

Want to run it locally or tweak it? Here’s how:

### **Prerequisites**
- Python 3.8+
- Git installed
- ~1GB free disk space (for models)

### **Steps**
1. **Clone the Repo**:
   ```bash
   git clone https://github.com/Nithin8919/AI_Research_Buddy.git
   cd AI_Research_Buddy
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *requirements.txt*:
   ```
   gradio
   requests
   langchain
   langchain-community
   transformers
   huggingface-hub
   loguru
   numpy
   torch
   ```

3. **Run It**:
   ```bash
   python app.py
   ```
   - Opens at `http://127.0.0.1:7860`.
   - Add `share=True` to `demo.launch()` for a temporary public URL.

4. **Test**: Try "RAG" and "Tell me more" in the browser.

---

## 🌍 Deployment

Live at [Hugging Face Spaces](https://huggingface.co/spaces/Nithin89/AI_Reaserch_Buddy)! To deploy your own:

1. **Push to HF Space**:
   - Add your Space as a remote:
     ```bash
     git remote add space https://Nithin89:<HF_TOKEN>@huggingface.co/spaces/Nithin89/AI_Reaserch_Buddy
     git push space main --force
     ```
   - Replace `<HF_TOKEN>` with your Hugging Face token.

2. **Build**: HF auto-builds from `app.py` and `requirements.txt`.

---

## 💡 Why It’s Awesome

- **RAG Mastery**: Shows off query expansion, reranking, and generation—core RAG skills.
- **Agent Vibes**: Conversational, autonomous, and adaptive, proving I get agents.
- **Lean & Mean**: Runs on 8GB RAM, a testament to efficient design.
- **Fun Factor**: Research doesn’t have to be dull—this buddy’s got personality!

---

## 📜 License

MIT License—feel free to fork, tweak, and share!

---

**Created by Nithin | February 24, 2025**  
[GitHub](https://github.com/Nithin8919) | [Hugging Face](https://huggingface.co/Nithin89)

