from setuptools import setup, find_packages

setup(
    name="ai_research_buddy",
    version="0.1.0",
    author="Nithin",
    author_email="your_email@example.com",
    description="An AI-powered research assistant using RAG, FAISS, LangChain, and Transformers.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Nithin8919/AI_Research_Buddy",
    packages=find_packages(),
    install_requires=[
        "requests",
        "langchain",
        "faiss-cpu",
        "transformers",
        "torch",
        "sentence-transformers",
        "langchain_community",
        "faiss-cpu",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
