import requests
import xml.etree.ElementTree as ET
from src.logger import logger

class DataIngestion:
    def __init__(self, api_url="http://export.arxiv.org/api/query"):
        self.api_url = api_url

    def fetch_papers(self, topic, max_results=5):
        """Fetch papers from arXiv with logging and better error handling."""
        url = f"{self.api_url}?search_query=all:{topic}&start=0&max_results={max_results}"
        logger.info(f"Fetching papers from: {url}")
        
        try:
            response = requests.get(url, timeout=10)  # Added timeout
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching papers: {e}")
            return [], []
        
        # Parse XML
        root = ET.fromstring(response.text)
        titles, abstracts = [], []
        
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
            titles.append(title)
            abstracts.append(abstract)
        
        logger.info(f"Fetched {len(abstracts)} papers.")
        return titles, abstracts
