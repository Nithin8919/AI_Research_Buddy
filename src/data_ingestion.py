# data_ingestion.py
import requests
import xml.etree.ElementTree as ET

class DataIngestion:
    def __init__(self, api_url="http://export.arxiv.org/api/query"):
        self.api_url = api_url

    def fetch_papers(self, topic, max_results=5):
        """Fetch papers from arXiv API and return titles and abstracts."""
        url = f"{self.api_url}?search_query=all:{topic}&start=0&max_results={max_results}"
        response = requests.get(url)
        if response.status_code != 200:
            print("Oops, the API’s sulking—fetch failed!")
            return [], []

        # Parse XML response
        root = ET.fromstring(response.text)
        titles = []
        abstracts = []

        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
            titles.append(title)
            abstracts.append(abstract)

        print(f"Fetched {len(abstracts)} papers—time to dig in!")
        return titles, abstracts

# Test it standalone (optional)
if __name__ == "__main__":
    ingestor = DataIngestion()
    titles, abstracts = ingestor.fetch_papers("deep learning")
    print(titles[0], abstracts[0])