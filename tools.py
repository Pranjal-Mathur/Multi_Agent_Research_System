from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information on a topic.
    Returns Titles, URLs and snippets.
    """
    results = tavily.search(
    query=query,
    max_results=5,
    search_depth="advanced",
    include_answer=False,
    include_raw_content=False,
)

    out = []

    for r in results["results"]:
        out.append(
            f"""Title: {r['title']}
URL: {r['url']}
Snippet: {r['content'][:300]}
"""
        )

    return "\n----\n".join(out)


@tool
def scrape_url(url: str) -> str:
    """
    Scrape and return clean text content from a given URL for deeper reading.
    """
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/138.0 Safari/537.36"
                )
            },
        )

        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        return text[:3000]

    except Exception as e:
        return f"Could not scrape URL: {str(e)}"

  