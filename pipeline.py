import re

from agents import (
    build_reader_agent,
    build_search_agent,
    writer_chain,
    critic_chain,
)


def run_research_pipeline(topic: str):

    state = {}

    # ---------------- Search Agent ----------------
    print("\n" + " =" * 50)
    print("Step 1 - Search Agent")
    print("=" * 50)

    search_agent = build_search_agent()

    search_result = search_agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"Search the web for detailed information about '{topic}'. "
                    "Return the raw search results with Title, URL and Snippet."
                )
            ]
        }
    )

    state["search_results"] = search_result["messages"][-1].content

    print(state["search_results"])

    # verify URLs exist
    urls = re.findall(r"https?://[^\s]+", state["search_results"])

    print("\nURLs found:")
    print(urls)

    # ---------------- Reader Agent ----------------
    print("\n" + " =" * 50)
    print("Step 2 - Reader Agent")
    print("=" * 50)

    reader_agent = build_reader_agent()

    reader_result = reader_agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"""
Below are search results.

{state['search_results']}

Extract the best URL.

Use the scrape_url tool.

Return ONLY the scraped article.
"""
                )
            ]
        }
    )

    state["scraped_content"] = reader_result["messages"][-1].content

    print(state["scraped_content"])

    # ---------------- Writer ----------------
    print("\n" + " =" * 50)
    print("Step 3 - Writer")
    print("=" * 50)

    research = f"""
SEARCH RESULTS

{state['search_results']}


SCRAPED CONTENT

{state['scraped_content']}
"""

    state["report"] = writer_chain.invoke(
        {
            "topic": topic,
            "research": research,
        }
    )

    print(state["report"])

    # ---------------- Critic ----------------
    print("\n" + " =" * 50)
    print("Step 4 - Critic")
    print("=" * 50)

    state["feedback"] = critic_chain.invoke(
        {
            "report": state["report"],
        }
    )

    print(state["feedback"])

    return state


if __name__ == "__main__":
    topic = input("Enter research topic: ")
    run_research_pipeline(topic)