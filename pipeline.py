import re
import time

from langchain_core.messages import ToolMessage

from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
)

SLEEP_BETWEEN_CALLS = 60


def wait():
    print(f"\nWaiting {SLEEP_BETWEEN_CALLS} seconds...")
    time.sleep(SLEEP_BETWEEN_CALLS)


def get_tool_output(messages, tool_name):
    """
    Extract the output of a specific tool from the agent messages.
    """
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == tool_name:
            return msg.content
    return ""


def run_research_pipeline(topic):

    state = {}

    print("\n" + " =" * 50)
    print("Step 1 - Search Agent")
    print("=" * 50)

    search_agent = build_search_agent()

    search_result = search_agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"Search for {topic}"
                )
            ]
        },
        config={"recursion_limit": 10},
    )

    state["search_results"] = get_tool_output(
        search_result["messages"],
        "web_search",
    )

    print(state["search_results"])

    urls = re.findall(r"https?://\S+", state["search_results"])

    print("\nURLs Found:")
    for url in urls:
        print(url)

    if not urls:
        print("\nWARNING: No URLs returned by Search Agent.")
        return

    wait()

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
Search Results:

{state['search_results']}

Choose the single best URL.

Call scrape_url.

Return ONLY the scraped content.
"""
                )
            ]
        },
        config={"recursion_limit": 10},
    )

    state["scraped_content"] = get_tool_output(
        reader_result["messages"],
        "scrape_url",
    )

    print(state["scraped_content"])

    wait()

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

    return state


if __name__ == "__main__":
    topic = input("Enter research topic: ")
    
    run_research_pipeline(topic)