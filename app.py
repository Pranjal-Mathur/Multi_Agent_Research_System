import time
import re
import streamlit as st
from groq import BadRequestError

from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
)

st.set_page_config(page_title="Multi-Agent Research System", page_icon="🔎", layout="wide")

st.title("🔎 Multi-Agent Research System")
st.caption("Search Agent → Reader Agent → Writer → Critic ")


def invoke_with_retry(agent, payload, retries=3, delay=2):
    """Retry agent.invoke on Groq tool_use_failed / malformed tool-call errors."""
    last_err = None
    for attempt in range(retries):
        try:
            return agent.invoke(payload)
        except BadRequestError as e:
            last_err = e
            st.warning(f"Tool call failed (attempt {attempt + 1}/{retries}), retrying...")
            time.sleep(delay)
    raise RuntimeError(f"Agent failed after {retries} attempts: {last_err}")


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About")
    st.write(
        "This app runs a 4-stage pipeline:\n\n"
        "1. **Search Agent** — searches the web \n"
        "2. **Reader Agent** — scrapes the  URL\n"
        "3. **Writer** — drafts a structured report\n"
        "4. **Critic** — scores and reviews the report"
    )
    st.divider()
    show_intermediate = st.checkbox("Show intermediate steps", value=True)


# ---------------- Main input ----------------
topic = st.text_input("Enter a research topic", placeholder="e.g. CJP Protest")
run_button = st.button("Run Research Pipeline", type="primary")

if run_button:
    if not topic.strip():
        st.error("Please enter a topic first.")
        st.stop()

    state = {}

    try:
        # ---------------- Step 1: Search Agent ----------------
        with st.status("Step 1 — Searching the web...", expanded=show_intermediate) as status:
            search_agent = build_search_agent()

            search_result = invoke_with_retry(
                search_agent,
                {
                    "messages": [
                        (
                            "user",
                            f"Search the web for detailed information about '{topic}'. "
                            "Return the raw search results with Title, URL and Snippet."
                        )
                    ]
                },
            )

            state["search_results"] = search_result["messages"][-1].content
            urls = re.findall(r"https?://[^\s]+", state["search_results"])

            if show_intermediate:
                st.text_area("Raw search results", state["search_results"], height=200)
                st.write("**URLs found:**", urls if urls else "None")

            status.update(label="Step 1 — Search complete ✅", state="complete")

        # ---------------- Step 2: Reader Agent ----------------
        with st.status("Step 2 — Reading best source...", expanded=show_intermediate) as status:
            reader_agent = build_reader_agent()

            reader_result = invoke_with_retry(
                reader_agent,
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
                },
            )

            state["scraped_content"] = reader_result["messages"][-1].content

            if show_intermediate:
                st.text_area("Scraped content", state["scraped_content"], height=200)

            status.update(label="Step 2 — Reading complete ✅", state="complete")

        # ---------------- Step 3: Writer ----------------
        with st.status("Step 3 — Writing report...", expanded=False) as status:
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
            status.update(label="Step 3 — Report drafted ✅", state="complete")

        # ---------------- Step 4: Critic ----------------
        with st.status("Step 4 — Reviewing report...", expanded=False) as status:
            state["feedback"] = critic_chain.invoke({"report": state["report"]})
            status.update(label="Step 4 — Review complete ✅", state="complete")

        # ---------------- Final output ----------------
        st.success("Pipeline finished!")

        tab1, tab2 = st.tabs(["📄 Report", "🧪 Critic Feedback"])

        with tab1:
            st.markdown(state["report"])
            st.download_button(
                "Download report as .md",
                data=state["report"],
                file_name=f"{topic.replace(' ', '_')}_report.md",
                mime="text/markdown",
            )

        with tab2:
            st.markdown(state["feedback"])

    except Exception as e:
        st.error(f"Pipeline failed: {e}")

