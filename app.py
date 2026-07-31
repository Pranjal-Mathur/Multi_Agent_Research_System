import time
import re
import traceback

import streamlit as st
from langchain_core.messages import ToolMessage

from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
)

st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Multi-Agent Research System")
st.caption("Search Agent → Reader Agent → Writer")

SLEEP_BETWEEN_CALLS = 60


def wait():
    with st.spinner("Generating..."):
        time.sleep(SLEEP_BETWEEN_CALLS)


def extract_text(message):

    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for block in content:

            if isinstance(block, str):
                parts.append(block)

            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])

        return "\n".join(parts)

    return str(content)


def get_tool_output(messages, tool_name):

    for msg in messages:

        if isinstance(msg, ToolMessage):

            if msg.name == tool_name:
                return msg.content

    return ""


def invoke_with_retry(agent, payload, retries=3, delay=3):

    last_error = None

    for attempt in range(retries):

        try:
            return agent.invoke(payload)

        except Exception as e:

            last_error = e

            st.warning(
                f"Attempt {attempt+1}/{retries} failed. Retrying..."
            )

            time.sleep(delay)

    raise RuntimeError(
        f"Agent failed after {retries} retries:\n{last_error}"
    )


# ---------------- Sidebar ----------------

with st.sidebar:

    st.header("About")

    st.write(
        """
This app runs a 3-stage research pipeline.

1. 🔍 Search Agent
2. 📖 Reader Agent
3. ✍️ Writer
"""
    )

    st.divider()

    show_intermediate = st.checkbox(
        "Show intermediate steps",
        value=True,
    )


# ---------------- Input ----------------

topic = st.text_input(
    "Enter a research topic",
    placeholder="e.g. System Design for Freshers",
)

run_button = st.button(
    "Run Research Pipeline",
    type="primary",
)

if run_button:

    if not topic.strip():

        st.error("Please enter a research topic.")

        st.stop()

    state = {}

    try:

        # --------------------------------------------------
        # SEARCH AGENT
        # --------------------------------------------------

        with st.status(
            "Step 1 — Searching the web...",
            expanded=show_intermediate,
        ) as status:

            search_agent = build_search_agent()

            search_result = invoke_with_retry(
                search_agent,
                {
                    "messages": [
                        (
                            "user",
                            f"Search for {topic}"
                        )
                    ]
                },
            )

            state["search_results"] = get_tool_output(
                search_result["messages"],
                "web_search",
            )

            urls = re.findall(
                r"https?://\S+",
                state["search_results"],
            )

            if show_intermediate:

                st.text_area(
                    "Raw Search Results",
                    state["search_results"],
                    height=260,
                )

                st.markdown("### 🌐 URLs Found")

                if urls:

                    for url in urls:
                        st.link_button(url, url)

                else:

                    st.warning("No URLs found.")

            status.update(
                label="Step 1 completed ✅",
                state="complete",
            )

        wait()

        # --------------------------------------------------
        # READER AGENT
        # --------------------------------------------------

        with st.status(
            "Step 2 — Reading the best source...",
            expanded=show_intermediate,
        ) as status:

            reader_agent = build_reader_agent()

            reader_result = invoke_with_retry(
                reader_agent,
                {
                    "messages": [
                        (
                            "user",
                            f"""
Below are the search results.

{state['search_results']}

Choose the BEST URL.

Use scrape_url.

Return ONLY the scraped article.
"""
                        )
                    ]
                },
            )

            state["scraped_content"] = get_tool_output(
                reader_result["messages"],
                "scrape_url",
            )

            if show_intermediate:

                st.text_area(
                    "Scraped Content",
                    state["scraped_content"],
                    height=300,
                )

            status.update(
                label="Step 2 completed ✅",
                state="complete",
            )

        wait()
                # --------------------------------------------------
        # WRITER
        # --------------------------------------------------

        with st.status(
            "Step 3 — Generating research report...",
            expanded=True,
        ) as status:

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

            status.update(
                label="Step 3 completed ✅",
                state="complete",
            )

        # --------------------------------------------------
        # FINAL OUTPUT
        # --------------------------------------------------

        st.success(
            "Research pipeline completed successfully 🚀"
        )

        st.divider()

        st.subheader("📄 Research Report")

        st.markdown(state["report"])

        

        if show_intermediate:

            st.divider()

            with st.expander(
                "📊 Pipeline Summary",
                expanded=False,
            ):

                st.write("### URLs Used")

                urls = re.findall(
                    r"https?://\S+",
                    state["search_results"],
                )

                if urls:

                    for url in urls:
                        st.markdown(
                            f"- [{url}]({url})"
                        )

                else:

                    st.info(
                        "No URLs were extracted."
                    )

                

                
    except Exception:

        st.error("Pipeline failed!")

        st.code(
            traceback.format_exc(),
            language="python",
        )

