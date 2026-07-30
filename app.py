import time
import re
import traceback
import streamlit as st

from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
)

st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Multi-Agent Research System")
st.caption("Search Agent → Reader Agent → Writer → Critic")


# Extract only LangChain message content
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
        f"Agent failed after {retries} retries: {last_error}"
    )


# ---------------- Sidebar ----------------

with st.sidebar:

    st.header("About")

    st.write(
        """
This app runs a 4-stage pipeline:

1. **Search Agent**
   - Uses Tavily search

2. **Reader Agent**
   - Selects best URL
   - Scrapes article

3. **Writer**
   - Creates research report

4. **Critic**
   - Reviews report
"""
    )

    st.divider()

    show_intermediate = st.checkbox(
        "Show intermediate steps",
        value=True
    )


# ---------------- Input ----------------

topic = st.text_input(
    "Enter a research topic",
    placeholder="e.g. System Design for freshers"
)


run_button = st.button(
    "Run Research Pipeline",
    type="primary"
)



if run_button:

    if not topic.strip():

        st.error("Please enter a topic first.")
        st.stop()


    state = {}


    try:


        # ---------------- Search Agent ----------------

        with st.status(
            "Step 1 — Searching web...",
            expanded=show_intermediate
        ) as status:


            search_agent = build_search_agent()


            search_result = invoke_with_retry(
                search_agent,
                {
                    "messages":[
                        (
                            "user",
                            f"""
Search the web for detailed information about '{topic}'.

Return raw search results with:

Title:
URL:
Snippet:
"""
                        )
                    ]
                }
            )


            state["search_results"] = extract_text(
                search_result["messages"][-1]
            )


            urls = re.findall(
                r"https?://[^\s]+",
                state["search_results"]
            )


            if show_intermediate:

                st.text_area(
                    "Raw Search Results",
                    state["search_results"],
                    height=250
                )

                st.write(
                    "URLs:",
                    urls
                )


            status.update(
                label="Step 1 completed ✅",
                state="complete"
            )



        # ---------------- Reader Agent ----------------


        with st.status(
            "Step 2 — Reading source...",
            expanded=show_intermediate
        ) as status:


            reader_agent = build_reader_agent()


            reader_result = invoke_with_retry(
                reader_agent,
                {
                    "messages":[
                        (
                            "user",
                            f"""
Below are search results:

{state['search_results']}

Extract the best URL.

Use scrape_url tool.

Return ONLY scraped article text.
"""
                        )
                    ]
                }
            )


            state["scraped_content"] = extract_text(
                reader_result["messages"][-1]
            )


            if show_intermediate:

                st.text_area(
                    "Scraped Content",
                    state["scraped_content"],
                    height=250
                )


            status.update(
                label="Step 2 completed ✅",
                state="complete"
            )



        # ---------------- Writer ----------------


        with st.status(
            "Step 3 — Writing report..."
        ) as status:


            research = f"""

SEARCH RESULTS:

{state['search_results']}


SCRAPED CONTENT:

{state['scraped_content']}

"""


            # IMPORTANT:
            # writer_chain already returns string

            state["report"] = writer_chain.invoke(
                {
                    "topic": topic,
                    "research": research
                }
            )


            status.update(
                label="Step 3 completed ✅",
                state="complete"
            )



        # ---------------- Critic ----------------


        with st.status(
            "Step 4 — Reviewing report..."
        ) as status:


            # IMPORTANT:
            # critic_chain already returns string

            state["feedback"] = critic_chain.invoke(
                {
                    "report": state["report"]
                }
            )


            status.update(
                label="Step 4 completed ✅",
                state="complete"
            )



        # ---------------- Output ----------------


        st.success(
            "Pipeline finished successfully 🚀"
        )


        tab1, tab2 = st.tabs(
            [
                "📄 Report",
                "🧪 Critic Feedback"
            ]
        )


        with tab1:

            st.markdown(
                state["report"]
            )


            st.download_button(
                "Download Report",
                data=state["report"],
                file_name=f"{topic.replace(' ','_')}_report.md",
                mime="text/markdown"
            )


        with tab2:

            st.markdown(
                state["feedback"]
            )


    except Exception:

        st.error(
            "Pipeline failed!"
        )

        st.code(
            traceback.format_exc()
        )

        