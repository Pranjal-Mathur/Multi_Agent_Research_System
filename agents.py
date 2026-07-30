from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()




# llm = ChatMistralAI(
#     model="mistral-large-latest",
#     temperature=0
# )
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


# 1st agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="""
You are a research search agent.

Always use the web_search tool.

Your job is ONLY to return the raw search results.

For every result, preserve exactly:

Title:
URL:
Snippet:

DO NOT summarize.
DO NOT rewrite.
DO NOT remove URLs.
"""
    )

# 2nd agent
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="""
You are a research reader agent.

You will receive search results containing Titles, URLs and snippets.

Extract the BEST URL.

Call scrape_url on that URL.

Return ONLY the scraped text.

Do not answer from your own knowledge.
"""
    )

# writer chain
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. Write clear, structured and insightful reports."
    ),
    (
        "human",
        """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# critic chain
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a sharp and constructive research critic. Be honest and specific."
    ),
    (
        "human",
        """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()