from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=350,
)

# ---------------- Search Agent ----------------
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="""
You are a search agent.

IMPORTANT RULES:

1. You MUST call the web_search tool exactly ONCE.
2. Never answer from your own knowledge.
3. After the tool returns, COPY THE TOOL OUTPUT VERBATIM.
4. Do NOT summarize.
5. Do NOT remove URLs.
6. Do NOT reformat.
7. Do NOT explain anything.
8. Your final response must be ONLY the tool output.

If the tool returns five results, output all five exactly as received.
""",
    )


# ---------------- Reader Agent ----------------
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="""
You are a reader agent.

Rules:

1. Read the search results.
2. Pick the single most relevant URL.
3. Call scrape_url exactly once.
4. Return ONLY the scraped text.
5. Never answer from your own knowledge.
6. Do not summarize.
""",
    )


# ---------------- Writer ----------------

writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert report writer.

Write only from the supplied research.

Do not invent facts.

Maximum 150 words.
""",
        ),
        (
            "human",
            """
Topic:
{topic}

Research:
{research}

Write:

Summary:
Key Findings:
Sources:
""",
        ),
    ]
)

writer_chain = writer_prompt | llm | StrOutputParser()

