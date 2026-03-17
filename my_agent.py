cd "/Users/orkhanjavadli/Documents/STANFORD/CS329T Trustworthy ML/mymcp2 copy 2/tei-loop"
clear
pip3 install tei-loop
"""
my_agent.py  -  A simple customer support agent
================================================
This is a typical LLM-powered agent with a basic system prompt.
It works, but we don't know HOW WELL it works.
That is what TEI will tell us.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.expanduser(
    "~/Documents/STANFORD/STRAMGT-356-01 Startup Garage/InterviewAgent/.env"
))

SYSTEM_PROMPT = """Answer customer questions about orders.
Be short. Just give the facts."""


def customer_support_agent(query: str) -> str:
    """Handle a customer support query and return a response."""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return response.choices[0].message.content or ""
