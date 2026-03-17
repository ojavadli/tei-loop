"""
Quickstart: TEI on a customer support agent.

Run:
    python3 examples/quickstart_agent.py

Requires: OPENAI_API_KEY in environment or .env file.
"""
import asyncio
import os
from openai import OpenAI

SYSTEM_PROMPT = """You are a customer support agent.
Answer questions about orders, shipping, and returns.
Be helpful and empathetic. Provide specific next steps."""


def customer_support(query: str) -> str:
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


async def main():
    from tei_loop import TEILoop

    loop = TEILoop(
        agent=customer_support,
        agent_file=__file__,
        verbose=True,
        interactive=False,
        num_iterations=10,
    )

    result = await loop.run(
        query="My order #4821 hasn't arrived. It's been 3 weeks and I'm frustrated.",
        test_queries=[
            "My order #4821 hasn't arrived. It's been 3 weeks and I'm frustrated.",
            "I want to return a damaged item. Order #2233.",
            "What is your refund policy for electronics?",
        ],
    )

    print(result.summary())


if __name__ == "__main__":
    asyncio.run(main())
