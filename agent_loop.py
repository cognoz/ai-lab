"""
Phase 1a — A bare tool-calling loop. No framework.

Run: OPENAI_API_KEY=sk-... python agent_loop.py
"""
import json
import os
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from the environment

MODEL = "gpt-4o-mini"  # cheap + capable enough for Phase 1


def get_weather(city: str) -> dict:
    # Stubbed. In real life this would hit an API. The point is the loop,
    # not the data. Return something deterministic so you can see it flow.
    fake = {"malmo": 8, "stockholm": 5, "moscow": -4}
    return {"city": city, "temp_c": fake.get(city.lower(), 15)}


def add(a: float, b: float) -> dict:
    return {"result": a + b}


# The registry: name -> callable. The loop dispatches through this.
TOOLS = {"get_weather": get_weather, "add": add}


# ---------------------------------------------------------------------------
# 2. The schemas — how you DESCRIBE those tools to the model.
#    This JSON-schema shape is the contract. The model reads these
#    descriptions to decide when and how to call each tool.
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current temperature for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# 3. The loop. THIS is the agent.
# ---------------------------------------------------------------------------
def run(user_prompt: str, max_turns: int = 10) -> str:
    messages = [{"role": "user", "content": user_prompt}]

    for turn in range(max_turns):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
        msg = resp.choices[0].message

        # The model either (a) asks to call one or more tools, or
        # (b) returns a final text answer. That's the whole decision.
        if not msg.tool_calls:
            return msg.content  # (b) done

        # (a) Append the model's tool-call request to the transcript...
        messages.append(msg)

        # ...then execute each requested call and feed results back.
        for call in msg.tool_calls:
            fn = TOOLS[call.function.name]
            args = json.loads(call.function.arguments)
            result = fn(**args)
            print(f"  [tool] {call.function.name}({args}) -> {result}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                }
            )
        # loop again — model now sees the tool results and continues

    return "(stopped: hit max_turns)"


if __name__ == "__main__":
    for prompt in [
        "What's the weather in Malmo and Moscow, and what's the difference in temperature?",
        "What is 47 plus 55?",
    ]:
        print(f"\nUSER: {prompt}")
        print(f"ASSISTANT: {run(prompt)}")
