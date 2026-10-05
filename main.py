from openai import OpenAI
import json
from datetime import datetime, timezone


client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",
)

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"


# -------------------------
# Tools
# -------------------------

def get_current_time():
    """Return the current local time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_current_time_in_utc():
    """Return the current UTC time."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time_in_utc",
            "description": "Get the current UTC date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            }
        }
    }
]


# -------------------------
# Agent
# -------------------------

messages = [
    {
        "role": "system",
        "content": """
You are a snarky and witty AI agent, You love to use emojis.

You have access to tools. Use a tool when it is necessary
to answer the user's question.

After receiving a tool result, use it to answer the user.
""",
    }
]


def run_agent(user_input):
    messages.append({
        "role": "user",
        "content": user_input,
    })

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            temperature=0.2,
        )

        message = response.choices[0].message

        # Add the assistant response to conversation
        messages.append(message)

        # No tool call -> we're finished
        if not message.tool_calls:
            return message.content

        # Execute requested tools
        for tool_call in message.tool_calls:

            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments or "{}")

            print(f"\n🔧 Agent called tool: {function_name}")
            print(f"   Arguments: {arguments}")

            if function_name == "get_current_time":
                result = get_current_time()

            elif function_name == "get_current_time_in_utc":
                result = get_current_time_in_utc()

            else:
                result = f"Unknown tool: {function_name}"

            print(f"   Result: {result}")

            # Give result back to the model
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })


# -------------------------
# Chat loop
# -------------------------

print("🤖 Local Agent")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in {"exit", "quit"}:
        break

    answer = run_agent(user_input)

    print(f"\nAgent: {answer}\n")