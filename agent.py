import asyncio
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Connect to LM Studio local API



llm = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

server_params = StdioServerParameters(
    command="python",           # or full path to venv python
    args=["main.py"]
)

SYSTEM = "You are an AI agent. If a tool can help, you must call it."

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()

            print("🧰 Tools loaded:", [t["name"] for t in tools])

            while True:
                user = input("\nYou: ")
                if user.lower() in ["exit", "quit"]:
                    break

                resp = llm.chat.completions.create(
                    model="local-model",  # LM Studio ignores name, but keep something
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user}
                    ],
                    tools=tools,
                    tool_choice="auto"
                )

                msg = resp.choices[0].message

                if msg.tool_calls:
                    for call in msg.tool_calls:
                        result = await session.call_tool(
                            call.function.name,
                            call.function.arguments
                        )
                        print(f"\n🔧 {call.function.name} → {result}")
                else:
                    print("\n🤖", msg.content)


#update added

if __name__ == "__main__":
    asyncio.run(main())
