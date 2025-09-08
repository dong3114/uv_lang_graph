# Mcp를 쓰는 이유는 다양한 환경에서 동일하게 동작할 수 있도록 설정하려고 MCP 쓰는것.
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import asyncio
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o", streaming=True)

async def main():
    client = MultiServerMCPClient(
        {
            "test": {
                "command": "python",
                # "args": ["./mcp01_server.py"],
                "args": ["./mcp02_server.py"],
                "transport": "stdio",
            },
        }
    )
    tools = await client.get_tools()
    agent = create_react_agent(llm, tools=tools)
    # response = await agent.ainvoke({"messages": "what's (3 + 5) + 12?"})
    response = await agent.ainvoke({"messages": "연봉 5천만원 거주자의 소득세는 얼마인가요?"})
    
    final_message = response['messages'][-1]
    print(final_message)

if __name__ == "__main__":
    asyncio.run(main())