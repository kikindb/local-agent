import argparse
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


SYSTEM_PROMPT = """You are a careful research assistant.
Search the web for relevant, current information before answering.
Synthesize the findings clearly, distinguish facts from uncertainty, and cite
your sources with Markdown links using the URLs returned by the search tool.
If sources disagree, explain the disagreement rather than hiding it."""


def build_research_graph() -> Any:
    """Build a LangGraph research workflow backed by web search."""
    if not os.environ.get("TAVILY_API_KEY"):
        raise RuntimeError(
            "TAVILY_API_KEY is required for web research. "
            "Set it in your environment before starting the agent."
        )

    search_tool = TavilySearch(max_results=5)
    model = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "mlx-community/Qwen3.5-4B-MLX-4bit"),
        base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
        api_key=os.getenv("LLM_API_KEY", "not-needed"),
        temperature=0,
    ).bind_tools([search_tool])

    def research(state: MessagesState) -> dict[str, list[Any]]:
        response = model.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("research", research)
    graph.add_node("tools", ToolNode([search_tool]))
    graph.add_edge(START, "research")
    graph.add_conditional_edges("research", tools_condition)
    graph.add_edge("tools", "research")
    return graph.compile()


def run_research(query: str, graph: Any) -> str:
    """Run a research query and return the final answer."""
    result = graph.invoke({"messages": [HumanMessage(content=query)]})
    answer = result["messages"][-1].content
    if not isinstance(answer, str):
        raise RuntimeError("The research model returned a non-text answer.")
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Research a topic using web search.")
    parser.add_argument(
        "query",
        nargs="*",
        help="Research question; omit to start an interactive session.",
    )
    args = parser.parse_args()
    graph = build_research_graph()

    if args.query:
        print(run_research(" ".join(args.query), graph))
        return

    print("Research Agent (type 'exit' or 'quit' to stop)")
    while True:
        try:
            query = input("\nResearch: ").strip()
        except EOFError:
            break
        if query.lower() in {"exit", "quit"}:
            break
        if query:
            print(f"\n{run_research(query, graph)}")


if __name__ == "__main__":
    main()