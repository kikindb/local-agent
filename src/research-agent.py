import argparse
import json
import os
import re
import unicodedata
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


SYSTEM_PROMPT = """You are a careful research assistant.
You must search the web for relevant, current information before answering.
Synthesize the findings clearly, distinguish facts from uncertainty, and cite
your sources with Markdown links using the URLs returned by the search tool.
If sources disagree, explain the disagreement rather than hiding it."""


RESEARCH_DIR = Path(__file__).resolve().parents[1] / ".research"
REPORT_NAME_PATTERN = re.compile(r"^(?P<number>\d+)-")


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
    """Run a research query, save the sourced report, and return the answer."""
    result = graph.invoke({"messages": [HumanMessage(content=query)]})
    answer = result["messages"][-1].content
    if not isinstance(answer, str):
        raise RuntimeError("The research model returned a non-text answer.")
    sources = _research_sources(result["messages"], answer)
    if not sources:
        raise RuntimeError(
            "Research completed without source URLs; no report was saved."
        )
    report_path = save_research_report(query, answer, sources)
    return f"{answer}\n\nSaved report: {report_path}"


def _research_sources(messages: list[Any], answer: str) -> list[tuple[str, str]]:
    """Collect source links from search tool results and cited answer links."""
    sources: list[tuple[str, str]] = []
    seen_urls: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str):
            try:
                visit(json.loads(value))
            except json.JSONDecodeError:
                return
        elif isinstance(value, Mapping):
            url = value.get("url")
            if isinstance(url, str) and url not in seen_urls:
                parsed_url = urlparse(url)
                if parsed_url.scheme in {"http", "https"} and parsed_url.netloc:
                    title = value.get("title")
                    if not isinstance(title, str) or not title.strip():
                        title = parsed_url.netloc
                    sources.append((title.strip(), url))
                    seen_urls.add(url)
            for nested_value in value.values():
                visit(nested_value)
        elif isinstance(value, (list, tuple)):
            for nested_value in value:
                visit(nested_value)

    for message in messages:
        if getattr(message, "type", None) == "tool":
            visit(getattr(message, "content", ""))

    for title, url in re.findall(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", answer):
        if url not in seen_urls:
            sources.append((title.strip(), url))
            seen_urls.add(url)

    return sources


def _slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug[:60].rstrip("-") or "research"


def save_research_report(
    query: str,
    answer: str,
    sources: list[tuple[str, str]],
) -> Path:
    """Write a dated Markdown report using the .research filename convention."""
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    slug = _slugify(query)
    highest_number = 0

    for path in RESEARCH_DIR.iterdir():
        match = REPORT_NAME_PATTERN.match(path.name)
        if match and path.is_file():
            highest_number = max(highest_number, int(match.group("number")))

    report = [
        f"# Research: {query}",
        "",
        f"**Date:** {today}",
        "",
        "## Findings",
        "",
        answer,
        "",
        "## Sources",
        "",
    ]
    for title, url in sources:
        safe_title = title.replace("]", r"\]")
        report.append(f"- [{safe_title}](<{url}>)")

    contents = "\n".join(report) + "\n"
    number = highest_number + 1
    while True:
        report_path = RESEARCH_DIR / f"{number:03d}-{slug}-{today}.md"
        try:
            with report_path.open("x", encoding="utf-8") as report_file:
                report_file.write(contents)
            return report_path
        except FileExistsError:
            number += 1


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