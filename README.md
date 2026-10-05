# Local Research Agent

This LangGraph CLI agent researches questions with Tavily web search and
synthesizes the results using the project's local OpenAI-compatible model.

Install the project dependencies and add your Tavily API key to the root `.env`
file (it is excluded from Git):

```sh
uv sync
TAVILY_API_KEY=your-tavily-api-key
```

Start the local model server in a separate terminal with `./start-llm.sh 4B`,
then run a one-off query or start an interactive session:

```sh
uv run python src/research-agent.py "What changed in the latest Python release?"
uv run python src/research-agent.py
```

The model defaults to `mlx-community/Qwen3.5-4B-MLX-4bit` at
`http://127.0.0.1:8080/v1`. Override these defaults with `LLM_MODEL`,
`LLM_BASE_URL`, and `LLM_API_KEY`.