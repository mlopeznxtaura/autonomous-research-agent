# Autonomous Research Agent

Cluster 25 of the NextAura 500 SDKs / 25 Clusters project. The final cluster.

Multi-agent system that reads papers, reasons across them, designs experiments, writes code, and produces research reports — autonomously.

## Architecture

- LangGraph for the multi-agent orchestration graph
- LlamaIndex for scientific paper ingestion and RAG
- DSPy for optimized reasoning chain compilation
- AutoGen for multi-agent collaboration
- CrewAI for role-based agent teams
- Pydantic AI for structured output extraction
- Instructor for reliable LLM output parsing
- Chroma + Qdrant for the research knowledge base
- W&B + MLflow for experiment tracking
- Prefect for pipeline orchestration
- LangSmith for agent trace observability
- Gradio for interactive research UI

## SDKs Used

LangGraph, LlamaIndex, DSPy, AutoGen, CrewAI, Pydantic AI, Instructor, Anthropic SDK, OpenAI SDK, Ollama, Chroma, Qdrant, W&B, MLflow, Prefect, LangSmith, Gradio, FastAPI, Redis, Prometheus Client

## Quickstart

```bash
pip install -r requirements.txt
ollama pull mistral  # or any local model

# Run the research agent on a topic
python main.py --mode research --topic "transformer attention mechanisms"

# Run multi-agent paper review
python main.py --mode review --paper ./paper.pdf

# Launch interactive Gradio UI
python main.py --mode ui

# Run full autonomous research pipeline
python main.py --mode pipeline --topic "protein folding ML methods"
```

## Workflow

1. PaperIngestor: fetch + chunk + embed papers from arXiv/Semantic Scholar
2. ReasoningAgent: LangGraph graph to analyze, compare, and synthesize findings
3. ExperimentDesigner: propose and implement experiments based on gaps found
4. WriterAgent: produce structured research report with citations
5. ReviewCrew: CrewAI team to peer-review and refine the report
