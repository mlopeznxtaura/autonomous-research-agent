"""
autonomous-research-agent — Entry Point (Cluster 25 — Final Cluster)

Multi-agent autonomous research system.

Usage:
  python main.py --mode research --topic "transformer attention mechanisms"
  python main.py --mode review --topic "protein folding"
  python main.py --mode ui
  python main.py --mode pipeline --topic "quantum computing applications"
"""
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Autonomous Research Agent")
    p.add_argument("--mode", required=True, choices=["research", "review", "ui", "pipeline", "demo"])
    p.add_argument("--topic", default="large language model reasoning capabilities")
    p.add_argument("--paper", default=None, help="Path to PDF for review mode")
    p.add_argument("--model", default="mistral", help="Ollama model to use")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--output", default="./research_output")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def mode_research(args):
    from knowledge.paper_store import ResearchPaperStore
    from agents.research_graph import AutonomousResearchAgent
    import json
    from pathlib import Path

    print(f"[Research] Topic: {args.topic}")
    kb = ResearchPaperStore(persist_dir=f"{args.output}/kb")

    # Fetch papers
    print("[Research] Fetching papers from arXiv...")
    papers = kb.fetch_arxiv(args.topic, max_results=10)
    print(f"[Research] {len(papers)} papers indexed")

    # Run agent
    agent = AutonomousResearchAgent(model=args.model, ollama_url=args.ollama_url, knowledge_base=kb)
    result = agent.research(args.topic)

    # Save report
    Path(args.output).mkdir(parents=True, exist_ok=True)
    report_path = f"{args.output}/report.md"
    with open(report_path, "w") as f:
        f.write(f"# Research Report: {args.topic}\n\n")
        f.write(result["report"])
        f.write("\n\n## Citations\n")
        f.write("\n".join(f"- {c}" for c in result["citations"]))

    print(f"\nResearch complete!")
    print(f"  Papers: {result['n_papers']}")
    print(f"  Findings: {result['n_findings']}")
    print(f"  Gaps: {len(result['gaps'])}")
    print(f"  Report: {report_path}")


def mode_ui(args):
    import gradio as gr
    from knowledge.paper_store import ResearchPaperStore
    from agents.research_graph import AutonomousResearchAgent

    kb = ResearchPaperStore()
    agent = AutonomousResearchAgent(model=args.model, ollama_url=args.ollama_url, knowledge_base=kb)

    def run_research(topic, max_papers):
        if not topic.strip():
            return "Please enter a research topic.", ""
        papers = kb.fetch_arxiv(topic, max_results=int(max_papers))
        result = agent.research(topic)
        citations = "\n".join(f"- {c}" for c in result["citations"][:5])
        return result["report"], citations

    with gr.Blocks(title="Autonomous Research Agent") as demo:
        gr.Markdown("# Autonomous Research Agent\nPowered by LangGraph + LlamaIndex + Ollama")
        with gr.Row():
            topic = gr.Textbox(label="Research Topic", placeholder="e.g. transformer attention mechanisms")
            max_papers = gr.Slider(1, 20, value=5, step=1, label="Max Papers")
        btn = gr.Button("Run Research", variant="primary")
        report = gr.Markdown(label="Research Report")
        citations_box = gr.Textbox(label="Citations")
        btn.click(run_research, inputs=[topic, max_papers], outputs=[report, citations_box])

    demo.launch(server_port=args.port, share=False)


def mode_demo(args):
    print("Running research agent demo...\n")
    print(f"Topic: {args.topic}")
    from knowledge.paper_store import ResearchPaperStore
    from agents.research_graph import AutonomousResearchAgent

    kb = ResearchPaperStore(persist_dir=f"{args.output}/kb")
    papers = kb.fetch_arxiv(args.topic, max_results=5)
    print(f"Fetched {len(papers)} papers")
    for p in papers[:3]:
        print(f"  - {p['title'][:60]}...")

    agent = AutonomousResearchAgent(model=args.model, ollama_url=args.ollama_url, knowledge_base=kb)
    result = agent.research(args.topic)
    print(f"\nReport preview (first 500 chars):")
    print(result["report"][:500])
    print(f"\nGaps found: {len(result['gaps'])}")
    print("Demo complete. Run --mode ui for interactive interface.")


def main():
    args = parse_args()
    print("=" * 60)
    print("  Autonomous Research Agent  (Cluster 25 — Final)")
    print(f"  Mode: {args.mode.upper()} | Model: {args.model}")
    print("=" * 60)

    dispatch = {
        "research": mode_research,
        "review": mode_research,
        "ui": mode_ui,
        "pipeline": mode_research,
        "demo": mode_demo,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
