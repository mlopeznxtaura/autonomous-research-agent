"""
LangGraph multi-agent research orchestration graph.
Stages: ingest -> analyze -> synthesize -> experiment -> write -> review
SDKs: LangGraph, LangChain, Ollama, Anthropic
"""
import time
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END


@dataclass
class ResearchFinding:
    claim: str
    evidence: str
    source: str
    confidence: float
    novelty: float


class ResearchState(TypedDict):
    topic: str
    papers: List[Dict[str, Any]]
    findings: List[ResearchFinding]
    gaps: List[str]
    hypotheses: List[str]
    experiment_plan: str
    report_draft: str
    citations: List[str]
    review_feedback: str
    final_report: str
    iteration: int
    messages: List[Any]


RESEARCHER_PROMPT = """You are an expert AI researcher analyzing scientific literature.
Your task: extract key findings, identify research gaps, and synthesize insights.
Be rigorous, cite evidence, and flag areas of uncertainty.
Current research topic: {topic}"""

WRITER_PROMPT = """You are a scientific writer producing clear, rigorous research reports.
Structure your output with: Abstract, Introduction, Findings, Analysis, Gaps, Future Work.
Use precise language and maintain academic tone."""

REVIEWER_PROMPT = """You are a senior researcher reviewing a research synthesis report.
Identify: factual errors, missing citations, logical gaps, unsupported claims, areas needing deeper analysis.
Provide constructive, specific feedback."""


class AutonomousResearchAgent:
    """
    Multi-stage LangGraph research agent.
    Orchestrates: ingestion -> analysis -> synthesis -> writing -> review
    """

    def __init__(
        self,
        model: str = "mistral",
        ollama_url: str = "http://localhost:11434",
        knowledge_base=None,
    ):
        self.model = model
        self.kb = knowledge_base
        self.llm = ChatOllama(model=model, base_url=ollama_url, temperature=0.3)
        self._graph = self._build_graph()
        print(f"[ResearchAgent] Initialized | model={model}")

    def _build_graph(self):
        graph = StateGraph(ResearchState)
        graph.add_node("ingest", self._ingest_papers)
        graph.add_node("analyze", self._analyze_papers)
        graph.add_node("synthesize", self._synthesize_findings)
        graph.add_node("plan_experiments", self._plan_experiments)
        graph.add_node("write_report", self._write_report)
        graph.add_node("review", self._review_report)
        graph.add_node("revise", self._revise_report)

        graph.set_entry_point("ingest")
        graph.add_edge("ingest", "analyze")
        graph.add_edge("analyze", "synthesize")
        graph.add_edge("synthesize", "plan_experiments")
        graph.add_edge("plan_experiments", "write_report")
        graph.add_edge("write_report", "review")
        graph.add_conditional_edges(
            "review",
            lambda s: "revise" if s["iteration"] < 2 and "NEEDS REVISION" in s.get("review_feedback", "") else END,
            {"revise": "revise", END: END}
        )
        graph.add_edge("revise", "review")
        return graph.compile()

    def _ingest_papers(self, state: ResearchState) -> ResearchState:
        """Fetch and index relevant papers."""
        if self.kb:
            papers = self.kb.search(state["topic"], top_k=10)
            state["papers"] = [{"title": p.get("title", ""), "abstract": p.get("abstract", ""), "url": p.get("url", "")} for p in papers]
        else:
            # Stub: generate synthetic paper summaries
            state["papers"] = [
                {"title": f"Paper on {state['topic']} (Study {i+1})",
                 "abstract": f"This study investigates {state['topic']} using novel methods...",
                 "url": f"arxiv.org/abs/2024.{i:05d}"}
                for i in range(5)
            ]
        state["messages"].append(AIMessage(content=f"Ingested {len(state['papers'])} papers on '{state['topic']}'"))
        return state

    def _analyze_papers(self, state: ResearchState) -> ResearchState:
        """Extract key findings from papers."""
        papers_text = "

".join(
            f"Title: {p['title']}
Abstract: {p['abstract']}"
            for p in state["papers"][:5]
        )
        prompt = RESEARCHER_PROMPT.format(topic=state["topic"])
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Analyze these papers and extract 5 key findings:

{papers_text}")
        ]
        response = self.llm.invoke(messages)
        findings_text = response.content

        # Parse findings (stub: create structured findings)
        state["findings"] = [
            ResearchFinding(
                claim=f"Finding {i+1} about {state['topic']}",
                evidence=f"Supported by multiple papers",
                source=state["papers"][i % len(state["papers"])]["title"] if state["papers"] else "Unknown",
                confidence=0.8 - i * 0.1,
                novelty=0.6 + i * 0.05,
            )
            for i in range(min(5, len(state["papers"])))
        ]
        state["messages"].append(AIMessage(content=findings_text))
        return state

    def _synthesize_findings(self, state: ResearchState) -> ResearchState:
        """Identify gaps and formulate hypotheses."""
        findings_summary = "
".join(f"- {f.claim}" for f in state["findings"])
        response = self.llm.invoke([
            SystemMessage(content=RESEARCHER_PROMPT.format(topic=state["topic"])),
            HumanMessage(content=f"Given these findings:
{findings_summary}

Identify 3 research gaps and propose 2 testable hypotheses.")
        ])
        state["gaps"] = [f"Gap {i+1}: Underexplored aspect of {state['topic']}" for i in range(3)]
        state["hypotheses"] = [f"H{i+1}: {state['topic']} can be improved by X" for i in range(2)]
        state["messages"].append(AIMessage(content=response.content))
        return state

    def _plan_experiments(self, state: ResearchState) -> ResearchState:
        response = self.llm.invoke([
            HumanMessage(content=f"Design an experiment to test: {state['hypotheses'][0] if state['hypotheses'] else state['topic']}")
        ])
        state["experiment_plan"] = response.content
        return state

    def _write_report(self, state: ResearchState) -> ResearchState:
        """Generate the research report."""
        context = f"""
Topic: {state['topic']}
Key Findings: {[f.claim for f in state['findings']]}
Research Gaps: {state['gaps']}
Hypotheses: {state['hypotheses']}
"""
        response = self.llm.invoke([
            SystemMessage(content=WRITER_PROMPT),
            HumanMessage(content=f"Write a research report based on:
{context}")
        ])
        state["report_draft"] = response.content
        state["citations"] = [p["url"] for p in state["papers"][:5]]
        return state

    def _review_report(self, state: ResearchState) -> ResearchState:
        response = self.llm.invoke([
            SystemMessage(content=REVIEWER_PROMPT),
            HumanMessage(content=f"Review this report:

{state['report_draft'][:2000]}")
        ])
        state["review_feedback"] = response.content
        state["iteration"] = state.get("iteration", 0) + 1
        return state

    def _revise_report(self, state: ResearchState) -> ResearchState:
        response = self.llm.invoke([
            HumanMessage(content=f"Revise this report based on feedback:

Report:
{state['report_draft'][:1500]}

Feedback:
{state['review_feedback'][:500]}")
        ])
        state["report_draft"] = response.content
        return state

    def research(self, topic: str) -> Dict[str, Any]:
        """Run full research pipeline on a topic."""
        print(f"[ResearchAgent] Starting research: {topic}")
        initial_state = ResearchState(
            topic=topic, papers=[], findings=[], gaps=[], hypotheses=[],
            experiment_plan="", report_draft="", citations=[],
            review_feedback="", final_report="", iteration=0, messages=[],
        )
        final_state = self._graph.invoke(initial_state)
        print(f"[ResearchAgent] Done: {len(final_state['findings'])} findings, {len(final_state['gaps'])} gaps")
        return {
            "topic": topic,
            "n_papers": len(final_state["papers"]),
            "n_findings": len(final_state["findings"]),
            "gaps": final_state["gaps"],
            "hypotheses": final_state["hypotheses"],
            "report": final_state["report_draft"],
            "citations": final_state["citations"],
        }
