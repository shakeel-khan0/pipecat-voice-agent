"""Small local-Qdrant evaluation set for the voice agent's RAG path."""

from dataclasses import dataclass
from statistics import mean

from rag.integration import needs_rag, retrieval_query, select_results
from rag.retrieve import HybridRetriever


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    query: str
    should_retrieve: bool
    expected_heading: str | None = None


CASES = (
    EvaluationCase("company overview", "What does Agentix Labs AI do?", True, "What does Agentix"),
    EvaluationCase("service capability", "What solutions can your team provide?", True, "Service Hierarchy"),
    EvaluationCase(
        "semantic paraphrase",
        "Could you create something that answers calls after hours and updates our CRM?",
        True,
        "What the service is",
    ),
    EvaluationCase("exact project", "What is PongVerse?", True, "PongVerse"),
    EvaluationCase("exact technology", "Do you use DeepSORT?", True, "Computer Vision"),
    EvaluationCase("pricing", "How much would an AI voice agent cost?", True, "project bands"),
    EvaluationCase("unknown fact", "Do you have offices in Dubai?", True, "Knowledge Boundaries"),
    EvaluationCase("greeting", "Hello", False),
    EvaluationCase("booking only", "I want to book a meeting tomorrow at 5 PM", False),
    EvaluationCase(
        "company question during booking",
        "While we book, what services do you offer?",
        True,
        "Company Description",
    ),
)


def main() -> None:
    retriever = HybridRetriever()
    latencies = []
    failures = []
    for case in CASES:
        used = needs_rag(case.query)
        selected = []
        latency_ms = 0.0
        if used:
            results, latency_ms = retriever.search(retrieval_query(case.query))
            selected = select_results(case.query, results)
            latencies.append(latency_ms)
        heading = selected[0].heading if selected else "-"
        passed = used == case.should_retrieve and (
            not used
            or (bool(selected) and case.expected_heading.lower() in heading.lower())
        )
        if not passed:
            failures.append(case.name)
        print(f"{'PASS' if passed else 'FAIL'} | {case.name} | {latency_ms:.2f} ms | {heading}")

    print(f"average retrieval latency: {mean(latencies):.2f} ms")
    print(f"worst retrieval latency: {max(latencies):.2f} ms")
    if failures:
        raise SystemExit(f"Evaluation failures: {', '.join(failures)}")


if __name__ == "__main__":
    main()
