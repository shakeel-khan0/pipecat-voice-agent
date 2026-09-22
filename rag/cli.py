"""Interactive command-line test for standalone hybrid retrieval."""

import argparse

from rag.retrieve import HybridRetriever


def print_results(query: str, retriever: HybridRetriever):
    results, latency_ms = retriever.search(query)
    print(f"\nRetrieval latency: {latency_ms:.2f} ms")
    for result in results:
        print(f"\nRank {result.rank} | Score {result.score:.6f}")
        print(f"Heading: {result.heading}")
        print(result.text)


def main():
    parser = argparse.ArgumentParser(description="Test Agentix hybrid RAG retrieval.")
    parser.add_argument("query", nargs="?", help="Optional single query; omit for interactive mode.")
    args = parser.parse_args()

    retriever = HybridRetriever()
    if args.query:
        print_results(args.query, retriever)
        return

    print("Agentix hybrid retrieval. Enter a blank line to exit.")
    while True:
        query = input("\nQuery: ").strip()
        if not query:
            break
        print_results(query, retriever)


if __name__ == "__main__":
    main()

