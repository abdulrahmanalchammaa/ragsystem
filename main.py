#!/usr/bin/env python3
"""
RAG System CLI — index documents and ask questions.

Usage:
  python main.py index [--dir data/documents]
  python main.py query "your question here"
  python main.py interactive
  python main.py status
  python main.py reset
"""

import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def get_pipeline():
    from src.rag_pipeline import RAGPipeline
    return RAGPipeline()


def cmd_index(args):
    pipeline = get_pipeline()
    count = pipeline.index(args.dir)
    print(f"\nDone. Total chunks indexed: {pipeline.document_count()}")


def cmd_query(args):
    pipeline = get_pipeline()
    print(f"\nQuestion: {args.question}\n")
    result = pipeline.query(args.question)
    print(f"Answer:\n{result['answer']}")
    if result["sources"]:
        print(f"\nSources: {', '.join(result['sources'])}")
    if args.verbose and result.get("retrieved_chunks"):
        print("\n--- Retrieved Chunks ---")
        for i, chunk in enumerate(result["retrieved_chunks"], 1):
            print(f"\n[{i}] {chunk['source']} (score: {chunk['score']})\n{chunk['text']}")


def cmd_interactive(args):
    pipeline = get_pipeline()
    print("\nRAG Interactive Mode — type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break
        result = pipeline.query(question)
        print(f"\nAssistant: {result['answer']}")
        if result["sources"]:
            print(f"Sources: {', '.join(result['sources'])}")
        print()


def cmd_status(args):
    pipeline = get_pipeline()
    count = pipeline.document_count()
    print(f"\nVector store contains {count} chunks.")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    key_status = "set" if api_key and not api_key.startswith("your_") else "NOT SET"
    print(f"ANTHROPIC_API_KEY: {key_status}")


def cmd_reset(args):
    confirm = input("This will delete all indexed documents. Type 'yes' to confirm: ").strip()
    if confirm.lower() == "yes":
        pipeline = get_pipeline()
        pipeline.reset_index()
        print("Index cleared.")
    else:
        print("Aborted.")


def main():
    parser = argparse.ArgumentParser(description="RAG System CLI")
    subparsers = parser.add_subparsers(dest="command")

    p_index = subparsers.add_parser("index", help="Index documents from a directory")
    p_index.add_argument("--dir", default="data/documents", help="Directory containing documents")

    p_query = subparsers.add_parser("query", help="Ask a single question")
    p_query.add_argument("question", help="The question to answer")
    p_query.add_argument("--verbose", "-v", action="store_true", help="Show retrieved chunks")

    subparsers.add_parser("interactive", help="Start interactive Q&A session")
    subparsers.add_parser("status", help="Show system status")
    subparsers.add_parser("reset", help="Clear the vector index")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "reset":
        cmd_reset(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
