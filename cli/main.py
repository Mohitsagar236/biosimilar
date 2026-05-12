#!/usr/bin/env python
"""Interactive CLI for querying the RAG agent."""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

import config
from src.agent.rag_agent import RAGAgent
from src.embeddings.embedding_generator import get_embeddings
from src.utils.helpers import setup_logging
from src.vectorstore.vector_db import VectorDatabase

console = Console()


def run_cli():
    setup_logging("WARNING")

    console.print(Panel.fit(
        "[bold cyan]Agentic RAG Chatbot[/bold cyan]\n"
        "[dim]Type your question, 'clear' to reset memory, or 'quit' to exit.[/dim]",
        border_style="cyan",
    ))

    embeddings = get_embeddings()
    db = VectorDatabase(embeddings)
    chunk_count = db.count()

    if chunk_count == 0:
        console.print(
            "[yellow]Warning:[/yellow] No documents found in vector store. "
            "Run [bold]python ingest.py[/bold] first."
        )
    else:
        sources = db.list_sources()
        console.print(f"[green]Loaded {chunk_count} chunks from {len(sources)} document(s).[/green]")

    agent = RAGAgent(db)

    while True:
        try:
            console.print()
            question = Prompt.ask("[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not question.strip():
            continue

        if question.strip().lower() in {"quit", "exit", "q"}:
            console.print("[dim]Goodbye![/dim]")
            break

        if question.strip().lower() == "clear":
            agent.clear_memory()
            console.print("[dim]Conversation memory cleared.[/dim]")
            continue

        if question.strip().lower() == "sources":
            sources = db.list_sources()
            console.print("\n".join(f"  • {Path(s).name}" for s in sources) or "No sources ingested.")
            continue

        with console.status("[bold blue]Thinking…[/bold blue]"):
            result = agent.chat(question)

        console.print(Rule(style="dim"))
        console.print("[bold blue]Assistant:[/bold blue]")
        console.print(Markdown(result["answer"]))

        if result["sources"]:
            console.print("\n[dim]Sources:[/dim]")
            for src in result["sources"]:
                console.print(f"  [dim]• {Path(src).name}[/dim]")


if __name__ == "__main__":
    run_cli()
