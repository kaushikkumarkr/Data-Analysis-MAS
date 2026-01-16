#!/usr/bin/env python3
"""DataVault CLI - Interactive REPL for data analysis.

Usage:
    python scripts/cli.py [OPTIONS]

Options:
    --backend TEXT    LLM backend: mlx, ollama, or auto (default: auto)
    --model TEXT      Model name (default: depends on backend)
    --help            Show this help message
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

from src.mcp.client import create_client
from src.agents.graph import DataVaultGraph
from src.agents.state import create_initial_state


console = Console()


def print_banner() -> None:
    """Print welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                     🏛️  DataVault CLI                         ║
║         Privacy-Preserving Data Analytics Agent               ║
╚══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold blue"))


def print_help() -> None:
    """Print help message."""
    help_text = """
## Commands

| Command | Description |
|---------|-------------|
| `load <path> [name]` | Load a CSV file into a table |
| `tables` | List all loaded tables |
| `describe <table>` | Show table schema |
| `sql <query>` | Execute raw SQL |
| `ask <question>` | Ask a natural language question |
| `stats <table> <column>` | Show column statistics |
| `clear` | Clear the screen |
| `help` | Show this help |
| `exit` / `quit` | Exit the CLI |

## Examples

```
load data/sample/sales_data.csv sales
tables
describe sales
ask How many transactions are in each region?
sql SELECT region, COUNT(*) FROM sales GROUP BY region
```
    """
    console.print(Markdown(help_text))


def print_table(data: list[dict], title: str = "") -> None:
    """Print data as a rich table."""
    if not data:
        console.print("[dim]No results[/dim]")
        return

    table = Table(title=title, show_header=True, header_style="bold cyan")

    # Add columns from first row
    for col in data[0].keys():
        table.add_column(str(col))

    # Add rows (limit to 50)
    for row in data[:50]:
        table.add_row(*[str(v) for v in row.values()])

    if len(data) > 50:
        console.print(f"[dim]Showing 50 of {len(data)} rows[/dim]")

    console.print(table)


def main() -> None:
    """Run the CLI."""
    parser = argparse.ArgumentParser(description="DataVault CLI")
    parser.add_argument("--backend", default="auto", help="LLM backend: mlx, ollama, auto")
    parser.add_argument("--model", default=None, help="Model name")
    args = parser.parse_args()

    print_banner()

    console.print("[dim]Initializing DataVault...[/dim]")

    try:
        with create_client() as client:
            console.print("✓ MCP client connected", style="green")

            # Try to initialize graph (may fail without LLM)
            graph = None
            try:
                graph = DataVaultGraph(client, backend=args.backend)
                console.print("✓ Agent graph initialized", style="green")
            except Exception as e:
                console.print(f"[yellow]⚠ Graph not available: {e}[/yellow]")
                console.print("[dim]You can still use: load, tables, describe, sql, stats[/dim]")

            console.print()
            print_help()
            console.print()

            while True:
                try:
                    user_input = Prompt.ask("[bold cyan]datavault[/bold cyan]")

                    if not user_input.strip():
                        continue

                    parts = user_input.strip().split(maxsplit=1)
                    cmd = parts[0].lower()
                    arg = parts[1] if len(parts) > 1 else ""

                    if cmd in ("exit", "quit", "q"):
                        console.print("[dim]Goodbye![/dim]")
                        break

                    elif cmd == "help":
                        print_help()

                    elif cmd == "clear":
                        console.clear()
                        print_banner()

                    elif cmd == "tables":
                        tables = client.list_tables()
                        if tables:
                            for t in tables:
                                console.print(f"  📊 {t}")
                        else:
                            console.print("[dim]No tables loaded[/dim]")

                    elif cmd == "load":
                        load_parts = arg.split()
                        if not load_parts:
                            console.print("[red]Usage: load <path> [table_name][/red]")
                            continue

                        path = load_parts[0]
                        name = load_parts[1] if len(load_parts) > 1 else Path(path).stem

                        result = client.load_dataset(path, name)
                        if result["success"]:
                            console.print(f"✓ Loaded {result['row_count']} rows into '{name}'", style="green")
                        else:
                            console.print(f"[red]Error: {result.get('error', 'Unknown')}[/red]")

                    elif cmd == "describe":
                        if not arg:
                            console.print("[red]Usage: describe <table_name>[/red]")
                            continue

                        schema = client.describe_table(arg)
                        if "columns" in schema:
                            table = Table(title=f"Schema: {arg}")
                            table.add_column("Column")
                            table.add_column("Type")
                            table.add_column("Nullable")

                            for col in schema["columns"]:
                                table.add_row(
                                    col["name"],
                                    col["type"],
                                    "✓" if col.get("nullable", True) else ""
                                )
                            console.print(table)
                        else:
                            console.print(f"[red]Error: {schema.get('error', 'Unknown')}[/red]")

                    elif cmd == "sql":
                        if not arg:
                            console.print("[red]Usage: sql <query>[/red]")
                            continue

                        result = client.execute_sql(arg)
                        if result["success"]:
                            print_table(result["rows"], f"Results ({result['row_count']} rows)")
                        else:
                            console.print(f"[red]SQL Error: {result.get('error', 'Unknown')}[/red]")

                    elif cmd == "stats":
                        stat_parts = arg.split()
                        if len(stat_parts) < 2:
                            console.print("[red]Usage: stats <table> <column>[/red]")
                            continue

                        table_name, column = stat_parts[0], stat_parts[1]
                        stats = client.get_statistics(table_name, column)

                        if "error" not in stats:
                            stat_table = Table(title=f"Statistics: {table_name}.{column}")
                            stat_table.add_column("Metric")
                            stat_table.add_column("Value")

                            for k, v in stats.items():
                                stat_table.add_row(k, str(round(v, 2) if isinstance(v, float) else v))
                            console.print(stat_table)
                        else:
                            console.print(f"[red]Error: {stats['error']}[/red]")

                    elif cmd == "ask":
                        if not graph:
                            console.print("[red]Agent not available. Use 'sql' for direct queries.[/red]")
                            continue

                        if not arg:
                            console.print("[red]Usage: ask <your question>[/red]")
                            continue

                        console.print("[dim]Thinking...[/dim]")

                        # Get data context
                        tables = client.list_tables()
                        schemas = {}
                        for t in tables:
                            schemas[t] = client.describe_table(t)

                        state = create_initial_state(
                            task=arg,
                            data_context={
                                "tables": tables,
                                "schemas": schemas,
                            }
                        )

                        try:
                            result = graph.run(state)

                            if result.get("final_answer"):
                                console.print(Panel(
                                    Markdown(result["final_answer"]),
                                    title="Answer",
                                    border_style="green"
                                ))

                            if result.get("sql_results"):
                                for sr in result["sql_results"]:
                                    if sr.get("success") and sr.get("rows"):
                                        print_table(sr["rows"][:20])

                            if result.get("errors"):
                                for err in result["errors"]:
                                    console.print(f"[yellow]Warning: {err}[/yellow]")

                        except Exception as e:
                            console.print(f"[red]Agent error: {e}[/red]")

                    else:
                        console.print(f"[red]Unknown command: {cmd}[/red]")
                        console.print("[dim]Type 'help' for available commands[/dim]")

                except KeyboardInterrupt:
                    console.print("\n[dim]Use 'exit' to quit[/dim]")
                    continue

    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
