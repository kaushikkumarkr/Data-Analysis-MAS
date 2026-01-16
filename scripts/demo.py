#!/usr/bin/env python3
"""DataVault Demo Script.

This script demonstrates the multi-agent workflow with MLX LLM
on Mac M1/M2/M3. It loads sample sales data and runs analysis queries.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

console = Console()


def print_header():
    """Print demo header."""
    console.print(Panel.fit(
        "[bold cyan]DataVault Demo[/bold cyan]\n"
        "[dim]Privacy-preserving multi-agent data analytics[/dim]",
        border_style="cyan"
    ))
    console.print()


def demo_mcp_tools():
    """Demo the MCP tools directly."""
    console.print("[bold yellow]━━━ Demo 1: MCP Tools ━━━[/bold yellow]")
    console.print()
    
    from src.mcp.client import create_client
    
    with create_client() as client:
        # Load sample data
        csv_path = Path(__file__).parent.parent / "data" / "sample" / "sales_data.csv"
        console.print(f"📂 Loading data from: [cyan]{csv_path.name}[/cyan]")
        
        result = client.load_dataset(str(csv_path), "sales_data")
        if result.success:
            console.print(f"✅ Loaded [green]{result.data['rows_loaded']}[/green] rows into [cyan]sales_data[/cyan]")
        else:
            console.print(f"❌ Error: {result.error}")
            return
        
        console.print()
        
        # List tables
        console.print("[bold]Available Tables:[/bold]")
        result = client.list_tables()
        for table in result.data["tables"]:
            console.print(f"  • {table}")
        
        console.print()
        
        # Describe table
        console.print("[bold]Table Schema:[/bold]")
        result = client.describe_table("sales_data")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Column")
        table.add_column("Type")
        for col in result.data["columns"]:
            table.add_row(col["name"], col["type"])
        console.print(table)
        
        console.print()
        
        # Run a sample query
        console.print("[bold]Sample Query - Revenue by Region:[/bold]")
        result = client.execute_sql("""
            SELECT 
                region,
                COUNT(*) as transactions,
                ROUND(SUM(total_amount), 2) as total_revenue,
                ROUND(AVG(total_amount), 2) as avg_transaction
            FROM sales_data
            GROUP BY region
            ORDER BY total_revenue DESC
        """)
        
        if result.success:
            table = Table(show_header=True, header_style="bold green")
            table.add_column("Region")
            table.add_column("Transactions", justify="right")
            table.add_column("Total Revenue", justify="right")
            table.add_column("Avg Transaction", justify="right")
            
            for row in result.data["rows"]:
                table.add_row(
                    row["region"],
                    str(row["transactions"]),
                    f"${row['total_revenue']:,.2f}",
                    f"${row['avg_transaction']:,.2f}"
                )
            console.print(table)
        
        console.print()
        
        # Get statistics
        console.print("[bold]Column Statistics - total_amount:[/bold]")
        result = client.get_statistics("sales_data", "total_amount")
        if result.success:
            stats = result.data
            console.print(f"  • Count: {stats['count']}")
            console.print(f"  • Min: ${stats['min']:,.2f}")
            console.print(f"  • Max: ${stats['max']:,.2f}")
            console.print(f"  • Mean: ${stats['mean']:,.2f}")
            console.print(f"  • Std Dev: ${stats['std']:,.2f}")
    
    console.print()
    console.print("[green]✓ MCP Tools demo complete![/green]")
    console.print()


def demo_llm_factory():
    """Demo the LLM factory and available backends."""
    console.print("[bold yellow]━━━ Demo 2: LLM Backend ━━━[/bold yellow]")
    console.print()
    
    from src.utils.llm_factory import get_available_backends, create_chat_model
    from src.utils.config import get_config
    
    backends = get_available_backends()
    console.print(f"[bold]Available LLM Backends:[/bold] {', '.join(backends)}")
    
    config = get_config()
    if config.mlx.enabled:
        console.print(f"[green]✓ MLX is enabled[/green]")
        console.print(f"  Model: [cyan]{config.mlx.model}[/cyan]")
    else:
        console.print(f"[yellow]○ MLX is disabled, using Ollama[/yellow]")
        console.print(f"  Model: [cyan]{config.ollama.model}[/cyan]")
    
    console.print()
    
    # Create LLM and test
    console.print("[bold]Testing LLM with a simple prompt...[/bold]")
    console.print("[dim](First run will download the model - ~2GB)[/dim]")
    console.print()
    
    try:
        llm = create_chat_model(temperature=0.1)
        response = llm.invoke("Respond with just 'Hello from MLX!' and nothing else.")
        console.print(f"[green]LLM Response:[/green] {response.content}")
    except Exception as e:
        console.print(f"[red]LLM Error:[/red] {e}")
        console.print("[dim]This is expected if MLX model isn't downloaded yet.[/dim]")
    
    console.print()
    console.print("[green]✓ LLM Backend demo complete![/green]")
    console.print()


def demo_agent_workflow():
    """Demo the full agent workflow."""
    console.print("[bold yellow]━━━ Demo 3: Multi-Agent Workflow ━━━[/bold yellow]")
    console.print()
    
    from src.agents.graph import create_datavault_graph
    from pathlib import Path
    
    # Create graph
    console.print("[bold]Initializing DataVault Graph...[/bold]")
    graph = create_datavault_graph()
    
    # Load sample data
    csv_path = Path(__file__).parent.parent / "data" / "sample" / "sales_data.csv"
    graph.mcp_client.load_dataset(str(csv_path), "sales_data")
    console.print(f"✅ Loaded sales_data")
    console.print()
    
    # Run an analysis task
    console.print("[bold]Running Analysis Task:[/bold]")
    console.print("[cyan]'What are the total sales by region?'[/cyan]")
    console.print()
    console.print("[dim]This will route to the Analyst Agent...[/dim]")
    console.print()
    
    try:
        result = graph.run("What are the total sales by region?")
        
        # Display results
        console.print(f"[bold]Task Type:[/bold] {result.get('task_type', 'unknown')}")
        console.print()
        
        sql_results = result.get("sql_results", [])
        if sql_results:
            console.print(f"[bold]SQL Queries Executed:[/bold] {len(sql_results)}")
            for i, sql_result in enumerate(sql_results):
                if sql_result.get("success"):
                    console.print(f"  Query {i+1}: ✅ {sql_result.get('row_count', 0)} rows")
                    
                    # Display results
                    rows = sql_result.get("rows", [])
                    if rows:
                        console.print()
                        table = Table(show_header=True, header_style="bold blue")
                        for col in rows[0].keys():
                            table.add_column(str(col))
                        for row in rows[:10]:
                            table.add_row(*[str(v) for v in row.values()])
                        console.print(table)
                else:
                    console.print(f"  Query {i+1}: ❌ {sql_result.get('error', 'Unknown error')}")
        
        errors = result.get("errors", [])
        if errors:
            console.print()
            console.print("[bold red]Errors:[/bold red]")
            for error in errors:
                console.print(f"  • {error}")
    
    except Exception as e:
        console.print(f"[red]Workflow Error:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())
    
    finally:
        graph.mcp_client.close()
    
    console.print()
    console.print("[green]✓ Agent workflow demo complete![/green]")


def main():
    """Run all demos."""
    print_header()
    
    # Demo 1: MCP Tools (no LLM needed)
    demo_mcp_tools()
    
    console.print("─" * 50)
    console.print()
    
    # Ask if user wants to continue with LLM demos
    console.print("[bold]Continue with LLM demos?[/bold]")
    console.print("[dim]This will download the MLX model (~2GB) if not cached.[/dim]")
    console.print()
    
    try:
        response = input("Continue? [y/N]: ").strip().lower()
        if response != 'y':
            console.print()
            console.print("[yellow]Skipping LLM demos.[/yellow]")
            return
    except (KeyboardInterrupt, EOFError):
        console.print()
        console.print("[yellow]Skipping LLM demos.[/yellow]")
        return
    
    console.print()
    
    # Demo 2: LLM Factory
    demo_llm_factory()
    
    console.print("─" * 50)
    console.print()
    
    # Demo 3: Full agent workflow
    demo_agent_workflow()
    
    console.print()
    console.print(Panel.fit(
        "[bold green]Demo Complete![/bold green]\n"
        "DataVault is ready for use.",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
