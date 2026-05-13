"""Terminal UI helpers: rendering panels and reading prompts."""
from __future__ import annotations

from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

BANNER = r"""
       _   _____ _      _    ____
      / \ |_   _| |    / \  / ___|
     / _ \  | | | |   / _ \ \___ \
    / ___ \ | | | |__/ ___ \ ___) |
   /_/   \_\|_| |_____/_/   \_\____/
"""

console = Console()

_PROMPT_STYLE = PTStyle.from_dict({
    "arrow": "ansibrightmagenta bold",
    "prompt": "ansicyan",
})


def render_banner(provider: str, session_id: str, checkpoint_db: Path, resuming: bool) -> None:
    """Print the startup banner, session info, and the first agent greeting."""
    subtitle = Text.assemble(
        ("Autonomous Travel Agent", "bold white"),
        ("  •  ", "dim"),
        (provider.upper(), "bright_cyan bold"),
    )
    console.print(Panel(
        Text.assemble(Text(BANNER, style="bright_magenta"), "\n", subtitle),
        border_style="bright_magenta",
        padding=(0, 2),
    ))

    info = Table.grid(padding=(0, 1))
    info.add_column(style="dim")
    info.add_column()
    info.add_row("Session", Text(session_id, style="bright_cyan bold"))
    info.add_row("State", Text(str(checkpoint_db), style="dim"))
    info.add_row("Exit", Text("type 'exit' or 'quit' (or press Ctrl-D)", style="dim"))
    console.print(info)

    if resuming:
        msg = f"Welcome back! Resuming session **{session_id}**. How can I help you continue?"
    else:
        msg = "Hello! I'm your travel assistant. Where are you starting from, and where would you like to go?"
    console.print(Panel(Markdown(msg), title="agent", border_style="cyan", title_align="left"))


def render_node(node: str) -> None:
    console.print(Rule(Text(node, style="bold magenta"), style="magenta"))


def render_agent_message(msg_type: str, content: str) -> None:
    body: Markdown | Text = Markdown(content) if content.strip() else Text("(empty)", style="dim italic")
    console.print(Panel(body, title=msg_type.lower(), border_style="cyan", title_align="left"))


def render_state(origin: str, destination: str, budget: str, trip_days: str) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bright_white")
    table.add_row("origin", str(origin))
    table.add_row("destination", str(destination))
    table.add_row("budget", str(budget))
    table.add_row("trip days", str(trip_days))
    console.print(Panel(table, title="state", border_style="green", title_align="left", padding=(0, 1)))


def render_error(err: Exception) -> None:
    console.print(Panel(
        Text.assemble(
            ("Connection failed. Check your internet connection and API key.\n\n", "bold red"),
            (str(err), "dim"),
        ),
        title="error",
        border_style="red",
        title_align="left",
    ))
    console.print(Text("Shutting down gracefully...", style="dim"))


def render_goodbye(newline: bool = False) -> None:
    console.print(Text(("\n" if newline else "") + "Goodbye!", style="bright_magenta"))


def thinking():
    """Context manager that shows a spinner while the agent works."""
    return console.status("[bright_magenta]thinking...[/]", spinner="dots")


def make_prompt_session() -> PromptSession:
    return PromptSession(history=InMemoryHistory())


def ask_user(session: PromptSession) -> str | None:
    """Prompt the user. Returns None on Ctrl-D / Ctrl-C."""
    console.print()
    try:
        return session.prompt(HTML("<arrow>❯</arrow> <prompt>you</prompt> "), style=_PROMPT_STYLE)
    except (EOFError, KeyboardInterrupt):
        return None
