"""Terminal UI helpers: rendering panels and reading prompts."""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
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
    "state.corner": "ansibrightmagenta",
    "state.label": "ansibrightblack",
    "state.value": "ansiwhite bold",
    "state.sep": "ansibrightblack",
})

_CHOICE_STYLE = PTStyle.from_dict({
    "state.corner": "ansibrightmagenta",
    "state.label": "ansibrightblack",
    "state.value": "ansiwhite bold",
    "state.sep": "ansibrightblack",
    "choice.selected": "ansibrightmagenta bold",
    "choice.normal": "ansiwhite",
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


def render_node(node: str, elapsed_ms: float | None = None) -> None:
    if elapsed_ms is not None:
        label = Text.assemble(
            (node, "bold magenta"),
            ("  ", ""),
            (f"{elapsed_ms:,.0f} ms", "dim white"),
        )
    else:
        label = Text(node, style="bold magenta")
    console.print(Rule(label, style="magenta"))


def render_agent_message(msg_type: str, content: str) -> None:
    body: Markdown | Text = Markdown(content) if content.strip() else Text("(empty)", style="dim italic")
    console.print(Panel(body, title=msg_type.lower(), border_style="cyan", title_align="left"))


def render_node_status(msg: str) -> None:
    console.print(Text(f"  {msg.strip()}", style="dim"))


def _state_html(state: tuple[str, str, str, str, str]) -> HTML:
    def fmt(v: str) -> str:
        s = str(v)
        return "—" if s in {"", "None"} else s

    origin, destination, budget, trip_days, trip_start = state
    fields = [
        ("origin", fmt(origin)),
        ("destination", fmt(destination)),
        ("budget", fmt(budget)),
        ("trip days", fmt(trip_days)),
        ("est. start", fmt(trip_start)),
    ]
    parts = ["<state.corner>  ╰─ </state.corner>"]
    for i, (label, value) in enumerate(fields):
        if i:
            parts.append("<state.sep>  ·  </state.sep>")
        parts.append(
            f"<state.label>{label} </state.label>"
            f"<state.value>{value}</state.value>"
        )
    return HTML("".join(parts))


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


class ThinkingDisplay:
    """Live spinner + state line that types out updated values."""

    _SECONDS_PER_CHAR = 0.04
    _FIELD_COUNT = 5

    def __init__(self, state: tuple[str, str, str, str, str]) -> None:
        self._state = tuple(state)
        self._changed_at: list[float] = [float("-inf")] * self._FIELD_COUNT
        self._spinner = Spinner(
            "dots",
            text=Text("thinking...", style="bright_magenta"),
            style="bright_magenta",
        )

    def update(self, new_state: tuple[str, str, str, str, str]) -> None:
        new_state = tuple(new_state)
        now = time.monotonic()
        for i in range(self._FIELD_COUNT):
            if self._state[i] != new_state[i]:
                self._changed_at[i] = now
        self._state = new_state

    def __rich__(self) -> Group:
        return Group(self._spinner, self._render_state_line())

    def _render_state_line(self) -> Text:
        def fmt(v: str) -> str:
            s = str(v)
            return "—" if s in {"", "None"} else s

        now = time.monotonic()
        fields = [
            ("origin", fmt(self._state[0])),
            ("destination", fmt(self._state[1])),
            ("budget", fmt(self._state[2])),
            ("trip days", fmt(self._state[3])),
            ("est. start", fmt(self._state[4])),
        ]
        line = Text("  ╰─ ", style="bright_magenta")
        for i, (label, value) in enumerate(fields):
            if i:
                line.append("  ·  ", style="dim")
            line.append(f"{label} ", style="dim")
            line.append(self._typed(value, now - self._changed_at[i]), style="bright_white bold")
        return line

    def _typed(self, value: str, elapsed: float) -> str:
        if elapsed >= len(value) * self._SECONDS_PER_CHAR:
            return value
        if elapsed < 0:
            return " " * len(value)
        chars = int(elapsed / self._SECONDS_PER_CHAR)
        return value[:chars] + " " * (len(value) - chars)


@contextmanager
def thinking(state: tuple[str, str, str, str, str]):
    """Show a spinner and the live state line while the agent works."""
    display = ThinkingDisplay(state)
    with Live(display, console=console, refresh_per_second=20, transient=True):
        yield display


def make_prompt_session() -> PromptSession:
    return PromptSession(history=InMemoryHistory())


def ask_user(
    session: PromptSession,
    state: tuple[str, str, str, str, str] | None = None,
) -> str | None:
    """Prompt the user. Renders [input row, state row] directly inline.

    Returns None on Ctrl-D / Ctrl-C.
    """
    console.print()

    buffer = Buffer(multiline=False, history=session.history)

    kb = KeyBindings()

    @kb.add("enter")
    def _(event):
        event.app.exit(result=buffer.text)

    @kb.add("c-c")
    @kb.add("c-d")
    def _(event):
        event.app.exit(exception=KeyboardInterrupt())

    @kb.add("up")
    def _(event):
        buffer.history_backward()

    @kb.add("down")
    def _(event):
        buffer.history_forward()

    label = Window(
        FormattedTextControl(HTML("<arrow>❯</arrow> <prompt>you</prompt> ")),
        dont_extend_width=True,
        height=1,
    )
    input_win = Window(BufferControl(buffer=buffer), height=1, wrap_lines=False)
    state_win = Window(
        FormattedTextControl(_state_html(state) if state is not None else ""),
        height=1,
        always_hide_cursor=True,
    )

    app: Application = Application(
        layout=Layout(HSplit([VSplit([label, input_win]), state_win])),
        key_bindings=kb,
        style=_PROMPT_STYLE,
        full_screen=False,
    )

    try:
        return app.run()
    except (EOFError, KeyboardInterrupt):
        return None


def ask_choice(
    options: list[tuple[str, str]],
    state: tuple[str, str, str, str, str] | None = None,
) -> str | None:
    """Render an arrow-key selection widget.

    options — list of (value, display_label) pairs
    Returns the value of the chosen option, or None on Ctrl-C / Ctrl-D.
    """
    if not options:
        return None

    selected = [0]
    console.print()

    def get_text() -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        for i, (_value, label) in enumerate(options):
            if i == selected[0]:
                result.append(("class:choice.selected", f"  ❯  {label}"))
            else:
                result.append(("class:choice.normal", f"     {label}"))
            result.append(("", "\n"))
        return result

    kb = KeyBindings()

    @kb.add("up")
    def _(event):
        selected[0] = (selected[0] - 1) % len(options)

    @kb.add("down")
    def _(event):
        selected[0] = (selected[0] + 1) % len(options)

    @kb.add("enter")
    def _(event):
        event.app.exit(result=options[selected[0]][0])

    @kb.add("c-c")
    @kb.add("c-d")
    def _(event):
        event.app.exit(exception=KeyboardInterrupt())

    choice_win = Window(
        FormattedTextControl(get_text, focusable=True),
        height=len(options),
        always_hide_cursor=True,
    )
    state_win = Window(
        FormattedTextControl(_state_html(state) if state is not None else ""),
        height=1,
        always_hide_cursor=True,
    )

    app: Application = Application(
        layout=Layout(HSplit([choice_win, state_win])),
        key_bindings=kb,
        style=_CHOICE_STYLE,
        full_screen=False,
    )

    try:
        return app.run()
    except (EOFError, KeyboardInterrupt):
        return None
