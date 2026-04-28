"""Chainlit chat UI for the LightshowAI agentic workflow.

Wires the Claude Agent SDK to two local MCP stdio servers:
  - materials-project — Materials Project lookup tools
  - lightshowai     — OmniXAS XANES prediction tools

Inline-renders any HTML files written by the tools (XANES plots, 3D structure
viewers) by mounting ~/tmp/ at /plots and emitting an iframe custom element.

Run locally:
    chainlit run app.py -h --port 8000

Run on EC2 via systemd: see lightshowai-chatbot.service.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import chainlit as cl
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from dotenv import load_dotenv

load_dotenv()

# --- Paths ------------------------------------------------------------------
# app.py lives at <LightshowAI>/aws/chatbot/app.py — fully self-contained:
#   <LightshowAI>/lightshowai/   (the python package)
#   <LightshowAI>/mcp/           (vendored MCP server scripts)
#   <LightshowAI>/aws/chatbot/   (this app)
LIGHTSHOWAI_DIR = Path(__file__).resolve().parents[2]
MCP_DIR = LIGHTSHOWAI_DIR / "mcp"

# Plotly HTML files (XANES plots + crystal structures) land here. Served by a
# SEPARATE static-file server (see lightshowai-plots.service) on PLOT_PORT so
# the iframes embedded in chat messages can pull them inline.
#
# Why not mount in-process via Chainlit's FastAPI: Chainlit rebuilds its app
# during startup, which silently drops module-level mounts, and on_app_startup
# isn't available in every Chainlit version. The separate process is bulletproof.
PLOT_DIR = Path.home() / "tmp"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Public hostname/IP the BROWSER reaches the static server at. The chatbot
# itself never connects to this — only the user's browser does.
# Default to localhost for local dev; on EC2 set PLOTS_PUBLIC_URL in .env.
PLOTS_PUBLIC_URL = os.environ.get(
    "PLOTS_PUBLIC_URL", "http://localhost:8001"
).rstrip("/")

MP_API_KEY = os.environ.get("MP_API_KEY", "")
SHARED_PASSWORD = os.environ.get("CHAINLIT_PASSWORD", "")

# Model: prefer the AmSC i2 convention (ANTHROPIC_MODEL), then legacy CLAUDE_MODEL,
# then a sensible default for the i2 gateway.
MODEL = (
    os.environ.get("ANTHROPIC_MODEL")
    or os.environ.get("CLAUDE_MODEL")
    or "claude-sonnet-4-6"
)

# Auth: ANTHROPIC_AUTH_TOKEN (Bearer; for proxies like i2 LiteLLM) or
# ANTHROPIC_API_KEY (direct Anthropic). At least one must be set.
if not (os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")):
    sys.stderr.write(
        "ERROR: neither ANTHROPIC_AUTH_TOKEN nor ANTHROPIC_API_KEY is set.\n"
    )
    sys.exit(1)

# --- System prompt ----------------------------------------------------------
SYSTEM_PROMPT = """\
You are an XANES analysis assistant for the AmSC LightshowAI workflow.

You have access to two MCP toolsets:
  • materials-project — search Materials Project for crystal structures by formula,
    chemsys, element, oxidation state, etc.
  • lightshowai      — predict K-edge XANES spectra (OmniXAS / FEFF for Co, Cr, Cu,
    Fe, Mn, Ni, Ti, V; OmniXAS / VASP for Ti and Cu only).

Workflow guidance:
  1. When the user asks about a phase or oxidation state, search Materials Project
     first to get the canonical material_id and formula.
  2. Predict XANES with the lightshowai MCP. Prefer FEFF unless the user names VASP.
  3. When comparing multiple structures, save plots into the user's tmp dir and
     return the file path. Do not embed binary base64 in chat.
  4. If a metric is ambiguous (Brookite degeneracy, Mn intermediate phases, etc.),
     surface multiple ranking metrics — Pearson, Spearman, Cos(∂), ΔE50% — and
     explain when each is reliable. Do not pick a winner without evidence.
  5. Be concise. Show the call you made, the key numbers, and a one-paragraph
     interpretation. No long preambles.

File output convention (REQUIRED — the chat UI auto-renders files in ~/tmp/):
  • Always pass output_path explicitly when calling tools that save HTML:
      - lightshowai.plot_xanes        → output_path="~/tmp/<material_id>_xanes.html"
      - lightshowai.compare_xanes     → output_path="~/tmp/compare_<element>.html"
      - mp_visualize_structure        → output_path="~/tmp/<material_id>_structure.html"
  • Use ~/tmp/ for everything. Do not write to ~/Downloads/ — it doesn't exist.
  • Pass open_browser=False (the user views via the chat, not a local browser).

If the user asks for something outside XANES / Materials Project, say so plainly
and ask whether they want you to proceed anyway.
"""

# --- MCP server config ------------------------------------------------------
PYTHON = sys.executable

# AmSC MLflow tracking via the user's own MCP server (vendored from
# Documents/Research/AmSC/mlflow/intro-to-mlflow-pytorch/mcp_server/server.py).
# Auth env name on that server is AM_SC_API_KEY; tracking URI defaults to
# https://mlflow.american-science-cloud.org. Both can be overridden in .env.
AM_SC_API_KEY = os.environ.get("AM_SC_API_KEY", "")
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "https://mlflow.american-science-cloud.org"
)
MLFLOW_TRACKING_INSECURE_TLS = os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", "true")

MCP_SERVERS = {
    "materials-project": {
        "type": "stdio",
        "command": PYTHON,
        "args": [str(MCP_DIR / "materials_project_server.py")],
        "env": {"MP_API_KEY": MP_API_KEY},
    },
    "lightshowai": {
        "type": "stdio",
        "command": PYTHON,
        "args": [str(MCP_DIR / "lightshowai_server.py")],
        "env": {
            "MP_API_KEY": MP_API_KEY,
            "PYTHONPATH": str(LIGHTSHOWAI_DIR),
        },
    },
    "mlflow-amsc": {
        "type": "stdio",
        "command": PYTHON,
        "args": [str(MCP_DIR / "mlflow_server.py")],
        "env": {
            "AM_SC_API_KEY": AM_SC_API_KEY,
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
            "MLFLOW_TRACKING_INSECURE_TLS": MLFLOW_TRACKING_INSECURE_TLS,
        },
    },
}


# --- Auth -------------------------------------------------------------------
@cl.password_auth_callback
def auth(username: str, password: str) -> cl.User | None:
    if not SHARED_PASSWORD:
        return cl.User(identifier=username or "anonymous") if username else None
    if password == SHARED_PASSWORD:
        return cl.User(identifier=username or "user")
    return None


# --- Inline HTML rendering --------------------------------------------------
# Match absolute or ~-prefixed paths ending in .html (the tools return both forms).
_HTML_PATH_RE = re.compile(r"(?:/|~)[\w./~-]+\.html\b")


def _extract_html_files(text: str) -> list[Path]:
    """Pull every .html file path from a tool result that lives under PLOT_DIR."""
    out: list[Path] = []
    seen: set[str] = set()
    for raw in _HTML_PATH_RE.findall(text):
        p = Path(raw).expanduser().resolve()
        if not p.is_file():
            continue
        try:
            p.relative_to(PLOT_DIR.resolve())
        except ValueError:
            continue  # outside the served directory — skip for safety
        if str(p) in seen:
            continue
        seen.add(str(p))
        out.append(p)
    return out


async def _send_iframe(path: Path) -> None:
    """Render one HTML file inline.

    Uses raw <iframe> markup (works because unsafe_allow_html=true in
    .chainlit/config.toml) PLUS a cl.File attachment as a clickable
    fallback in case the browser blocks the iframe.

    The iframe src is an absolute URL pointing at the SEPARATE static-file
    server (lightshowai-plots.service on port 8001), not Chainlit. This
    sidesteps Chainlit's lifespan replacing in-process route mounts.
    """
    url = f"{PLOTS_PUBLIC_URL}/{path.name}"
    iframe_html = (
        f'<iframe src="{url}" '
        f'style="width:100%; height:520px; border:1px solid #243240; '
        f'border-radius:8px; background:#fff;" '
        f'sandbox="allow-scripts allow-same-origin allow-popups" '
        f'loading="lazy" title="{path.name}"></iframe>'
    )
    await cl.Message(
        content=f"**{path.name}**\n\n{iframe_html}",
        elements=[cl.File(name=path.name, path=str(path), display="side")],
    ).send()


# --- Session lifecycle ------------------------------------------------------
@cl.on_chat_start
async def on_chat_start() -> None:
    # Belt-and-braces: also try the mount here, in case on_app_startup didn't
    # fire (older Chainlit). Idempotent.
    _mount_plots_dir_once()

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        model=MODEL,
        mcp_servers=MCP_SERVERS,
        permission_mode="bypassPermissions",
        max_turns=30,
    )
    client = ClaudeSDKClient(options=options)
    await client.__aenter__()
    cl.user_session.set("client", client)
    cl.user_session.set("steps", {})  # tool_use_id -> cl.Step

    await cl.Message(
        content=(
            f"**LightshowAI XANES chatbot** — model `{MODEL}`\n\n"
            "Try: *Show the structure of mp-2657 and predict its Ti K-edge XANES "
            "with FEFF.*"
        )
    ).send()


@cl.on_chat_end
async def on_chat_end() -> None:
    client: ClaudeSDKClient | None = cl.user_session.get("client")
    if client is not None:
        await client.__aexit__(None, None, None)


# --- Message handler --------------------------------------------------------
@cl.on_message
async def on_message(message: cl.Message) -> None:
    client: ClaudeSDKClient = cl.user_session.get("client")
    steps: dict[str, cl.Step] = cl.user_session.get("steps")

    await client.query(message.content)

    reply = cl.Message(content="")
    await reply.send()

    async for msg in client.receive_response():
        if not isinstance(msg, AssistantMessage):
            continue
        for block in msg.content:
            if isinstance(block, TextBlock):
                await reply.stream_token(block.text)

            elif isinstance(block, ToolUseBlock):
                step = cl.Step(name=block.name, type="tool")
                step.input = block.input
                steps[block.id] = step
                await step.send()

            elif isinstance(block, ToolResultBlock):
                step = steps.pop(block.tool_use_id, None)
                if step is None:
                    continue
                step.output = block.content
                if getattr(block, "is_error", False):
                    step.is_error = True
                await step.update()

                # Inline-embed any HTML files the tool produced.
                for html in _extract_html_files(str(block.content)):
                    await _send_iframe(html)

    await reply.update()
