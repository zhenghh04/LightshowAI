"""Chainlit chat UI for the LightshowAI agentic workflow.

Wires the Claude Agent SDK to two local MCP stdio servers:
  - materials-project — Materials Project lookup tools
  - lightshowai     — OmniXAS XANES prediction tools

Run locally:
    chainlit run app.py -h --port 8000

Run on EC2 via systemd: see lightshowai-chatbot.service.
"""

from __future__ import annotations

import os
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

MP_API_KEY = os.environ.get("MP_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SHARED_PASSWORD = os.environ.get("CHAINLIT_PASSWORD", "")
MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-7")

if not ANTHROPIC_API_KEY:
    sys.stderr.write("ERROR: ANTHROPIC_API_KEY not set in environment.\n")
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

If the user asks for something outside XANES / Materials Project, say so plainly
and ask whether they want you to proceed anyway.
"""

# --- MCP server config ------------------------------------------------------
# Both servers run as stdio subprocesses. They inherit the venv's Python.
PYTHON = sys.executable

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
            # lightshowai_server.py adds examples/LightshowAI to sys.path; we
            # also need the package's site-packages, which the venv provides.
            "PYTHONPATH": str(LIGHTSHOWAI_DIR),
        },
    },
}


# --- Auth (shared password) -------------------------------------------------
@cl.password_auth_callback
def auth(username: str, password: str) -> cl.User | None:
    """One shared password for all collaborators. Set CHAINLIT_PASSWORD in .env."""
    if not SHARED_PASSWORD:
        # Auth disabled: any non-empty password lets you in. Fine for solo dev.
        return cl.User(identifier=username or "anonymous") if username else None
    if password == SHARED_PASSWORD:
        return cl.User(identifier=username or "user")
    return None


# --- Session lifecycle ------------------------------------------------------
@cl.on_chat_start
async def on_chat_start() -> None:
    """Spin up a per-session ClaudeSDKClient with both MCP servers attached."""
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        model=MODEL,
        mcp_servers=MCP_SERVERS,
        permission_mode="acceptEdits",  # auto-approve filesystem writes (plots, etc.)
        max_turns=30,
    )
    client = ClaudeSDKClient(options=options)
    await client.__aenter__()
    cl.user_session.set("client", client)
    cl.user_session.set("steps", {})  # tool_use_id -> cl.Step

    await cl.Message(
        content=(
            f"**LightshowAI XANES chatbot** — model `{MODEL}`\n\n"
            "Try: *Compare Ti K-edge XANES for the three TiO₂ polymorphs (rutile, "
            "anatase, brookite) using FEFF.*"
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

    # One streaming reply bubble for the assistant's text output.
    reply = cl.Message(content="")
    await reply.send()

    async for msg in client.receive_response():
        if not isinstance(msg, AssistantMessage):
            continue
        for block in msg.content:
            # Text → stream into the visible reply.
            if isinstance(block, TextBlock):
                await reply.stream_token(block.text)

            # Tool use → render as a collapsible Step with the input args.
            elif isinstance(block, ToolUseBlock):
                step = cl.Step(name=block.name, type="tool")
                step.input = block.input
                steps[block.id] = step
                await step.send()

            # Tool result → close the matching Step.
            elif isinstance(block, ToolResultBlock):
                step = steps.pop(block.tool_use_id, None)
                if step is None:
                    continue
                step.output = block.content
                if getattr(block, "is_error", False):
                    step.is_error = True
                await step.update()

    await reply.update()
