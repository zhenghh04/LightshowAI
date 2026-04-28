"""Chainlit chat UI for the LightshowAI agentic workflow.

Wires the Claude Agent SDK to two local MCP stdio servers:
  - materials-project — Materials Project lookup tools
  - lightshowai     — OmniXAS XANES prediction tools

Inline-renders any HTML files written by the tools (XANES plots, 3D structure
viewers) by serving ~/tmp/ and emitting a Chainlit iframe custom element.

Run locally:
    chainlit run app.py -h --port 8000

Run on EC2 via systemd: see lightshowai-chatbot.service.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import chainlit as cl
import mlflow
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
UPLOAD_DIR = PLOT_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_MISSING = object()

# Public hostname/IP the BROWSER reaches the static server at. The chatbot
# itself never connects to this — only the user's browser does.
# Default to localhost for local dev; on EC2 set PLOTS_PUBLIC_URL in .env.
PLOTS_PUBLIC_URL = os.environ.get(
    "PLOTS_PUBLIC_URL", "http://localhost:8001"
).rstrip("/")

MP_API_KEY = os.environ.get("MP_API_KEY", "")
SHARED_PASSWORD = os.environ.get("CHAINLIT_PASSWORD", "")
CHAT_TURN_TIMEOUT_S = int(os.environ.get("CHAINLIT_CHAT_TURN_TIMEOUT_S", "900"))

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

# --- MLflow auto-logging ----------------------------------------------------
# Each chat turn = one MLflow run. Logs:
#   params : model, prompt (truncated), user, tool list
#   metrics: latency, tool_calls, response_chars
#   tags   : session id
#   artifacts: every HTML plot the tools produced
#
# Auth: AmSC MLflow rejects the standard Authorization: Bearer header (it
# returns the HTML login page on /api/2.0/...). It expects an X-Api-Key
# header instead. We monkey-patch mlflow.utils.rest_utils.http_request to
# inject it on every request — same approach as the existing
# examples/LightshowAI/mlflow_tracker.py.
AM_SC_API_KEY = os.environ.get("AM_SC_API_KEY", "")
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "https://mlflow.american-science-cloud.org"
)
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "LightshowAI-XANES-chatbot")
MLFLOW_INSECURE_TLS = os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", "false")
MLFLOW_HTTP_TIMEOUT = os.environ.get("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
MLFLOW_HTTP_MAX_RETRIES = os.environ.get("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

os.environ.setdefault("MLFLOW_TRACKING_INSECURE_TLS", MLFLOW_INSECURE_TLS)
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", MLFLOW_HTTP_TIMEOUT)
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", MLFLOW_HTTP_MAX_RETRIES)


def _is_enabled(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _configure_mlflow_tls_warnings() -> None:
    """Silence urllib3 warnings only when insecure TLS is explicitly enabled."""
    if not _is_enabled(MLFLOW_INSECURE_TLS):
        return
    import urllib3
    from urllib3.exceptions import InsecureRequestWarning

    urllib3.disable_warnings(InsecureRequestWarning)
    sys.stderr.write(
        "[chatbot] WARNING: MLflow TLS verification is disabled because "
        "MLFLOW_TRACKING_INSECURE_TLS is true.\n"
    )


def _patch_mlflow_x_api_key(api_key: str) -> None:
    """Inject X-Api-Key into every MLflow REST request (AmSC auth scheme)."""
    import mlflow.utils.rest_utils as rest_utils
    if getattr(rest_utils.http_request, "_amsc_patched", False):
        return  # idempotent
    original = rest_utils.http_request

    def patched(host_creds, endpoint, method, *args, **kwargs):
        headers = dict(kwargs.get("extra_headers") or {})
        headers["X-Api-Key"] = api_key
        kwargs["extra_headers"] = headers
        kwargs.pop("headers", None)
        return original(host_creds, endpoint, method, *args, **kwargs)

    patched._amsc_patched = True  # type: ignore[attr-defined]
    rest_utils.http_request = patched


_configure_mlflow_tls_warnings()

MLFLOW_ENABLED = bool(AM_SC_API_KEY and MLFLOW_TRACKING_URI)
if MLFLOW_ENABLED:
    try:
        _patch_mlflow_x_api_key(AM_SC_API_KEY)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        sys.stderr.write(
            f"[chatbot] MLflow logging configured for {MLFLOW_TRACKING_URI} "
            f"experiment='{MLFLOW_EXPERIMENT}' timeout={MLFLOW_HTTP_TIMEOUT}s "
            f"retries={MLFLOW_HTTP_MAX_RETRIES}\n"
        )
    except Exception as exc:
        sys.stderr.write(f"[chatbot] MLflow init failed (logging disabled): {exc}\n")
        MLFLOW_ENABLED = False
else:
    sys.stderr.write(
        "[chatbot] MLflow logging disabled (set AM_SC_API_KEY + MLFLOW_TRACKING_URI to enable)\n"
    )

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

Uploaded files:
  • When the user uploads files, the chat UI appends their stable local paths to
    the user message. Use those exact paths. Do not ask the user for a file path
    when an uploaded .dat file path is already listed.
  • For experimental XANES data, treat uploaded .dat files as standards and use
    their parent directory as the experimental data directory.

Slash commands available (located in .claude/commands/ within this bundle):
  • /xanes-analysis — full Ti/Fe/V/etc K-edge benchmarking workflow vs experimental
    .dat standards with shift-optimized Pearson/Spearman/Cos(∂) ranking.
    Invoke when the user says things like "benchmark TiO2 against my experimental
    data" or "rank Fe oxide candidates against this XAS spectrum".

If the user asks for something outside XANES / Materials Project, say so plainly
and ask whether they want you to proceed anyway.
"""

# --- MCP server config ------------------------------------------------------
PYTHON = sys.executable
# AM_SC_API_KEY, MLFLOW_TRACKING_URI, MLFLOW_INSECURE_TLS are read above
# in the MLflow init block — reused here to wire the mlflow-amsc MCP server.

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
            "MLFLOW_TRACKING_INSECURE_TLS": MLFLOW_INSECURE_TLS,
            "MLFLOW_HTTP_REQUEST_TIMEOUT": MLFLOW_HTTP_TIMEOUT,
            "MLFLOW_HTTP_REQUEST_MAX_RETRIES": MLFLOW_HTTP_MAX_RETRIES,
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
_HTML_PATH_RE = re.compile(r"(?:/|~)[^\s<>'\"]+?\.html\b")


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


def _html_url(path: Path) -> str:
    """Return the browser-visible URL for a saved HTML artifact."""
    rel_path = path.resolve().relative_to(PLOT_DIR.resolve()).as_posix()
    return f"{PLOTS_PUBLIC_URL}/{quote(rel_path, safe='/')}"


def _html_label(path: Path) -> str:
    """Return a compact label for a saved plot/viewer artifact."""
    return "Crystal structure" if "structure" in path.name else "XANES spectrum"


def _iframe_height(path: Path) -> int:
    """Structure viewers need more vertical room than spectrum plots."""
    return 680 if "structure" in path.name else 540


async def _send_inline_html(path: Path) -> None:
    """Render a saved HTML file with the Chainlit iframe custom element."""
    url = _html_url(path)
    size_kb = path.stat().st_size // 1024 if path.exists() else 0
    label = _html_label(path)
    elements = []
    custom_element = getattr(cl, "CustomElement", None)
    if custom_element is not None:
        elements.append(
            custom_element(
                name="iframe",
                display="inline",
                props={
                    "src": url,
                    "height": _iframe_height(path),
                    "title": f"{label}: {path.name}",
                },
            )
        )
    else:
        sys.stderr.write(
            "[chatbot] cl.CustomElement unavailable; sending HTML as link only\n"
        )

    await cl.Message(
        content=(
            f"### {label}: [{path.name}]({url})\n\n"
            f"Interactive preview should appear below. Open the link above for "
            f"a full-page view or to save the HTML file ({size_kb} KB)."
        ),
        elements=elements,
    ).send()


async def _render_html_files(text: str) -> list[Path]:
    """Inline-render each newly mentioned HTML artifact exactly once."""
    rendered = cl.user_session.get("rendered_html_paths")
    if rendered is None:
        rendered = set()
        cl.user_session.set("rendered_html_paths", rendered)

    artifacts: list[Path] = []
    for html in _extract_html_files(text):
        key = str(html)
        if key in rendered:
            continue
        rendered.add(key)
        artifacts.append(html)
        await _send_inline_html(html)
    return artifacts


# --- User uploads -----------------------------------------------------------
def _safe_path_part(value: str, default: str) -> str:
    """Return a filesystem-safe name while preserving useful extensions."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or default


def _element_value(element: object, key: str, default=None):
    """Read a Chainlit element field across dataclass/object/dict versions."""
    if isinstance(element, dict):
        return element.get(key, default)
    value = getattr(element, key, _MISSING)
    if value is not _MISSING:
        return value
    to_dict = getattr(element, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict().get(key, default)
        except Exception:
            return default
    return default


def _unique_upload_path(upload_dir: Path, filename: str) -> Path:
    candidate = upload_dir / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    return upload_dir / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"


def _persist_message_uploads(message: cl.Message, session_id: str) -> list[dict[str, str]]:
    """Copy uploaded files to stable paths that the Claude CLI can read."""
    uploads: list[dict[str, str]] = []
    elements = getattr(message, "elements", None) or []
    if not elements:
        return uploads

    session_dir = UPLOAD_DIR / _safe_path_part(session_id, "session")
    session_dir.mkdir(parents=True, exist_ok=True)

    for element in elements:
        source_raw = _element_value(element, "path")
        name = str(_element_value(element, "name", "") or "")
        mime = str(_element_value(element, "mime", "") or "")
        if not source_raw:
            sys.stderr.write(f"[chatbot] uploaded element has no local path: {name}\n")
            continue

        source = Path(str(source_raw)).expanduser()
        if not source.is_file():
            sys.stderr.write(
                f"[chatbot] uploaded element path is not readable: {source}\n"
            )
            continue

        filename = _safe_path_part(name or source.name, source.name or "upload.dat")
        dest = _unique_upload_path(session_dir, filename)
        try:
            shutil.copy2(source, dest)
        except Exception as exc:
            sys.stderr.write(f"[chatbot] failed to copy uploaded file {source}: {exc}\n")
            continue

        uploads.append(
            {
                "name": name or source.name,
                "path": str(dest),
                "directory": str(dest.parent),
                "mime": mime or "unknown",
                "size_bytes": str(dest.stat().st_size),
            }
        )
    return uploads


def _build_agent_prompt(user_text: str, uploads: list[dict[str, str]]) -> str:
    """Append uploaded file paths to the prompt sent to the Claude agent."""
    if not uploads:
        return user_text

    upload_lines = [
        "The user uploaded files with this message. The Chainlit app copied them "
        "to these stable local paths readable by your tools and Python scripts:",
    ]
    for upload in uploads:
        upload_lines.append(
            "- "
            f"{upload['name']}: {upload['path']} "
            f"(directory: {upload['directory']}, mime: {upload['mime']}, "
            f"size: {upload['size_bytes']} bytes)"
        )

    dat_dirs = sorted(
        {u["directory"] for u in uploads if u["path"].lower().endswith(".dat")}
    )
    if dat_dirs:
        upload_lines.append(
            "For experimental XANES .dat standards, use these directories directly: "
            + ", ".join(dat_dirs)
            + ". Do not ask the user to provide a path for these uploaded files."
        )

    return "\n".join(upload_lines) + "\n\nUser message:\n" + user_text


# --- Session lifecycle ------------------------------------------------------
@cl.on_chat_start
async def on_chat_start() -> None:
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        model=MODEL,
        mcp_servers=MCP_SERVERS,
        permission_mode="bypassPermissions",
        max_turns=30,
        # No cwd override needed — the chatbot's own directory is the cwd,
        # and .claude/commands/xanes-analysis.md sits next to this app.py.
    )
    client = ClaudeSDKClient(options=options)
    await client.__aenter__()
    cl.user_session.set("client", client)
    cl.user_session.set("steps", {})  # tool_use_id -> cl.Step
    cl.user_session.set("rendered_html_paths", set())

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
        try:
            await client.__aexit__(None, None, None)
        except Exception as exc:
            if not _is_sigterm_exit(exc):
                sys.stderr.write(f"[chatbot] client shutdown failed: {exc}\n")


# --- MLflow run helpers -----------------------------------------------------
def _mlflow_start_turn(prompt: str, user: str, session_id: str) -> dict[str, str] | None:
    """Capture turn metadata without making network calls before the reply."""
    if not MLFLOW_ENABLED:
        return None
    return {
        "run_name": f"{(prompt[:40] or 'turn').strip()}-{uuid.uuid4().hex[:6]}",
        "prompt": prompt[:500],
        "user": user,
        "session_id": session_id,
    }


def _mlflow_finish_turn(
    run_meta: dict[str, str] | None,
    *,
    latency_s: float,
    tool_calls: int,
    tool_names: list[str],
    response_text: str,
    artifacts: list[Path],
) -> None:
    """Log metrics, response, and artifact files; errors are logged but not raised."""
    if run_meta is None:
        return
    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=run_meta["run_name"]):
            mlflow.log_param("model", MODEL)
            mlflow.log_param("prompt", run_meta["prompt"])
            mlflow.log_param("user", run_meta["user"])
            mlflow.set_tag("session_id", run_meta["session_id"])
            mlflow.set_tag("source", "lightshowai-chatbot")
            mlflow.log_metric("latency_seconds", round(latency_s, 3))
            mlflow.log_metric("tool_calls", tool_calls)
            mlflow.log_metric("response_chars", len(response_text or ""))
            if tool_names:
                mlflow.log_param("tools_used", ",".join(tool_names)[:500])
            if response_text:
                mlflow.log_text(response_text, "response.md")
            for art in artifacts:
                try:
                    mlflow.log_artifact(str(art), artifact_path="plots")
                except Exception as exc:
                    sys.stderr.write(
                        f"[chatbot] log_artifact({art.name}) failed: {exc}\n"
                    )
    except Exception as exc:
        sys.stderr.write(f"[chatbot] mlflow log failed: {exc}\n")


def _is_sigterm_exit(exc: Exception) -> bool:
    """Claude CLI reports systemd/service shutdown as command exit 143."""
    message = str(exc)
    return "exit code 143" in message or "exit code: 143" in message


async def _reset_client_after_failure(client: ClaudeSDKClient) -> None:
    """Close a wedged SDK client; the browser should start a new chat session."""
    try:
        await client.__aexit__(None, None, None)
    except Exception as exc:
        if not _is_sigterm_exit(exc):
            sys.stderr.write(f"[chatbot] client reset failed: {exc}\n")
    cl.user_session.set("client", None)


# --- Message handler --------------------------------------------------------
@cl.on_message
async def on_message(message: cl.Message) -> None:
    client: ClaudeSDKClient | None = cl.user_session.get("client")
    if client is None:
        await cl.Message(
            content=(
                "The agent session is not initialized. Refresh the page and "
                "start a new chat."
            )
        ).send()
        return

    steps: dict[str, cl.Step] = cl.user_session.get("steps") or {}
    steps.clear()
    cl.user_session.set("steps", steps)

    user = cl.user_session.get("user")
    user_id = getattr(user, "identifier", "anon") if user else "anon"
    session_id = cl.user_session.get("id") or "unknown"
    uploads = _persist_message_uploads(message, session_id)
    agent_prompt = _build_agent_prompt(message.content, uploads)

    run = _mlflow_start_turn(agent_prompt, user_id, session_id)
    t0 = time.monotonic()
    tool_calls = 0
    tool_names: list[str] = []
    artifacts: list[Path] = []
    assistant_text: list[str] = []
    skip_mlflow_finish = False
    cl.user_session.set("rendered_html_paths", set())

    reply = cl.Message(content="")
    await reply.send()

    try:
        async with asyncio.timeout(CHAT_TURN_TIMEOUT_S):
            await client.query(agent_prompt)

            async for msg in client.receive_response():
                if not isinstance(msg, AssistantMessage):
                    continue
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        assistant_text.append(block.text)
                        await reply.stream_token(block.text)

                    elif isinstance(block, ToolUseBlock):
                        step = cl.Step(name=block.name, type="tool")
                        step.input = block.input
                        steps[block.id] = step
                        await step.send()
                        tool_calls += 1
                        tool_names.append(block.name)

                    elif isinstance(block, ToolResultBlock):
                        step = steps.pop(block.tool_use_id, None)
                        if step is None:
                            continue
                        step.output = block.content
                        if getattr(block, "is_error", False):
                            step.is_error = True
                        await step.update()

                        artifacts.extend(await _render_html_files(str(block.content)))

            await reply.update()
            artifacts.extend(await _render_html_files("".join(assistant_text)))
    except TimeoutError:
        sys.stderr.write(
            f"[chatbot] chat turn timed out after {CHAT_TURN_TIMEOUT_S}s; "
            "resetting Claude SDK client.\n"
        )
        await _reset_client_after_failure(client)
        timeout_text = (
            "\n\nThe agent timed out and the session was reset. Refresh the page "
            "and start a new chat, or retry with a narrower request."
        )
        assistant_text.append(timeout_text)
        await reply.stream_token(timeout_text)
        await reply.update()
    except Exception as exc:
        if _is_sigterm_exit(exc):
            skip_mlflow_finish = True
            sys.stderr.write(
                "[chatbot] Claude CLI exited with 143/SIGTERM during service shutdown; "
                "suppressing traceback.\n"
            )
            return
        raise
    finally:
        if not skip_mlflow_finish:
            _mlflow_finish_turn(
                run,
                latency_s=time.monotonic() - t0,
                tool_calls=tool_calls,
                tool_names=tool_names,
                response_text="".join(assistant_text),
                artifacts=artifacts,
            )
