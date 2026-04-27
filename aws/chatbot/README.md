# LightshowAI XANES Chatbot — EC2 Deployment

A Chainlit-based web chatbot that gives users an agentic XANES analysis assistant
in the browser. Uses the Claude Agent SDK to orchestrate two local MCP servers:

- **`materials-project`** — search Materials Project for crystal structures
- **`lightshowai`** — predict K-edge XANES via OmniXAS (FEFF for Co/Cr/Cu/Fe/Mn/Ni/Ti/V; VASP for Ti and Cu)

Replaces the need to run Claude Code in a terminal for casual users — same agent
loop, different surface.

## What you get

```
Browser  →  http://<ec2-ip>:8000  →  Chainlit UI
                                       │
                                       │  ClaudeSDKClient (one per session)
                                       ▼
                                Anthropic API (claude-opus-4-7 by default)
                                       │
                          ┌────────────┴────────────┐
                          │                         │
                  materials-project MCP      lightshowai MCP
                  (stdio subprocess)         (stdio subprocess)
                          │                         │
                  Materials Project API      OmniXAS checkpoints
                                              (already in this repo)
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Chainlit handler (auth, session, message loop with streaming + tool steps) |
| `requirements.txt` | Python deps (chainlit, claude-agent-sdk, mp-api) |
| `.env.example` | Template for ANTHROPIC_API_KEY, MP_API_KEY, CHAINLIT_PASSWORD |
| `setup.sh` | One-shot installer for a fresh Ubuntu 22.04 EC2 box |
| `lightshowai-chatbot.service` | systemd unit for running as a service |

## Deploy on EC2 (Ubuntu 22.04, t3.large or larger)

Assumes the EC2 instance you already launched. SSH in, then:

```bash
# 1. Clone the repo
sudo apt-get update -y && sudo apt-get install -y git
git clone <YOUR-REPO-URL> ~/agentic_workflows
cd ~/agentic_workflows/examples/LightshowAI/aws/chatbot

# 2. Run the installer (creates ~/venv-chatbot, installs everything)
./setup.sh

# 3. Edit secrets
nano .env
# fill in ANTHROPIC_API_KEY, MP_API_KEY, CHAINLIT_PASSWORD

# 4. Smoke test interactively
source ~/venv-chatbot/bin/activate
chainlit run app.py -h --host 0.0.0.0 --port 8000
# visit http://<ec2-public-ip>:8000

# 5. (Optional) run as a service
sudo cp lightshowai-chatbot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now lightshowai-chatbot
sudo journalctl -u lightshowai-chatbot -f
```

## Open the security group

The chatbot listens on port **8000**. In the EC2 console:

- Instance → Security tab → click the SG link → **Edit inbound rules** → **Add rule**
- Type: `Custom TCP`, Port: `8000`, Source: `My IP` (or `0.0.0.0/0` for public)
- Save

## URL

```
http://<ec2-public-ip>:8000
```

For HTTPS, put Caddy or nginx in front (Caddy auto-provisions Let's Encrypt with one config line). Out of scope for this README.

## Try it

In the chat:

> Compare Ti K-edge XANES for the three TiO₂ polymorphs (rutile, anatase, brookite) using FEFF.

> Search Materials Project for stable Cu oxides and predict the Cu K-edge XANES for each.

> Look at experiments/Fe/ in the repo and tell me which Fe oxide had the worst FEFF prediction.

The agent will call the MCP tools, render each tool call as a collapsible "Step", and stream the prose answer in the main bubble.

## Cost

Roughly:

- **EC2 `t3.large`**: ~$60/mo if always on; ~$2/mo if you stop it when not in use
- **Anthropic API (claude-opus-4-7)**: $5 / 1M input, $25 / 1M output. A typical XANES analysis turn is ~5K input + 1K output ≈ **$0.05/turn**
- **Materials Project API**: free
- **Total**: a couple of dollars a month for light demo use

To cut Anthropic cost ~3×, set `CLAUDE_MODEL=claude-sonnet-4-6` in `.env`. Slightly weaker tool-use quality, much cheaper.

## Known limitations / next steps

- **No HTTPS** — fine for `My IP` SG rule, not for cross-org demos. Add Caddy.
- **Shared password only** — for per-user accounts, swap to Cognito + Chainlit OAuth (~30 min change).
- **No persistence** — chat history is per-browser-tab. To persist conversations, enable Chainlit's data layer (Postgres backend; you have RDS available).
- **No tool-call rate-limiting** — a chatty user can burn API credit fast. Consider per-user budgets in `app.py` if this is exposed publicly.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ANTHROPIC_API_KEY not set` on startup | `.env` not loaded — check the file exists and `EnvironmentFile=` path in the systemd unit |
| Browser shows blank page | SG doesn't allow port 8000 from your IP |
| MCP calls return "tool not found" | The MCP server subprocess crashed at startup — check `journalctl -u lightshowai-chatbot` for stderr |
| LightshowAI predictions hang | First call loads 75 MB of model weights into memory — give it 30–60 s on `t3.medium`. Bump to `t3.large` if it's a recurring problem. |
| `mp_api` rate-limit errors | The MP free tier is generous but not infinite. If hit during a heavy benchmark session, pause and retry. |
