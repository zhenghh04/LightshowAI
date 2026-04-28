#!/bin/bash -e
# One-shot installer for the self-contained LightshowAI chatbot on Ubuntu 22.04.
#
# Expected layout (everything under one folder, no broader repo needed):
#   <root>/lightshowai/        -- the python package
#   <root>/model_checkpoints/  -- baked OmniXAS .ckpt files
#   <root>/mcp/                -- materials_project + lightshowai MCP servers
#   <root>/aws/chatbot/        -- this app
#
# Run as the `ubuntu` user from anywhere; paths resolve from the script's location.

set -euo pipefail

CHATBOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIGHTSHOWAI_DIR="$(cd "${CHATBOT_DIR}/../.." && pwd)"
VENV="${HOME}/venv-chatbot"

echo "==> LightshowAI root: ${LIGHTSHOWAI_DIR}"
echo "==> Chatbot dir:      ${CHATBOT_DIR}"
echo "==> Venv target:      ${VENV}"

# --- 1. system packages ----------------------------------------------------
echo "==> Installing system packages..."
sudo apt-get update -y
sudo apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    git build-essential cmake \
    nodejs npm    # claude-agent-sdk shells out to a node-based MCP transport

# --- 2. claude-code CLI (claude_agent_sdk runtime dependency) --------------
echo "==> Installing @anthropic-ai/claude-code CLI..."
sudo npm install -g @anthropic-ai/claude-code

# --- 3. virtualenv ---------------------------------------------------------
echo "==> Creating venv at ${VENV}..."
python3.11 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install --upgrade pip wheel

# --- 4. chatbot deps -------------------------------------------------------
echo "==> Installing chatbot requirements..."
pip install -r "${CHATBOT_DIR}/requirements.txt"

# --- 5. LightshowAI package (editable install, picks up local checkpoints) -
echo "==> Installing LightshowAI from ${LIGHTSHOWAI_DIR}..."
pip install -e "${LIGHTSHOWAI_DIR}"

# --- 6. .env -----------------------------------------------------------------
if [ ! -f "${CHATBOT_DIR}/.env" ]; then
    echo "==> Copying .env.example -> .env (EDIT IT!)"
    cp "${CHATBOT_DIR}/.env.example" "${CHATBOT_DIR}/.env"
fi

cat <<EOF

==> Done.

Next steps:
  1. Edit ${CHATBOT_DIR}/.env and fill in:
       ANTHROPIC_API_KEY        (required)
       MP_API_KEY               (required)
       CHAINLIT_PASSWORD        (recommended)
       CLAUDE_MODEL             (optional; default claude-opus-4-7 — set to
                                claude-sonnet-4-6 if your key lacks 4-7 access)

  2. Open port 8000 in the EC2 security group
       Type: Custom TCP, Port: 8000, Source: My IP

  3. Test interactively:
       cd ${CHATBOT_DIR}
       source ${VENV}/bin/activate
       chainlit run app.py -h --host 0.0.0.0 --port 8000

  4. Or install as a systemd service (after editing the WorkingDirectory /
     EnvironmentFile / ReadWritePaths to match ${LIGHTSHOWAI_DIR} on this box):
       sudo cp ${CHATBOT_DIR}/lightshowai-chatbot.service /etc/systemd/system/
       sudo systemctl daemon-reload
       sudo systemctl enable --now lightshowai-chatbot
       sudo journalctl -u lightshowai-chatbot -f

URL: http://<ec2-public-ip>:8000
EOF
