#!/bin/bash -e
# One-shot installer for the LightshowAI chatbot on a fresh Ubuntu 22.04 EC2 box.
#
# Assumes:
#   - Ubuntu 22.04 (Amazon Linux: swap apt-get for dnf, package names differ)
#   - Run as the `ubuntu` user (sudo where needed)
#   - This script lives at <repo>/examples/LightshowAI/aws/chatbot/setup.sh and
#     the repo has been cloned to ~/agentic_workflows (or wherever — paths are
#     resolved from this script's location).
#
# After this finishes:
#   chainlit run app.py -h --host 0.0.0.0 --port 8000

set -euo pipefail

CHATBOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CHATBOT_DIR}/../../../.." && pwd)"
LIGHTSHOWAI_DIR="${REPO_ROOT}/examples/LightshowAI"
VENV="${HOME}/venv-chatbot"

echo "==> Repo root:    ${REPO_ROOT}"
echo "==> Chatbot dir:  ${CHATBOT_DIR}"
echo "==> LightshowAI:  ${LIGHTSHOWAI_DIR}"
echo "==> Venv target:  ${VENV}"

# --- 1. system packages ----------------------------------------------------
echo "==> Installing system packages..."
sudo apt-get update -y
sudo apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    git build-essential cmake \
    nodejs npm   # claude-agent-sdk shells out to a node-based MCP transport

# --- 2. claude-code CLI (claude_agent_sdk depends on it at runtime) --------
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

# --- 5. LightshowAI package (editable install from this repo) --------------
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
  1. Edit ${CHATBOT_DIR}/.env and fill in ANTHROPIC_API_KEY, MP_API_KEY, CHAINLIT_PASSWORD.
  2. Open port 8000 in the EC2 security group (Custom TCP, source: My IP or 0.0.0.0/0).
  3. Test interactively:
       cd ${CHATBOT_DIR}
       source ${VENV}/bin/activate
       chainlit run app.py -h --host 0.0.0.0 --port 8000
  4. Or install as a systemd service:
       sudo cp ${CHATBOT_DIR}/lightshowai-chatbot.service /etc/systemd/system/
       sudo systemctl daemon-reload
       sudo systemctl enable --now lightshowai-chatbot
       sudo journalctl -u lightshowai-chatbot -f

URL: http://<ec2-public-ip>:8000
EOF
