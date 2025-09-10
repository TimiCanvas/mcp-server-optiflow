import os
import json
import re
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI
from mcp.server.fastmcp import FastMCP

# Load environment variables
load_dotenv()

# Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Define webhooks
WEBHOOKS = {
    "leave_request": "https://starfish-special-bulldog.ngrok-free.app/webhook/leave-request",
    "onboarding": "https://starfish-special-bulldog.ngrok-free.app/webhook/onboarding",
    "pulse_check": "https://starfish-special-bulldog.ngrok-free.app/webhook/pulse-check",
}

# Required fields
REQUIRED_FIELDS = {
    "onboarding": [
        "employee_id", "first_name", "last_name", "email",
        "department", "role", "start_date", "manager_email"
    ],
    "leave_request": ["employee_id", "start_date", "end_date", "reason"],
    "pulse_check": [],
}

# Initialize MCP server
server = FastMCP("mcp-server-optiflow")

def classify_intent(message: str) -> str:
    system_prompt = (
        "You're a friendly HR assistant. Respond politely to greetings/small talk, "
        "but also classify HR-related messages into one of the following only: "
        "onboarding, leave_request, pulse_check.\n"
        "Examples:\n"
        "I want to take Monday off => leave_request\n"
        "We have a new developer joining => onboarding\n"
        "Here's the team’s monthly feedback => pulse_check"
    )
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        temperature=0,
    )
    intent = response.choices[0].message.content.strip().lower()
    if intent not in WEBHOOKS:
        return f"smalltalk::{intent}"
    return intent

def extract_fields(intent: str, message: str) -> dict:
    if intent not in REQUIRED_FIELDS:
        return {}
    prompt = (
        f"You are extracting structured JSON from this HR message. Intent: {intent}. "
        f"Return ONLY a JSON with fields: {REQUIRED_FIELDS[intent]}"
    )
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ],
        temperature=0,
    )
    content = re.sub(r"^```(json)?\s*|\s*```$", "", response.choices[0].message.content.strip())
    return json.loads(content)

# -----------------------------
# MCP Tools
# -----------------------------
@server.tool()
def classify(message: str) -> str:
    return classify_intent(message)

@server.tool()
def extract(intent: str, message: str) -> dict:
    return extract_fields(intent, message)

@server.tool()
def get_required_fields(intent: str) -> list:
    return REQUIRED_FIELDS.get(intent, [])

@server.tool()
def get_webhook(intent: str) -> str:
    return WEBHOOKS.get(intent, "")

@server.tool()
def confirm_routing(intent: str, data: dict, confirm: bool = False) -> str:
    """Ask for confirmation before sending data to the webhook."""
    if not confirm:
        return f"⚠️ Do you want me to send this {intent} data to {WEBHOOKS[intent]}? Reply 'yes' to confirm."
    
    try:
        response = requests.post(WEBHOOKS[intent], json=data, timeout=10)
        if response.status_code == 200:
            return f"✅ Data successfully sent to {intent} webhook."
        else:
            return f"❌ Failed to send data to webhook. Status: {response.status_code}"
    except Exception as e:
        return f"❌ Error while sending to webhook: {e}"

# -----------------------------
# Run MCP server
# -----------------------------
if __name__ == "__main__":
    if os.getenv("RAILWAY_ENVIRONMENT"):  # ✅ running on Railway
        from fastapi import FastAPI
        import uvicorn

        app = FastAPI()
        app.mount("/", server.app)  # expose MCP server at root

        uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
    else:  # ✅ local dev / stdio
        server.run()