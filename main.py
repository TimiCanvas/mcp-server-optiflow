import os, re, json, requests
from dotenv import load_dotenv
from openai import AzureOpenAI
from mcp.server import Server
from mcp.server.stdio import stdio_server

load_dotenv()

# === Azure OpenAI client ===
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# === HR Webhooks ===
WEBHOOKS = {
    "leave_request": "https://starfish-special-bulldog.ngrok-free.app/webhook/leave-request",
    "onboarding": "https://starfish-special-bulldog.ngrok-free.app/webhook/onboarding",
    "pulse_check": "https://starfish-special-bulldog.ngrok-free.app/webhook/pulse-check"
}

REQUIRED_FIELDS = {
    "onboarding": ["employee_id", "first_name", "last_name", "email", "department", "role", "start_date", "manager_email"],
    "leave_request": ["employee_id", "start_date", "end_date", "reason"],
    "pulse_check": []
}

# === Helper Functions ===
def classify_intent(message: str) -> str:
    system_prompt = (
        "Classify this HR message into one of the following only: onboarding, leave_request, pulse_check."
    )
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": message}],
        temperature=0
    )
    intent = response.choices[0].message.content.strip().lower()
    if intent not in WEBHOOKS:
        raise ValueError(f"Invalid intent: {intent}")
    return intent

def extract_fields(intent: str, message: str) -> dict:
    prompt = (
        f"You are extracting structured JSON from this HR message. Intent: {intent}. "
        f"Return ONLY a JSON with fields: {REQUIRED_FIELDS[intent]}"
    )
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": message}],
        temperature=0
    )
    content = re.sub(r"^```(json)?\s*|\s*```$", "", response.choices[0].message.content.strip())
    return json.loads(content)

# === MCP Server ===
server = Server("optiflow-mcp")

@server.tool()
def process_hr_message(user: str, message: str) -> dict:
    """
    Process an HR request. Supports: onboarding, leave_request, pulse_check.
    """
    try:
        intent = classify_intent(message)
        fields = extract_fields(intent, message)

        # Pulse check auto-fills email
        if intent == "pulse_check":
            fields["email"] = user

        missing = [k for k in REQUIRED_FIELDS[intent] if not fields.get(k)]

        if missing:
            return {
                "status": "incomplete",
                "intent": intent,
                "message": f"Still need: {', '.join(missing)}"
            }

        # Submit workflow
        fields["source_user"] = user
        response = requests.post(WEBHOOKS[intent], json=fields)
        response.raise_for_status()

        return {
            "status": "success",
            "intent": intent,
            "message": f"✅ {intent} submitted successfully!"
        }

    except Exception as e:
        return {"status": "error", "details": str(e)}

# === Entry point ===
if __name__ == "__main__":
    stdio_server(server).run()