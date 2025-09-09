from fastapi import FastAPI
from pydantic import BaseModel
from openai import AzureOpenAI
import os, re, json, requests
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

app = FastAPI()

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

user_sessions = {}

class ChatbotInput(BaseModel):
    user: str
    message: str

def classify_intent(message: str) -> str:
    system_prompt = (
        "Classify this HR message into one of the following only: onboarding, leave_request, pulse_check."
        "Examples:\n"
        "I want to take Monday off => leave_request\n"
        "We have a new developer joining => onboarding\n"
        "Here's the team’s monthly feedback => pulse_check"
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

@app.post("/route")
async def route_request(data: ChatbotInput):
    user = data.user
    message = data.message.strip().lower()

    session = user_sessions.get(user, {
        "intent": None,
        "fields": {},
        "pending_confirmation": False
    })

    # ✅ Handle pending confirmation
    if session["pending_confirmation"]:
        if message in ["yes", "confirm", "yes please", "submit"]:
            try:
                intent = session["intent"]
                payload = session["fields"]
                payload["source_user"] = user

                response = requests.post(WEBHOOKS[intent], json=payload)
                response.raise_for_status()
                del user_sessions[user]

                return {
                    "status": "success",
                    "message": f"✅ Your `{intent}` workflow has been submitted successfully!"
                }

            except Exception as e:
                return {
                    "status": "error",
                    "message": "🚫 Submission failed.",
                    "details": str(e)
                }
        elif message in ["no", "cancel", "not now"]:
            del user_sessions[user]
            return {
                "status": "cancelled",
                "message": "❌ Okay! I’ve cancelled the request."
            }
        else:
            return {
                "status": "waiting",
                "message": "⚠️ I’m still waiting for a *yes* or *no* to proceed."
            }

    # ✅ Ignore greetings or filler input
    casual_messages = ["hi", "hello", "hey", "yo", "how are you", "good morning", "good afternoon"]
    if message in casual_messages:
        return {
            "status": "neutral",
            "message": "👋 Hi again! I can help with onboarding, leave requests, or pulse checks. What would you like to do?"
        }

    # ✅ Reclassify intent based on real message
    try:
        intent = classify_intent(message)
    except Exception as e:
        return {"status": "error", "message": "Could not classify intent", "details": str(e)}

    # New or changed intent? Reset fields
    if session["intent"] != intent:
        session = {
            "intent": intent,
            "fields": {},
            "pending_confirmation": False
        }

    # Try to extract new fields
    try:
        new_fields = extract_fields(intent, message)
        session["fields"].update({k: v for k, v in new_fields.items() if v})
    except Exception:
        pass

    # Pulse check should auto-fill known email
    if intent == "pulse_check":
        session["fields"]["email"] = user

    # Check if all required fields are ready
    required = REQUIRED_FIELDS[intent]
    missing = [k for k in required if not session["fields"].get(k)]

    if not missing:
        session["pending_confirmation"] = True
        user_sessions[user] = session
        return {
            "status": "confirm",
            "message": f"✅ Got everything for `{intent}`. Shall I go ahead and submit it?"
        }
    else:
        user_sessions[user] = session
        return {
            "status": "incomplete",
            "intent": intent,
            "message": f"Thanks! I still need: {', '.join(missing)}"
        }
