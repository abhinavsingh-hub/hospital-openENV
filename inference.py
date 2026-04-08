import os
import json
import time
from openai import OpenAI
from env.hospital_env import HospitalEnv

# Setup client
base_url = os.getenv("API_BASE_URL")
api_key = os.getenv("HF_TOKEN")

if not base_url or not api_key:
    raise ValueError("Missing environment variables")

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

MODEL_NAME = os.getenv("MODEL_NAME")


# JSON extraction
def safe_parse(text):
    try:
        return json.loads(text)
    except:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except:
            return {
                "department": "general",
                "seriousness": 3
            }


# FALLBACK POLICY
def fallback_policy(state):
    symptoms = " ".join(state["symptoms"]).lower()

    if "unconscious" in symptoms or "severe bleeding" in symptoms:
        return {"department": "emergency", "seriousness": 5}

    if "chest pain" in symptoms or "palpitations" in symptoms:
        return {"department": "cardiology", "seriousness": 4}

    if "shortness of breath" in symptoms or "cough" in symptoms:
        return {"department": "pulmonology", "seriousness": 3}

    if "head injury" in symptoms or "dizziness" in symptoms:
        return {"department": "neurology", "seriousness": 3}

    if "fracture" in symptoms:
        return {"department": "orthopedics", "seriousness": 3}

    return {"department": "general", "seriousness": 2}


# LLM decision
def ask_llm(state):
    prompt = f"""
    You are a STRICT hospital triage system.

    You MUST follow these rules EXACTLY:

    CRITICAL RULES (highest priority):
    - If "unconscious" OR "severe bleeding" → department = emergency, seriousness = 5
    - If BOTH "chest pain" AND "shortness of breath" → department = cardiology, seriousness = 5

    DEPARTMENT RULES:
    - chest pain OR palpitations → cardiology
    - shortness of breath OR cough → pulmonology
    - head injury OR dizziness → neurology
    - fracture → orthopedics
    - fever → general

    SCORING PRIORITY:
    - Emergency overrides EVERYTHING
    - Multi-symptom conflicts → choose MOST SEVERE condition
    - If multiple serious symptoms → increase seriousness

    SERIOUSNESS RULES:
    - 5 → life-threatening (unconscious, bleeding, chest pain + breath)
    - 4 → severe (fracture + other symptoms, strong vitals)
    - 3 → moderate
    - 2 → mild
    - 1 → very mild

    Patient:
    Symptoms: {state['symptoms']}
    Age: {state['age']}
    Heart Rate: {state['heart_rate']}
    Blood Pressure: {state['blood_pressure']}

    Return ONLY JSON:
    {{
    "department": "...",
    "seriousness": <1-5>
    }}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        text = (response.choices[0].message.content or "").strip()
        return safe_parse(text)

    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return fallback_policy(state)


# MAIN LOOP
def run():
    env = HospitalEnv(task="hard", max_steps=10)

    print("[START]", flush=True)

    state = env.reset()
    done = False
    step = 0

    total_reward = 0.0
    steps = 0

    while not done:
        action = ask_llm(state)

        next_state, reward, done, info = env.step(action)

        # NORMALIZE REWARD
        reward = (reward + 3) / 8
        reward = max(0.0, min(1.0, reward))

        total_reward += reward
        steps += 1

        print(
            f"[STEP] step={step} state={state} action={action} reward={reward}",
            flush=True
        )

        state = next_state
        step += 1

    score = total_reward / steps if steps > 0 else 0.0
    score = max(0.0, min(1.0, score))

    print(f"[END] score={score}", flush=True)


if __name__ == "__main__":
    run()

    while True:
        time.sleep(60)