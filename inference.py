print("DEBUG: inference started", flush=True)
import os
import json
from openai import OpenAI
from env.hospital_env import HospitalEnv

# ==============================
# 🔑 Setup client
# ==============================
client = OpenAI(
    base_url=os.getenv("API_BASE_URL").strip(),
    api_key=os.getenv("HF_TOKEN").strip(),
)

MODEL_NAME = os.getenv("MODEL_NAME")


# ==============================
# 🧠 Safe JSON extraction
# ==============================
def safe_parse(text):
    try:
        return json.loads(text)
    except:
        # try extracting JSON block
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except:
            return {
                "department": "general",
                "seriousness": 3
            }


# ==============================
# 🤖 LLM decision
# ==============================
def ask_llm(state):
    prompt = f"""
You are a hospital triage system.

STRICT RULES:
- Use "emergency" ONLY for:
  - unconscious
  - severe bleeding

- Use cardiology for:
  - chest pain
  - palpitations

- Use pulmonology for:
  - shortness of breath
  - cough

- Use neurology for:
  - head injury
  - dizziness

- Use orthopedics for:
  - fracture

- Use general for:
  - fever or mild symptoms

Seriousness:
1–2 → mild
3 → moderate
4–5 → severe

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

        return {
            "department": "general",
            "seriousness": 3
        }


# ==============================
# 🚀 MAIN LOOP
# ==============================
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

        # ==========================
        # 🔥 NORMALIZE REWARD (CRITICAL)
        # ==========================
        # your env gives roughly -3 → +5
        reward = (reward + 3) / 8
        reward = max(0.001, min(0.999, reward))

        total_reward += reward
        steps += 1

        print(
            f"[STEP] step={step} state={state} action={action} reward={reward}",
            flush=True
        )

        state = next_state
        step += 1

    # ==========================
    # 🏁 FINAL SCORE
    # ==========================
    score = total_reward / steps if steps > 0 else 0.0
    score = max(0.0, min(1.0, score))

    print(f"[END] score={score}", flush=True)


# ==============================
# ENTRY
# ==============================
if __name__ == "__main__":
    run()

    # keep container alive for HF
    import time
    while True:
        time.sleep(60)