import os
import json
import time
from openai import OpenAI
from env.hospital_env import HospitalEnv


#  ENV SETUP

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("HF_TOKEN")

USE_LLM = True
if not API_BASE_URL or not MODEL_NAME or not API_KEY:
    print("[WARNING] Missing env vars → fallback mode", flush=True)
    USE_LLM = False

client = None
if USE_LLM:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

TASK_NAME = "hospital-triage"
BENCHMARK = "hospital-env"
MAX_STEPS = 5  

#  SAFE PARSE

def safe_parse(text):
    try:
        return json.loads(text)
    except:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except:
            return fallback_policy({})

#  NORMALIZE ACTION 

def normalize_action(action):
    try:
        dept = action.get("department", "").lower().strip()

        allowed = {
            "cardiology",
            "neurology",
            "orthopedics",
            "pulmonology",
            "general",
            "emergency"
        }

        if dept not in allowed:
            return fallback_policy({})

        seriousness = int(action.get("seriousness", 3))
        seriousness = max(1, min(5, seriousness))

        return {
            "department": dept,
            "seriousness": seriousness
        }

    except:
        return fallback_policy({})

#  FALLBACK POLICY

def fallback_policy(state):
    symptoms = " ".join(state.get("symptoms", [])).lower()

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

#  LLM DECISION 

def ask_llm(state):
    if not USE_LLM or client is None:
        return fallback_policy(state)

    symptoms = state.get("symptoms", [])
    age = state.get("age", 0)
    hr = state.get("heart_rate", 0)
    bp = state.get("blood_pressure", 0)

    prompt = f"""
You are an intelligent hospital triage system.

Your task:
- Assign correct department
- Assign seriousness (1–5)

If multiple symptoms exist, prioritize life-threatening ones,
but still consider the dominant system affected.

GUIDELINES:
- Emergency: life-threatening (unconscious, severe bleeding)
- Cardiology: chest-related symptoms
- Pulmonology: breathing issues
- Neurology: brain-related symptoms
- Orthopedics: bone injuries
- General: mild or unclear cases

IMPORTANT:
- Use reasoning for unseen symptoms
- Consider combinations
- Use vitals and age for severity

Seriousness:
1–2 → mild
3 → moderate
4–5 → severe/critical

Patient:
Symptoms: {symptoms}
Age: {age}
Heart Rate: {hr}
Blood Pressure: {bp}

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
            temperature=0,
            timeout=5  
        )

        text = (response.choices[0].message.content or "").strip()
        return normalize_action(safe_parse(text))

    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return fallback_policy(state)

#  MAIN LOOP

def run():
    env = HospitalEnv(task="hard", max_steps=MAX_STEPS)

    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    state = env.reset()
    done = False

    step = 1
    rewards = []
    total_reward = 0.0

    start_time = time.time()
    MAX_RUNTIME = 30  

    try:
        while not done and step <= MAX_STEPS:

            if time.time() - start_time > MAX_RUNTIME:
                print("[DEBUG] Global timeout reached", flush=True)
                break

            action = ask_llm(state)

            next_state, reward, done, info = env.step(action)

            reward = (reward + 3) / 8
            reward = max(0.001, min(0.999, reward))

            rewards.append(reward)
            total_reward += reward

            done_str = str(done).lower()
            error_val = "null"

            print(
                f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_val}",
                flush=True
            )

            selected_dept = action.get("department", "general")
            queue_info = info.get("queue_status", {}).get(selected_dept, {})

            print(
                f"[DEBUG] symptoms={state.get('symptoms', [])} queue={selected_dept}:{queue_info}",
                flush=True
            )

            state = next_state
            step += 1

        steps_taken = len(rewards)
        score = total_reward / steps_taken if steps_taken > 0 else 0.0
        score = max(0.01, min(0.99, score))

        success = score >= 0.5

    except Exception as e:
        print(f"[DEBUG] Runtime error: {e}", flush=True)
        success = False
        steps_taken = len(rewards)
        score = 0.0

    finally:
        # safe close
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception as e:
                print(f"[DEBUG] env.close error: {e}", flush=True)

        rewards_str = ",".join(f"{r:.2f}" for r in rewards)

        print(
            f"[END] success={str(success).lower()} steps={steps_taken} score={score:.3f} rewards={rewards_str}",
            flush=True
        )

#  ENTRY

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"[FATAL] {e}", flush=True)
        print("[END] success=false steps=0 score=0.000 rewards=", flush=True)

    while True:
        time.sleep(60)