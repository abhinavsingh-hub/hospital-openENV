import os
import json
import time
from openai import OpenAI
from env.hospital_env import HospitalEnv

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("HF_TOKEN")

USE_LLM = True
if not API_BASE_URL or not MODEL_NAME or not API_KEY:
    print("[WARNING] Missing env vars → using fallback only", flush=True)
    USE_LLM = False

if USE_LLM:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

TASK_NAME = "hospital-triage"
BENCHMARK = "hospital-env"
MAX_STEPS = 10


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


def ask_llm(state):
    queue_info = state.get("queue", {})

    prompt = f"""
You are an intelligent hospital triage system.

CRITICAL RULES:
- unconscious OR severe bleeding → emergency (seriousness 5)
- chest pain + shortness of breath → cardiology (seriousness 5)

DEPARTMENTS:
- chest pain → cardiology
- breathing issues → pulmonology
- head injury → neurology
- fracture → orthopedics
- mild → general

QUEUE AWARENESS (VERY IMPORTANT):
- Avoid overloaded departments
- Prefer less crowded departments if medically safe
- If two departments are valid → choose the one with lower load

Patient:
Symptoms: {state.get('symptoms')}
Age: {state.get('age')}
Heart Rate: {state.get('heart_rate')}
Blood Pressure: {state.get('blood_pressure')}

Current Hospital Load:
{queue_info}

Return ONLY JSON:
{{
  "department": "...",
  "seriousness": <1-5>
}}
"""

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            text = (response.choices[0].message.content or "").strip()
            return safe_parse(text)

        except Exception as e:
            print(f"[DEBUG] attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(1)

    return fallback_policy(state)


def run():
    env = HospitalEnv(task="hard", max_steps=MAX_STEPS)

    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    state = env.reset()
    done = False

    step = 1
    rewards = []
    total_reward = 0.0

    try:
        while not done and step <= MAX_STEPS:

            state["queue"] = getattr(env, "queue_status", {})

            if USE_LLM:
                action = ask_llm(state)
            else:
                action = fallback_policy(state)

            next_state, reward, done, info = env.step(action)

            reward = (reward + 3) / 8
            reward = max(0.001, min(0.999, reward))

            rewards.append(reward)
            total_reward += reward

            print(
                f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
                flush=True
            )

            selected_dept = action["department"]
            queue_info = info.get("queue_status", {}).get(selected_dept, {})

            print(
                f"[DEBUG] symptoms={state.get('symptoms')} queue={selected_dept}:{queue_info}",
                flush=True
            )

            state = next_state
            step += 1

        # 📊 FINAL SCORE
        steps_taken = len(rewards)
        score = total_reward / steps_taken if steps_taken > 0 else 0.0
        score = max(0.0, min(1.0, score))

        success = score >= 0.5

    except Exception as e:
        print(f"[DEBUG] Runtime error: {e}", flush=True)
        success = False
        steps_taken = len(rewards)
        score = 0.0

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)

        print(
            f"[END] success={str(success).lower()} steps={steps_taken} score={score:.2f} rewards={rewards_str}",
            flush=True
        )


if __name__ == "__main__":
    run()

    # keep container alive (HF safe)
    while True:
        time.sleep(60)