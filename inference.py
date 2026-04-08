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
    print("[WARNING] Missing env vars → using fallback", flush=True)
    USE_LLM = False

client = None
if USE_LLM:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

TASK_NAME = "hospital-triage"
BENCHMARK = "hospital-env"
MAX_STEPS = 10


# SAFE JSON PARSER
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


# LLM DECISION
def ask_llm(state):
    if not USE_LLM or client is None:
        return fallback_policy(state)
    prompt = f"""
    You are a STRICT hospital triage system.

    CRITICAL RULES:
    - If unconscious OR severe bleeding → emergency, seriousness 5
    - chest pain + shortness of breath → cardiology, seriousness 5

    DEPARTMENTS:
    - chest pain → cardiology
    - breathing issues → pulmonology
    - head injury → neurology
    - fracture → orthopedics

    SERIOUSNESS:
    5 = critical, 4 = severe, 3 = moderate, 2 = mild

    Patient:
    Symptoms: {state.get('symptoms', [])}
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
    env = HospitalEnv(task="hard", max_steps=MAX_STEPS)

    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    state = env.reset()
    done = False

    step = 1
    rewards = []
    total_reward = 0.0

    try:
        while not done and step <= MAX_STEPS:

            # difficulty visibility (based on symptoms)
            num_symptoms = len(state.get("symptoms", []))
            if num_symptoms == 1:
                difficulty = "easy"
            elif num_symptoms == 2:
                difficulty = "medium"
            else:
                difficulty = "hard"

            # get action
            action = ask_llm(state)

            next_state, reward, done, info = env.step(action)

            # normalize reward → [0,1]
            reward = (reward + 3) / 8
            reward = max(0.001, min(0.999, reward))

            rewards.append(reward)
            total_reward += reward

            done_str = str(done).lower()

            # STRICT STEP FORMAT
            print(
                f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error=null",
                flush=True
            )

            # 🔥 OPTIONAL DEBUG (does NOT break format)
            selected_dept = action.get("department", "general")
            queue_info = info.get("queue_status", {}).get(selected_dept, {})

            print(
                f"[DEBUG] symptoms={state['symptoms']} difficulty={difficulty} queue={selected_dept}:{queue_info}",
                flush=True
            )

            state = next_state
            step += 1

        # FINAL SCORE
        steps_taken = len(rewards)
        score = total_reward / steps_taken if steps_taken > 0 else 0.0
        score = max(0.001, min(0.999, score))

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


# RUN
if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"[FATAL] {e}", flush=True)
        print("[END] success=false steps=0 score=0.00 rewards=", flush=True)

    while True:
        time.sleep(60)