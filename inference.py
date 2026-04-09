import os
import sys
import json
from openai import OpenAI
from env.hospital_env import HospitalEnv


old_stdout = sys.stdout
sys.stdout = sys.stderr

def print_log(msg):
    old_stdout.write(msg + "\n")
    old_stdout.flush()

def log_start(task, model):
    print_log(f"[START] task={task} env=hospital-env model={model}")

def log_step(step, action, reward, done, error="null"):
    print_log(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error}")

def log_end(success, steps, score, rewards):
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print_log(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}")


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b:groq")
API_KEY = os.getenv("HF_TOKEN")

USE_LLM = bool(API_KEY)

client = None
if USE_LLM:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except:
        USE_LLM = False

# RULE-BASED POLICY 

def fallback_policy(state):
    symptoms = " ".join(state.get("symptoms", [])).lower()

    # TRUE emergencies only
    if "unconscious" in symptoms or "severe bleeding" in symptoms:
        return {"department": "emergency", "seriousness": 5}

    # ❤️ cardiology
    if "chest pain" in symptoms or "palpitations" in symptoms:
        return {"department": "cardiology", "seriousness": 4}

    # 🫁 lungs
    if "shortness of breath" in symptoms or "cough" in symptoms:
        return {"department": "pulmonology", "seriousness": 3}

    # 🧠 neuro
    if "head injury" in symptoms or "dizziness" in symptoms:
        return {"department": "neurology", "seriousness": 3}

    # 🦴 ortho
    if "fracture" in symptoms:
        return {"department": "orthopedics", "seriousness": 3}

    return {"department": "general", "seriousness": 2}

# SAFE PARSE

def safe_parse(text):
    try:
        return json.loads(text)
    except:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except:
            return {}

def normalize_action(action):
    try:
        dept = action.get("department", "").lower().strip()
        allowed = {"cardiology","neurology","orthopedics","pulmonology","general","emergency"}

        if dept not in allowed:
            return fallback_policy({})

        seriousness = int(action.get("seriousness", 3))
        seriousness = max(1, min(5, seriousness))

        return {"department": dept, "seriousness": seriousness}
    except:
        return fallback_policy({})

# LLM DECISION 

def ask_llm(state):
    if not USE_LLM or client is None:
        return fallback_policy(state)

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
Symptoms: {state.get('symptoms', [])}
Age: {state.get('age', 0)}
Heart Rate: {state.get('heart_rate', 0)}
Blood Pressure: {state.get('blood_pressure', 0)}

Return ONLY JSON:
{{
  "department": "...",
  "seriousness": <1-5>
}}
"""

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=5
        )

        text = (res.choices[0].message.content or "").strip()
        return normalize_action(safe_parse(text))

    except Exception:
        return fallback_policy(state)

#  MAIN LOOP

def run_inference():

    tasks = ["easy", "medium", "hard"]

    for task_name in tasks:

        log_start(task_name, MODEL_NAME)

        env = HospitalEnv(task=task_name, max_steps=5)
        state = env.reset()

        rewards = []
        total_reward = 0.0
        done = False
        step = 1

        try:
            while not done and step <= 5:

                action = ask_llm(state)

                state, reward, done, info = env.step(action)

                # normalize reward
                reward = (reward + 3) / 8
                reward = max(0.001, min(0.999, reward))

                rewards.append(reward)
                total_reward += reward

                log_step(step, action, reward, done)

                step += 1

        except Exception:
            done = True

        steps_taken = len(rewards)
        score = total_reward / steps_taken if steps_taken > 0 else 0.0
        score = max(0.001, min(0.999, score))

        success = score >= 0.5

        log_end(success, steps_taken, score, rewards)

# ENTRY
if __name__ == "__main__":
    try:
        run_inference()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)