from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from env.hospital_env import HospitalEnv
import uvicorn
import random

app = FastAPI()


# ==============================
# 🏠 HOME UI
# ==============================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Smart Hospital Demo</title>
        </head>
        <body style="font-family: Arial; padding: 20px; text-align: center;">
            <h1>Smart Hospital RL Environment 🏥</h1>
            <p>Interactive triage simulation</p>

            <button onclick="runDemo()">▶️ Run Simulation</button>

            <pre id="output" style="margin-top:20px; background:#111; color:#0f0; padding:10px;"></pre>

            <script>
                async function runDemo() {
                    const res = await fetch('/demo');
                    const data = await res.json();
                    document.getElementById('output').innerText = JSON.stringify(data, null, 2);
                }
            </script>
        </body>
    </html>
    """


# ==============================
# 🔁 RESET (validator)
# ==============================
@app.post("/reset")
def reset():
    env = HospitalEnv(task="easy", max_steps=1)
    state = env.reset()
    return {"state": state}


# ==============================
# 🎮 DEMO SIMULATION
# ==============================
@app.get("/demo")
def demo():
    env = HospitalEnv(task="hard", max_steps=5)
    state = env.reset()

    steps = []

    for step in range(5):
        # simple heuristic (same as fallback)
        symptoms = " ".join(state["symptoms"]).lower()

        if "unconscious" in symptoms or "severe bleeding" in symptoms:
            action = {"department": "emergency", "seriousness": 5}
        elif "chest pain" in symptoms:
            action = {"department": "cardiology", "seriousness": 4}
        elif "shortness of breath" in symptoms:
            action = {"department": "pulmonology", "seriousness": 3}
        elif "head injury" in symptoms or "dizziness" in symptoms:
            action = {"department": "neurology", "seriousness": 3}
        elif "fracture" in symptoms:
            action = {"department": "orthopedics", "seriousness": 3}
        else:
            action = {"department": "general", "seriousness": 2}

        next_state, reward, done, info = env.step(action)

        steps.append({
            "step": step,
            "state": state,
            "action": action,
            "reward": reward
        })

        state = next_state
        if done:
            break

    return {"simulation": steps}


# ==============================
# 🚀 ENTRYPOINT (required)
# ==============================
def main():
    return app


if __name__ == "__main__":
    main()