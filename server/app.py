from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from env.hospital_env import HospitalEnv
from inference import ask_llm
import threading
import subprocess

app = FastAPI()



# HOME UI

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Smart Hospital Demo</title>
        </head>
        <body style="font-family: Arial; padding: 20px;">
        <div style="text-align: center;">
            <h1>Smart Hospital RL Environment 🏥</h1>
            <p>Interactive triage simulation</p>
            </div>
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


@app.post("/reset")
def reset():
    env = HospitalEnv(task="easy", max_steps=1)
    return {"state": env.reset()}


@app.get("/demo")
def demo():
    env = HospitalEnv(task="easy", max_steps=5)
    state = env.reset()

    steps = []

    for _ in range(5):
        action = ask_llm(state)
        state, reward, done, _ = env.step(action)

        steps.append({
            "action": action,
            "reward": reward
        })

        if done:
            break

    return {"simulation": steps}


def run_inference():
    try:
        subprocess.run(["python", "inference.py"], check=True)
    except Exception as e:
        print(f"[SERVER ERROR] {e}")


@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=run_inference)
    thread.start()


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()