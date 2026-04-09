from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from env.hospital_env import HospitalEnv
from inference import ask_llm
import threading
import subprocess

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Smart Hospital AI | RL Environment</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #00f2ff;
                --secondary: #7000ff;
                --bg: #050505;
                --card-bg: rgba(20, 20, 20, 0.7);
                --text: #ffffff;
                --text-dim: #a0a0a0;
                --accent: #10b981;
                --error: #ff4444;
                --warning: #ffbb33;
            }

            * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }

            body {
                background: var(--bg);
                color: var(--text);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                overflow-x: hidden;
                background-image: 
                    radial-gradient(circle at 20% 20%, rgba(0, 242, 255, 0.05) 0%, transparent 40%),
                    radial-gradient(circle at 80% 80%, rgba(112, 0, 255, 0.05) 0%, transparent 40%);
            }

            .container { max-width: 1000px; width: 90%; margin: 40px auto; }

            header { text-align: center; margin-bottom: 60px; animation: fadeInDown 0.8s ease-out; }

            .logo-container { display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 10px; }

            .pulse { width: 12px; height: 12px; background: var(--accent); border-radius: 50%; box-shadow: 0 0 15px var(--accent); animation: pulse 2s infinite; }

            h1 { font-size: 3rem; font-weight: 800; letter-spacing: -2px; background: linear-gradient(to right, #fff, var(--primary)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

            p.subtitle { color: var(--text-dim); font-size: 1.1rem; font-weight: 300; }

            .controls { display: flex; justify-content: center; margin-bottom: 40px; }

            button {
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                color: white; border: none; padding: 16px 40px; font-size: 1.1rem; font-weight: 600; border-radius: 12px;
                cursor: pointer; transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                box-shadow: 0 10px 30px rgba(0, 242, 255, 0.3); display: flex; align-items: center; gap: 10px;
            }

            button:hover { transform: translateY(-5px) scale(1.02); box-shadow: 0 15px 40px rgba(0, 242, 255, 0.5); }
            button:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }

            .simulation-viewport {
                background: var(--card-bg); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 24px; padding: 30px; min-height: 400px; box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5); position: relative;
            }

            .viewport-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding-bottom: 15px; }

            .console-label { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--primary); text-transform: uppercase; letter-spacing: 2px; }

            #output-container { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px; transition: opacity 0.3s; }

            .step-card {
                background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px;
                padding: 20px; animation: slideUp 0.5s ease-out backwards;
            }

            .step-card h3 { font-size: 0.9rem; color: var(--primary); margin-bottom: 15px; display: flex; justify-content: space-between; }

            .stat-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.85rem; }
            .stat-label { color: var(--text-dim); }
            .stat-value { color: #fff; font-weight: 500; }
            .stat-reward { color: var(--accent); font-weight: 600; }

            .raw-output {
                margin-top: 30px; background: #000; padding: 20px; border-radius: 12px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #0f0;
                overflow-x: auto; border: 1px solid #1a1a1a; display: none;
            }

            @keyframes pulse { 0% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.5); opacity: 0.5; } 100% { transform: scale(1); opacity: 1; } }
            @keyframes fadeInDown { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
            @keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

            .empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 300px; color: var(--text-dim); text-align: center; }

            .loading-dots:after { content: ' .'; animation: dots 1.5s steps(5, end) infinite; }
            @keyframes dots { 0%, 20% { content: ' .'; } 40% { content: ' . .'; } 60% { content: ' . . .'; } 80%, 100% { content: ''; } }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <div class="logo-container">
                    <div class="pulse"></div>
                    <h1>Smart Hospital AI</h1>
                </div>
                <p class="subtitle">Reinforcement Learning Triage Optimization Environment</p>
            </header>

            <div class="controls">
                <button id="run-btn" onclick="runDemo()">
                    <span>▶️ Launch Simulation</span>
                </button>
            </div>

            <div class="simulation-viewport">
                <div class="viewport-header">
                    <div class="console-label">Simulation Metrics</div>
                    <div id="status-tag" style="font-size: 0.7rem; padding: 4px 8px; border-radius: 4px; background: rgba(255,255,255,0.05);">READY</div>
                </div>
                
                <div id="output-container">
                    <div class="empty-state">
                        <p>Awaiting simulation signal...</p>
                    </div>
                </div>

                <pre id="raw-log" class="raw-output"></pre>
            </div>
        </div>

        <script>
            async function runDemo() {
                const btn = document.getElementById('run-btn');
                const container = document.getElementById('output-container');
                const statusTag = document.getElementById('status-tag');
                const rawLog = document.getElementById('raw-log');

                btn.disabled = true;
                btn.innerHTML = '<span>⏳ Processing Sim<span class="loading-dots"></span></span>';
                statusTag.innerText = 'EXECUTING';
                statusTag.style.color = 'var(--primary)';
                container.style.opacity = '0.5';

                try {
                    const res = await fetch('/demo');
                    const data = await res.json();
                    
                    container.innerHTML = '';
                    container.style.opacity = '1';
                    
                    if (data.simulation) {
                        data.simulation.forEach((step, index) => {
                            const card = document.createElement('div');
                            card.className = 'step-card';
                            card.style.animationDelay = `${index * 0.1}s`;
                            
                            const info = step.info || {};
                            const isDeptCorrect = info.true_department && step.action.department && info.true_department.toLowerCase() === step.action.department.toLowerCase();
                            const isSerCorrect = info.true_seriousness !== undefined && Math.abs(info.true_seriousness - step.action.seriousness) === 0;

                            card.innerHTML = `
                                <h3>Step 0${index + 1} <span class="stat-reward">+${parseFloat(step.reward).toFixed(2)}</span></h3>
                                <div class="stat-row">
                                    <span class="stat-label">Action Dept</span>
                                    <span class="stat-value" style="color: ${isDeptCorrect ? 'var(--accent)' : 'var(--error)'}">${step.action.department}</span>
                                </div>
                                ${info.true_department ? `
                                <div class="stat-row">
                                    <span class="stat-label">True Dept</span>
                                    <span class="stat-value">${info.true_department}</span>
                                </div>` : ''}
                                <div class="stat-row" style="margin-top: 8px;">
                                    <span class="stat-label">Action Ser.</span>
                                    <span class="stat-value" style="color: ${isSerCorrect ? 'var(--accent)' : 'var(--warning)'}">${step.action.seriousness}/5</span>
                                </div>
                                ${info.true_seriousness !== undefined ? `
                                <div class="stat-row">
                                    <span class="stat-label">True Ser.</span>
                                    <span class="stat-value">${info.true_seriousness}/5</span>
                                </div>` : ''}
                                <div class="stat-row" style="margin-top:10px; border-top: 1px solid rgba(255,255,255,0.05); padding-top:10px;">
                                    <span class="stat-label">Result</span>
                                    <span class="stat-reward" style="color: ${step.reward > 0 ? 'var(--accent)' : 'var(--error)'}">${step.reward > 0 ? 'Optimized' : 'Sub-optimal'}</span>
                                </div>
                            `;
                            container.appendChild(card);
                        });
                    }

                    rawLog.innerText = JSON.stringify(data, null, 2);
                    rawLog.style.display = 'block';
                    statusTag.innerText = 'COMPLETED';
                    statusTag.style.color = 'var(--accent)';

                } catch (err) {
                    container.innerHTML = `<div class="empty-state" style="color: var(--error)">Error connecting to environment API</div>`;
                    console.error(err);
                    statusTag.innerText = 'ERROR';
                    statusTag.style.color = 'var(--error)';
                } finally {
                    btn.disabled = false;
                    btn.innerHTML = '<span>▶️ Run Simulation</span>';
                }
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
        state, reward, done, info = env.step(action)

        steps.append({
            "action": action,
            "reward": reward,
            "info": info
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