# Smart Hospital: RL Management Dashboard

A Reinforcement Learning (RL) powered simulation dashboard designed to optimize hospital resource allocation and patient triage. This project transitions complex RL decision logic into a visual, interactive experience for monitoring agent performance and hospital resource impact.

## 🚀 Overview

The system simulates a hospital environment where an RL agent (or rule-based baseline) is responsible for:
1. **Triage**: Assigning priority levels to incoming patients based on symptoms and vitals.
2. **Allocation**: Directing patients to the most appropriate departments (Cardiology, Neurology, General).
3. **Resource Management**: Monitoring the impact of these decisions on hospital beds and staff availability.

## 🛠️ Technologies Used

- **[Streamlit](https://streamlit.io/)**: For building the premium, glassmorphic dashboard UI.
- **[Python](https://www.python.org/)**: The engine behind the simulation and RL environment.
- **[Pydantic](https://docs.pydantic.dev/)**: Ensuring robust data validation for patients and agent actions.
- **[Pandas](https://pandas.pydata.org/)**: Handling time-series data for metrics and activity logging.

## 🏗️ Project Structure

- `env/`: Contains the core RL environment logic (`hospital_env.py`), data models (`models.py`), and reward functions (`rewards.py`).
- `scripts/`:
    - `dashboard.py`: The interactive Streamlit dashboard.
    - `run_baseline.py`: CLI-based baseline agent execution.
- `data/`: Sample patient datasets (`patients.json`).

## 🚦 How to Run

### 1. Install Dependencies
```bash
python3 -m pip install streamlit pydantic pandas
```

### 2. Launch the Dashboard
```bash
streamlit run scripts/dashboard.py
```

## 🧠 Dashboard Core Panels

- **🏥 Hospital State**: Live view of bed availability, staff allocation progress, and the current patient queue.
- **🤖 Agent Decisions**: Instant feedback on the agent's latest action, comparing its choice with the ground truth outcome.
- **📊 Real-time Metrics**: Tracking Total Reward, Avg Reward, and System Throughput.
- **🎮 Controls**: Step-by-step simulation or full episode automation.

---

## UI Preview (Draft)
![alt text](https://file%2B.vscode-resource.vscode-cdn.net/Users/abhinavsingh/Desktop/SS/Screenshot%202026-04-05%20at%205.25.47%E2%80%AFAM.png?version%3D1775347039430)



---
Built with ❤️ for AI research and hospital efficiency.
