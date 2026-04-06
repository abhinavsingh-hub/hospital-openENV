import random
random.seed(42)

from env.hospital_env import HospitalEnv
from env.tasks import easy_task_reward, medium_task_reward, hard_task_reward

# 🧠 Smart Agent (rule-based)
def smart_agent(state):
    symptoms = state["symptoms"]
    hr = state["heart_rate"]
    bp = state["blood_pressure"]

    # 🏥 Department decision
    if "chest pain" in symptoms:
        department = "cardiology"
    elif "head" in symptoms:
        department = "neurology"
    elif "fracture" in symptoms:
        department = "orthopedics"
    elif "breath" in symptoms:
        department = "pulmonology"
    elif "unconscious" in symptoms or "bleeding" in symptoms:
        department = "emergency"
    else:
        department = "general"

    # ⚠️ Priority decision
    priority = 1

    if hr > 110:
        priority += 2
    if bp < 100:
        priority += 2
    if "chest pain" in symptoms:
        priority += 1

    priority = min(5, priority)

    return {
        "priority": priority,
        "department": department
    }

if __name__ == "__main__":
    # Initialize environment
    env = HospitalEnv(data=None)
    
    episodes = 20
    total_reward = 0
    easy_score = 0
    medium_score = 0
    hard_score = 0

    for _ in range(episodes):
        state = env.reset()
        action = smart_agent(state)
        _, reward, _, info = env.step(action)
        patient = env.patient  # ground truth

        # scoring
        easy_score += grade_easy(patient, action)
        medium_score += grade_medium(patient, action)
        hard_score += grade_hard(patient, action)

        print("State:", state)
        print("Action:", action)
        print("Reward:", reward)
        print("True:", {
            "priority": patient["true_priority"],
            "department": patient["department"]
        })
        print("------")

        total_reward += reward

    print("\n===== FINAL RESULTS =====")
    print("Average Reward:", total_reward / episodes)
    print("Easy Score:", easy_score / episodes)
    print("Medium Score:", medium_score / episodes)
    print("Hard Score:", hard_score / episodes)