from env.hospital_env import HospitalEnv
import random
from collections import defaultdict
import pickle

# 🎯 All possible actions
DEPARTMENTS = [
    "cardiology",
    "neurology",
    "orthopedics",
    "general",
    "pulmonology",
    "emergency"
]

SERIOUSNESS = [1, 2, 3, 4, 5]

ACTIONS = [(d, s) for d in DEPARTMENTS for s in SERIOUSNESS]


# 🧠 Q-Table
Q = defaultdict(lambda: [0] * len(ACTIONS))


# 🔑 Improved state representation
def state_to_key(state):
    symptoms = tuple(sorted(state["symptoms"]))

    # 🔥 risk-based abstraction
    high_hr = state["heart_rate"] > 0.6
    low_bp = state["blood_pressure"] < 0.5
    elderly = state["age"] > 0.6

    return (
        symptoms,
        high_hr,
        low_bp,
        elderly
    )


# 🎯 epsilon-greedy action selection
def choose_action(state, epsilon):
    key = state_to_key(state)

    if random.random() < epsilon:
        action_idx = random.randint(0, len(ACTIONS) - 1)
    else:
        q_values = Q[key]
        action_idx = q_values.index(max(q_values))

    return ACTIONS[action_idx], action_idx


# 🔁 TRAINING LOOP (FIXED ACCURACY)
def train(env, episodes=500):
    # save_q_table()

    alpha = 0.2
    gamma = 0.95

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.99

    for ep in range(episodes):
        state = env.reset()
        done = False

        total_reward = 0

        # ✅ accuracy tracking
        correct_department = 0
        correct_seriousness = 0
        total = 0
        total_score = 0

        while not done:
            key = state_to_key(state)

            (dept, ser), action_idx = choose_action(state, epsilon)

            action_dict = {
                "department": dept,
                "seriousness": ser
            }

            next_state, reward, done, info = env.step(action_dict)

            total += 1
            total_reward += reward

            # 🔥 stronger learning from mistakes
            if reward < 0:
                reward *= 2

            # ✅ correct evaluation (NOT using env accuracy)
            dept_correct = action_dict["department"] == info["true_department"]
            ser_correct = action_dict["seriousness"] == info["true_seriousness"]

            if dept_correct:
                correct_department += 1

            if ser_correct:
                correct_seriousness += 1

            step_score = (dept_correct + ser_correct) / 2
            total_score += step_score

            # 🧠 Q-learning update
            next_key = state_to_key(next_state) if next_state else None
            max_future = max(Q[next_key]) if next_key else 0

            Q[key][action_idx] += alpha * (
                reward + gamma * max_future - Q[key][action_idx]
            )

            state = next_state

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(
            f"Episode {ep+1} | Reward: {total_reward:.2f} | "
            f"Dept Acc: {correct_department/total:.2f} | "
            f"Ser Acc: {correct_seriousness/total:.2f} | "
            f"Overall: {total_score/total:.2f} | "
            f"Epsilon: {epsilon:.2f}"
        )


# 🧪 TEST FUNCTION (FIXED OUTPUT)
def test(env):
    state = env.reset()
    done = False
    step_num = 1

    correct_department = 0
    correct_seriousness = 0
    total = 0
    total_score = 0

    while not done:
        key = state_to_key(state)
        q_values = Q[key]
        best_idx = q_values.index(max(q_values))

        dept, ser = ACTIONS[best_idx]

        action = {
            "department": dept,
            "seriousness": ser
        }

        print(f"\n🩺 Step {step_num}")
        print("INPUT STATE:")
        print(f"Symptoms: {state['symptoms']}")
        print(f"Age: {state['age']}")
        print(f"Heart Rate: {state['heart_rate']}")
        print(f"Blood Pressure: {state['blood_pressure']}")

        print("\nAGENT ACTION:")
        print(f"Department: {action['department']}")
        print(f"Seriousness: {action['seriousness']}")

        state, reward, done, info = env.step(action)

        dept_correct = action["department"] == info["true_department"]
        ser_correct = action["seriousness"] == info["true_seriousness"]

        total += 1

        if dept_correct:
            correct_department += 1
        if ser_correct:
            correct_seriousness += 1

        step_score = (dept_correct + ser_correct) / 2
        total_score += step_score

        print(f"\nREWARD: {reward}")
        print(f"TRUE DEPARTMENT: {info['true_department']}")
        print(f"TRUE SERIOUSNESS: {info['true_seriousness']}")
        print(f"Dept Correct: {dept_correct}")
        print(f"Seriousness Correct: {ser_correct}")
        print(f"RUNNING ACCURACY: {total_score/total:.2f}")
        print("-" * 40)

        step_num += 1


# 🚀 MAIN
if __name__ == "__main__":

    env = HospitalEnv(task="hard", max_steps=20)

    print("🚀 Training started...\n")
    train(env, episodes=5000)

    print("\n🧪 Testing trained agent...\n")
    test(env)

    # ==============================
# 🤖 RL AGENT (for inference/UI)
# ==============================
def rl_agent(state):
    key = state_to_key(state)

    # fallback if unseen state
    if key not in Q:
        from inference import get_action
        return get_action(state)

    q_values = Q[key]
    best_idx = q_values.index(max(q_values))

    dept, ser = ACTIONS[best_idx]

    return {
        "department": dept,
        "seriousness": ser
    }

    # ==============================
# 💾 SAVE / LOAD Q TABLE
# ==============================

def save_q_table(filename="q_table.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(dict(Q), f)

def load_q_table(filename="q_table.pkl"):
    global Q
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            Q.update(data)
        print("✅ Q-table loaded")
    except FileNotFoundError:
        print("⚠️ No Q-table found. Train first.")