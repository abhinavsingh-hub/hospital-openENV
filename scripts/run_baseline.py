from env.hospital_env import HospitalEnv
import random
from collections import defaultdict

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
    return (
        tuple(sorted(state["symptoms"])),   # ✅ FIX HERE
        state["age"] // 10,
        state["heart_rate"] // 10,
        state["blood_pressure"] // 10
    )


# 🎯 Correct epsilon-greedy action selection
def choose_action(state, epsilon):
    key = state_to_key(state)

    if random.random() < epsilon:
        action_idx = random.randint(0, len(ACTIONS) - 1)
    else:
        q_values = Q[key]
        action_idx = q_values.index(max(q_values))

    return ACTIONS[action_idx], action_idx


# 🔁 TRAINING LOOP
def train(env, episodes=500):

    alpha = 0.1
    gamma = 0.95

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    for ep in range(episodes):
        state = env.reset()
        done = False

        total_reward = 0

        while not done:
            key = state_to_key(state)

            (dept, ser), action_idx = choose_action(state, epsilon)

            action_dict = {
                "department": dept,
                "seriousness": ser
            }

            next_state, reward, done, info = env.step(action_dict)

            total_reward += reward

            # 🔥 stronger learning from mistakes
            if reward < 0:
                reward *= 2

            next_key = state_to_key(next_state) if next_state else None
            max_future = max(Q[next_key]) if next_key else 0

            # 🧠 Q-learning update
            Q[key][action_idx] += alpha * (
                reward + gamma * max_future - Q[key][action_idx]
            )

            state = next_state

        # 📉 reduce randomness over time
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {ep+1} | Reward: {total_reward:.2f} | Accuracy: {info['accuracy']:.2f} | Epsilon: {epsilon:.2f}")


# 🧪 TEST TRAINED MODEL (with your desired output format)
def test(env):
    state = env.reset()
    done = False
    step_num = 1

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

        print(f"\nREWARD: {reward}")
        print(f"TRUE DEPARTMENT: {info['true_department']}")
        print(f"TRUE SERIOUSNESS: {info['true_seriousness']}")
        print(f"RUNNING ACCURACY: {info['accuracy']:.2f}")
        print("-" * 40)

        step_num += 1


# 🚀 MAIN
if __name__ == "__main__":

    env = HospitalEnv(task="hard", max_steps=20)

    print("🚀 Training started...\n")
    train(env, episodes=500)   # 🔥 important

    print("\n🧪 Testing trained agent...\n")
    test(env)