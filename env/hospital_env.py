import random
from env.models import Action
from env.rewards import compute_reward

class HospitalEnv:

    def __init__(self, data):
        self.data = data
        self.patient = None

    def reset(self):
        self.patient = random.choice(self.data)
        return self.state()

    def state(self):
        return {
            "symptoms": self.patient["symptoms"],
            "age": self.patient["age"],
            "heart_rate": self.patient["heart_rate"],
            "blood_pressure": self.patient["blood_pressure"]
        }

    def step(self, action_dict):
        action = Action(**action_dict)

        reward, done = compute_reward(self.patient, action)

        next_state = None if done else self.state()

        return next_state, reward, done, {
            "true_priority": self.patient["true_priority"],
            "true_department": self.patient["department"]
        }