from env.models import Action
from env.generator import generate_patient

# 👉 Import your task-specific reward functions
from env.tasks import (
    easy_task_reward,
    medium_task_reward,
    hard_task_reward
)


class HospitalEnv:

    def __init__(self, task="easy", max_steps=20):
        self.task = task
        self.max_steps = max_steps

        self.queue = []
        self.current_step = 0
        self.patient = None

    # 🔄 RESET ENVIRONMENT
    def reset(self):
        import random  # ✅ add here (or at top of file)

        self.queue = [generate_patient() for _ in range(self.max_steps)]

        random.shuffle(self.queue)  # ✅ SHUFFLE HERE

        self.current_step = 0

        self.patient = self.queue.pop(0)

        return self.state()

    # 📊 CURRENT STATE
    def state(self):
        return {
            "symptoms": self.patient["symptoms"],
            "age": self.patient["age"],
            "heart_rate": self.patient["heart_rate"],
            "blood_pressure": self.patient["blood_pressure"]
        }

    # 🎯 STEP FUNCTION (CORE LOGIC)
    def step(self, action_dict):
        # Convert dict → Action object
        action = Action(**action_dict)

        # 🧠 SELECT REWARD BASED ON TASK
        reward = self._get_reward(self.patient, action.__dict__)

        self.current_step += 1

        # ✅ CHECK IF EPISODE DONE
        done = (len(self.queue) == 0) or (self.current_step >= self.max_steps)

        # 🔄 GET NEXT STATE
        if not done:
            self.patient = self.queue.pop(0)
            next_state = self.state()
        else:
            next_state = None

        # 📦 INFO (VERY IMPORTANT FOR DEBUGGING & TRAINING)
        info = {
            "task": self.task,
            "true_priority": self.patient["true_priority"] if not done else None,
            "true_department": self.patient["department"] if not done else None,
            "agent_action": action_dict
        }

        return next_state, reward, done, info

    # 🧠 REWARD ROUTER
    def _get_reward(self, patient, action):
        if self.task == "easy":
            return easy_task_reward(patient, action)

        elif self.task == "medium":
            return medium_task_reward(patient, action)

        elif self.task == "hard":
            return hard_task_reward(patient, action)

        else:
            raise ValueError(f"Unknown task: {self.task}")