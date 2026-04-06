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

        self.correct = 0
        self.total = 0

    # 🔄 RESET ENVIRONMENT
    def reset(self):
        import random

        # ✅ IMPORTANT CHANGE: pass task → enables multi-symptom in hard mode
        self.queue = [generate_patient(self.task) for _ in range(self.max_steps)]
        random.shuffle(self.queue)

        self.current_step = 0
        self.correct = 0
        self.total = 0

        self.patient = self.queue.pop(0)

        return self.state()

    # 📊 CURRENT STATE
    def state(self):
        return {
            "symptoms": self.patient["symptoms"],  # now list in hard mode
            "age": self.patient["age"],
            "heart_rate": self.patient["heart_rate"],
            "blood_pressure": self.patient["blood_pressure"]
        }

    # 🎯 STEP FUNCTION (CORE LOGIC)
    def step(self, action_dict):
        action = Action(**action_dict)

        # 🧠 Compute reward
        reward = self._get_reward(self.patient, action.__dict__)

        # ✅ Accuracy tracking (department-based)
        if action_dict["department"] == self.patient["department"]:
            self.correct += 1
        self.total += 1

        self.current_step += 1

        # ⚠️ STORE CURRENT PATIENT BEFORE MOVING AHEAD
        current_patient = self.patient

        # ✅ CHECK IF EPISODE DONE
        done = (len(self.queue) == 0) or (self.current_step >= self.max_steps)

        # 🔄 GET NEXT STATE
        if not done:
            self.patient = self.queue.pop(0)
            next_state = self.state()
        else:
            next_state = None

        # 📦 INFO
        info = {
            "task": self.task,
            "true_seriousness": current_patient["true_seriousness"],
            "true_department": current_patient["department"],
            "agent_action": action_dict,
            "accuracy": self.correct / self.total if self.total > 0 else 0
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