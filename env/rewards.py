# # 🟢 EASY → department only
# def grade_easy(patient, action):
#     return 1.0 if action["department"] == patient["department"] else 0.0


# # 🟡 MEDIUM → department + priority
# def grade_medium(patient, action):
#     if (
#         action["department"] == patient["department"]
#         and action["priority"] == patient["true_priority"]
#     ):
#         return 1.0
#     return 0.0


# # 🔴 HARD → same but stricter (can extend later)
# def grade_hard(patient, action):
#     score = 0.0

#     if action["department"] == patient["department"]:
#         score += 0.3

#     if action["priority"] == patient["true_priority"]:
#         score += 0.5

#     # penalty for critical miss
#     if patient["true_priority"] >= 4 and action["priority"] <= 2:
#         score -= 0.5

#     return max(0.0, min(1.0, score))
def compute_reward(patient, action):
    reward = 0.0

    # ✅ correct priority
    if action["priority"] == patient["true_priority"]:
        reward += 0.5

    # ✅ correct department
    if action["department"] == patient["department"]:
        reward += 0.3

    # ❌ penalty for critical mistake
    if patient["true_priority"] >= 4 and action["priority"] <= 2:
        reward -= 0.5

    return reward