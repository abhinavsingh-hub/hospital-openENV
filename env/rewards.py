# # 🟢 EASY → department only
# def grade_easy(patient, action):
#     return 1.0 if action["department"] == patient.department else 0.0


# # 🟡 MEDIUM → department + priority
# def grade_medium(patient, action):
#     if (
#         action["department"] == patient.department
#         and action["priority"] == patient["true_priority"]
#     ):
#         return 1.0
#     return 0.0


# # 🔴 HARD → same but stricter (can extend later)
# def grade_hard(patient, action):
#     score = 0.0

#     if action["department"] == patient.department:
#         score += 0.3

#     if action["priority"] == patient["true_priority"]:
#         score += 0.5

#     # penalty for critical miss
#     if patient["true_priority"] >= 4 and action["priority"] <= 2:
#         score -= 0.5

#     return max(0.0, min(1.0, score))
def compute_reward(patient, action):
    reward = 0.0

    true_ser = patient.true_seriousness
    pred_ser = action["seriousness"]

    true_dep = patient.department
    pred_dep = action["department"]

    # 🎯 1. Seriousness reward (with gradient)
    diff = abs(true_ser - pred_ser)

    if diff == 0:
        reward += 1.0
    elif diff == 1:
        reward += 0.6
    elif diff == 2:
        reward += 0.2
    else:
        reward -= 0.5   # far off

    # 🏥 2. Department reward (stronger signal)
    if pred_dep == true_dep:
        reward += 1.0
    else:
        reward -= 0.5

    # 🚨 3. Critical mistake penalty (VERY IMPORTANT)
    if true_ser >= 4 and pred_ser <= 2:
        reward -= 1.0

    # 🧠 4. Bonus for perfect prediction
    if pred_ser == true_ser and pred_dep == true_dep:
        reward += 0.5

    return reward