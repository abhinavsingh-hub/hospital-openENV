# EASY → only department match
def easy_task_reward(patient, action):
    return 1 if action["department"] == patient["department"] else 0


# MEDIUM → department + seriousness
def medium_task_reward(patient, action):
    score = 0

    if action["department"] == patient["department"]:
        score += 0.5

    if action["seriousness"] == patient["true_seriousness"]:
        score += 0.5

    return score


# HARD → full logic
def hard_task_reward(patient, action):
    reward = 0

    # Department match
    if action["department"] == patient["department"]:
        reward += 0.5
    else:
        reward -= 0.3

    # Seriousness match
    if action["seriousness"] == patient["true_seriousness"]:
        reward += 0.5
    else:
        reward -= 0.2

    # 🔥 Critical penalty
    if patient["true_seriousness"] == 5 and action["seriousness"] < 3:
        reward -= 0.5

    return reward