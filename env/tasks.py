# EASY → department only
def easy_task_reward(patient, action):
    if action["department"] == patient.department:
        return 1
    else:
        return -0.5  # penalize wrong


# MEDIUM → department + seriousness
def medium_task_reward(patient, action):
    reward = 0

    # Department
    if action["department"] == patient.department:
        reward += 1
    else:
        reward -= 0.5

    # Seriousness 
    true = patient.true_seriousness
    pred = action["seriousness"]

    diff = abs(true - pred)

    if diff == 0:
        reward += 1
    elif diff == 1:
        reward += 0.5
    elif diff == 2:
        reward += 0.2
    else:
        reward -= 0.5

    return reward


# HARD → full intelligent reward
def hard_task_reward(patient, action):
    reward = 0

    true = patient.true_seriousness
    pred = action["seriousness"]

    # Department
    if action["department"] == patient.department:
        reward += 1
    else:
        reward -= 1  # stronger penalty

    # Seriousness (graded)
    diff = abs(true - pred)

    if diff == 0:
        reward += 1.5
    elif diff == 1:
        reward += 1
    elif diff == 2:
        reward += 0.3
    else:
        reward -= 1

    # Missed emergency
    if true == 5 and pred <= 2:
        reward -= 2

    # Overreaction penalty 
    if true <= 2 and pred == 5:
        reward -= 0.5

    # Risk-based bonus 
    if patient.heart_rate > 120 and pred >= 4:
        reward += 0.3

    if patient.blood_pressure < 90 and pred >= 4:
        reward += 0.3

    if patient.age > 70 and pred >= 3:
        reward += 0.2

    return reward