# 🟢 EASY → department only (with penalty)
def easy_task_reward(patient, action):
    if action["department"] == patient.department:
        return 1.0
    else:
        return -0.5


# 🟡 MEDIUM → department + seriousness (graded)
def medium_task_reward(patient, action):
    reward = 0.0

    # ✅ Department (primary signal)
    if action["department"] == patient.department:
        reward += 1.0
    else:
        reward -= 0.5

    # ✅ Seriousness (graded learning)
    true = patient.true_seriousness
    pred = action["seriousness"]

    diff = abs(true - pred)

    if diff == 0:
        reward += 1.0
    elif diff == 1:
        reward += 0.5
    elif diff == 2:
        reward += 0.2
    else:
        reward -= 0.5

    return reward


# 🔴 HARD → realistic + safety-aware reward
def hard_task_reward(patient, action):
    reward = 0.0

    true = patient.true_seriousness
    pred = action["seriousness"]

   
    # Department (high weight)
   
    if action["department"] == patient.department:
        reward += 1.5
    else:
        reward -= 1.0

    # =========================
    # 🎯 Seriousness (graded)
    # =========================
    diff = abs(true - pred)

    if diff == 0:
        reward += 2.0
    elif diff == 1:
        reward += 1.2
    elif diff == 2:
        reward += 0.5
    else:
        reward -= 1.0

    # =========================
    # 🔥 CRITICAL SAFETY LOGIC
    # =========================
    # Missing severe case
    if true == 5 and pred <= 2:
        reward -= 3.0

    # Overreaction (less critical but still penalized)
    if true <= 2 and pred == 5:
        reward -= 0.5

    # =========================
    # 🧠 RISK-AWARE BONUS
    # =========================
    if patient.heart_rate > 120 and pred >= 4:
        reward += 0.5

    if patient.blood_pressure < 90 and pred >= 4:
        reward += 0.5

    if patient.age > 70 and pred >= 3:
        reward += 0.3

    # =========================
    # 🎯 PERFECT DECISION BONUS
    # =========================
    if action["department"] == patient.department and diff == 0:
        reward += 1.0

    return reward