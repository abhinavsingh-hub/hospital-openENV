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

    # Department reward 
    if pred_dep == true_dep:
        reward += 1.0
    else:
        reward -= 0.5

    # Critical mistake penalty
    if true_ser >= 4 and pred_ser <= 2:
        reward -= 1.0

    # Bonus for perfect prediction
    if pred_ser == true_ser and pred_dep == true_dep:
        reward += 0.5

    return reward