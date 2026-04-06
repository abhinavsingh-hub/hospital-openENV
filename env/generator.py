import random

def generate_patient():
    symptoms_list = [
        ("chest pain", "cardiology"),
        ("head injury", "neurology"),
        ("fracture", "orthopedics"),
        ("shortness of breath", "pulmonology"),
        ("fever", "general"),
        ("unconscious", "emergency"),
        ("severe bleeding", "emergency"),
    ]

    symptom, department = random.choice(symptoms_list)

    patient = {
        "symptoms": symptom,
        "age": random.randint(20, 80),
        "heart_rate": random.randint(60, 140),
        "blood_pressure": random.randint(80, 160),
        "department": department
    }

    # 🧠 SERIOUSNESS SCORING SYSTEM (replaces priority)
    score = 1

    # 🔴 Vital-based scoring
    if patient["heart_rate"] > 120:
        score += 2
    elif patient["heart_rate"] > 100:
        score += 1

    if patient["blood_pressure"] < 90:
        score += 2
    elif patient["blood_pressure"] < 100:
        score += 1

    # 🧠 Symptom-based severity
    if symptom in ["unconscious", "severe bleeding"]:
        score += 3
    elif symptom in ["chest pain", "shortness of breath"]:
        score += 2
    elif symptom in ["head injury"]:
        score += 2
    elif symptom in ["fracture"]:
        score += 1
    elif symptom in ["fever"]:
        score += 0

    # 👶 Age-based risk (real-world touch 🔥)
    if patient["age"] > 65:
        score += 1

    # ✅ Final seriousness (1–5 scale)
    patient["true_seriousness"] = min(5, score)

    return patient