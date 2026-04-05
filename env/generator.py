import random

def generate_patient():
    symptoms_list = [
        ("chest pain", "cardiology"),
        ("head injury", "neurology"),
        ("fever", "general"),
        ("shortness of breath", "cardiology"),
        ("dizziness", "general")
    ]

    symptom, department = random.choice(symptoms_list)

    patient = {
        "symptoms": symptom,
        "age": random.randint(20, 80),
        "heart_rate": random.randint(60, 130),
        "blood_pressure": random.randint(80, 140),
        "department": department
    }

    # 🧠 True priority logic (ground truth)
    score = 1

    if patient["heart_rate"] > 110:
        score += 2
    if patient["blood_pressure"] < 100:
        score += 2
    if "chest pain" in patient["symptoms"]:
        score += 1

    patient["true_priority"] = min(5, score)

    return patient