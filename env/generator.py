import random
from env.models import Patient


SYMPTOMS = [
    "chest pain",
    "head injury",
    "fracture",
    "shortness of breath",
    "fever",
    "unconscious",
    "severe bleeding",
    "palpitations",
    "dizziness",
    "cough"
]

#  DEPARTMENT DECISION
def get_department(symptoms):

    if isinstance(symptoms, str):
        symptoms = [symptoms]

    symptoms = [s.lower() for s in symptoms]

    # 🔥 CRITICAL PRIORITY
    if any(s in ["unconscious", "severe bleeding"] for s in symptoms):
        return "emergency"

    if "chest pain" in symptoms and "shortness of breath" in symptoms:
        return "cardiology"

    dept_scores = {
        "cardiology": 0,
        "pulmonology": 0,
        "neurology": 0,
        "orthopedics": 0,
        "emergency": 0,
        "general": 0
    }

    for s in symptoms:
        if "chest pain" in s:
            dept_scores["cardiology"] += 3

        if "palpitations" in s:
            dept_scores["cardiology"] += 2

        if "shortness of breath" in s or "cough" in s:
            dept_scores["pulmonology"] += 3

        if "head injury" in s or "dizziness" in s:
            dept_scores["neurology"] += 3

        if "fracture" in s:
            dept_scores["orthopedics"] += 3

        if "bleeding" in s or "trauma" in s:
            dept_scores["emergency"] += 2

        if "fever" in s:
            dept_scores["general"] += 2

    dept_scores["general"] += 1

    return max(dept_scores, key=dept_scores.get)


# PATIENT GENERATION
def generate_patient(task="easy"):

    #Symptoms based on difficulty
    if task == "easy":
        symptoms = [random.choice(SYMPTOMS)]

    elif task == "medium":
        symptoms = random.sample(SYMPTOMS, 2)

    elif task == "hard":
        symptoms = random.sample(SYMPTOMS, random.randint(2, 4))

    else:
        raise ValueError("Invalid task level")

    #reduce emergency bias
    if random.random() < 0.1:
        symptoms = ["unconscious"]

    #Department
    department = get_department(symptoms)

    #ifficulty-based vitals
    if task == "easy":
        age = random.randint(20, 50)
        heart_rate = random.randint(60, 100)
        blood_pressure = random.randint(100, 140)

    elif task == "medium":
        age = random.randint(20, 70)
        heart_rate = random.randint(60, 120)
        blood_pressure = random.randint(90, 140)

    else:  # hard
        age = random.randint(10, 90)
        heart_rate = random.randint(60, 150)
        blood_pressure = random.randint(80, 160)

    #SERIOUSNESS SCORING
    score = 1

    if heart_rate > 120:
        score += 2
    elif heart_rate > 100:
        score += 1

    if blood_pressure < 90:
        score += 2
    elif blood_pressure < 100:
        score += 1

    for symptom in symptoms:
        if symptom in ["unconscious", "severe bleeding"]:
            score += 3
        elif symptom in ["chest pain", "shortness of breath"]:
            score += 2
        elif symptom in ["head injury"]:
            score += 2
        elif symptom in ["fracture"]:
            score += 1

    if age > 65:
        score += 1

    seriousness = min(5, score)

    if department == "emergency":
        seriousness = max(seriousness, 4)

    #RETURN AS MODEL
    return Patient(
        symptoms=symptoms,
        age=age,
        heart_rate=heart_rate,
        blood_pressure=blood_pressure,
        department=department,
        true_seriousness=seriousness
    )