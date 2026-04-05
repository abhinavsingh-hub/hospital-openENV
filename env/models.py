from pydantic import BaseModel
from typing import Literal

class Patient(BaseModel):
    symptoms: str
    age: int
    heart_rate: int
    blood_pressure: int


class Action(BaseModel):
    priority: int
    department: Literal[
        "cardiology",
        "neurology",
        "orthopedics",
        "pulmonology",
        "general",
        "emergency"
    ]