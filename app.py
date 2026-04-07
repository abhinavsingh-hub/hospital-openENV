from fastapi import FastAPI
from env.hospital_env import HospitalEnv

app = FastAPI()

env = HospitalEnv(task="easy", max_steps=1)

@app.post("/reset")
def reset():
    state = env.reset()
    return {"state": state}