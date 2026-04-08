from fastapi import FastAPI
from env.hospital_env import HospitalEnv
import uvicorn

app = FastAPI()

env = HospitalEnv(task="easy", max_steps=1)


@app.get("/")
def home():
    return {
        "message": "Smart Hospital OpenEnv is running 🚀",
        "endpoint": "/reset"
    }


@app.post("/reset")
def reset():
    state = env.reset()
    return {"state": state}


# 🔥 REQUIRED for OpenEnv
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


# 🔥 REQUIRED ENTRYPOINT
if __name__ == "__main__":
    main()