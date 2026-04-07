import os
from openai import OpenAI
from customer_support_env.server.customer_support_env_environment import (
    CustomerSupportEnvironment,
)

from dotenv import load_dotenv
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

import requests
import os

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
}

def get_action_from_llm(obs, history):
    prompt = f"{obs}\nHistory: {history}\nWhat action should be taken?"

    response = requests.post(API_URL, headers=headers, json={
        "inputs": prompt
    })

    return response.json()

def parse_action(raw_output: str) -> str:
    for action in VALID_ACTIONS:
        if action in raw_output:
            return action

    if any(word in raw_output for word in ["refund", "money back", "return"]):
        return "refund"
    elif any(word in raw_output for word in ["sorry", "apologize", "apology"]):
        return "apologize"
    elif any(word in raw_output for word in ["status", "update", "track", "where"]):
        return "provide_status_update"
    elif any(word in raw_output for word in ["escalate", "human", "manager"]):
        return "escalate_to_human"
    elif any(word in raw_output for word in ["discount", "compensation", "coupon"]):
        return "give_discount"
    elif any(word in raw_output for word in ["acknowledge", "understand"]):
        return "acknowledge_issues"
    elif any(word in raw_output for word in ["close", "resolve", "done"]):
        return "close_case"
    
    return "apologize"

env = CustomerSupportEnvironment()
task = "medium"

# -----------------------------
# 🔹 START LOG (STRICT)
# -----------------------------
print(f"[START] task={task} env=CustomerSupportEnvironment model={MODEL_NAME}")

obs = env.reset(task=task)

total_reward = 0.0
step = 0
done = False

# -----------------------------
# 🔹 LOOP
# -----------------------------
while not done and step < 10:
    step += 1

    action = get_action_from_llm(obs, env.action_history)

    obs, reward, done, info = env.step(action)

    total_reward += reward

    # -----------------------------
    # 🔹 STEP LOG (STRICT)
    # -----------------------------
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()}"
    )

# -----------------------------
# 🔹 END LOG (STRICT)
# -----------------------------
print(
    f"[END] success={str(done).lower()} steps={step} rewards={total_reward:.2f}"
)