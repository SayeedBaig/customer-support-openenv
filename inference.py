import requests
import os
from customer_support_env.server.customer_support_env_environment import CustomerSupportEnvironment

API_URL = "https://router.huggingface.co/hf-inference/v1"
MODEL_NAME=os.getenv('MODEL_NAME')
headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
    "Content-Type": "application/json"
}

VALID_ACTIONS = ["respond_to_user", "check_order_status", "issue_refund"]

def extract_text(obs):
    try:
        return obs.text.lower()
    except:
        return str(obs).lower()
def get_action_from_llm(obs, history):
    obs_text = extract_text(obs)

    prompt = f"""
You are a customer support agent.

Choose the BEST action from:
{VALID_ACTIONS}

ONLY return the action name.

Customer:
{obs_text}

History:
{history}

Answer:
"""

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": 20}
            },
            timeout=15
        )

        if response.status_code != 200:
            print("HF ERROR:", response.text)
            return smart_fallback(obs_text)

        try:
            data = response.json()
        except:
            print("Invalid JSON:", response.text)
            return smart_fallback(obs_text)

        print("HF RAW:", data)

        if isinstance(data, dict) and "error" in data:
            print("HF API ERROR:", data["error"])
            return smart_fallback(obs_text)

        if isinstance(data, list) and "generated_text" in data[0]:
            output = data[0]["generated_text"].lower()
            return map_to_valid_action(output, obs_text)

        return smart_fallback(obs_text)
    except Exception as e:
        print("LLM Error:", e)
        return smart_fallback(obs_text)
def map_to_valid_action(output, obs):
    # LLM output mapping
    if "refund" in output:
        return "issue_refund"
    elif "delay" in output or "late" in output or "status" in output:
        return "check_order_status"
    elif "respond" in output or "help" in output:
        return "respond_to_user"

    # fallback to rule-based
    return smart_fallback(obs)
def smart_fallback(obs):
    obs = extract_text(obs)

    if "refund" in obs:
        return "issue_refund"
    elif "late" in obs or "delay" in obs:
        return "check_order_status"
    else:
        return "respond_to_user"
    
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