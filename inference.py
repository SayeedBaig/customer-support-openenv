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

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

VALID_ACTIONS = [
    "apologize",
    "refund",
    "provide_status_update",
    "escalate_to_human",
    "give_discount",
    "track_order",
    "acknowledge_issues",
    "close_case",
]

# Step 3 - Action generation
def get_action_from_llm(obs, action_history=[]) -> str:
    prompt = f"""You are a customer support agent. 
Current situation:
- Customer query: {obs.user_query}
- Customer sentiment: {obs.sentiment}
- Issue type: {obs.issue_type}
- Order status: {obs.order_status}
- Attempts so far: {obs.attempts}

Actions already taken: {action_history if action_history else "None"}

Choose ONE action from this list:
{", ".join(VALID_ACTIONS)}

Reply with ONLY the action name. Nothing else.
Important: Do NOT repeat an action already taken.
Reply with ONLY the action name. Nothing else.
"""

    response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "system", 
            "content": """You are a customer support AI agent. 
You must reply with ONLY one word from this exact list:
apologize, refund, provide_status_update, escalate_to_human, give_discount, track_order, acknowledge_issues, close_case

No explanation. No sentence. Just one word."""
        },
        {"role": "user", "content": prompt}
    ],
    max_tokens=10,
)
    raw_output = response.choices[0].message.content.strip().lower()
    print(f"LLM raw output: {raw_output}")  # debug
    return parse_action(raw_output)


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
obs = env.reset(task="medium")
total_reward = 0.0
reward_history = []

for step in range(10):
    
    action = get_action_from_llm(obs, env.action_history)

    obs, reward, done, info = env.step(action)
    total_reward += reward
    reward_history.append(reward)
    print(f"Step {step + 1} | action={action} | reward={reward} | done={done}")

    if done:
        break

total_reward = sum(reward_history)
normalized_score = max(0.0, min(1.0, total_reward / 10))

print(f"Total reward: {total_reward}")
print(f"Reward history: {reward_history}")
print(f"Normalized Score: {normalized_score}")
print("Episode finished")