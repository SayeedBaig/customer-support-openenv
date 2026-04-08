import os
from openai import OpenAI
from server.customer_support_env_environment import (
    CustomerSupportEnvironment,
)

API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME")
API_KEY = os.environ.get("API_KEY")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY if API_KEY else "dummy-key",
)

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

FALLBACK_ACTION = "provide_status_update"


def get_action_from_llm(obs, action_history=None) -> str:
    action_history = list(action_history or [])
    model = MODEL_NAME or "Qwen/Qwen2.5-7B-Instruct"
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

Important: Do NOT repeat an action already taken.
Reply with ONLY the action name. Nothing else."""

    try:
        response = client.chat.completions.create(
            model=model,
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
        return parse_action(raw_output)
    except Exception as exc:
        print(f"[ERROR] LLM request failed: {exc}. Falling back to rule-based action.")
        return get_rule_based_action(obs, action_history)


def get_rule_based_action(obs, action_history=None) -> str:
    action_history = list(action_history or [])

    if obs.order_status == "refunded" and "close_case" not in action_history:
        return "close_case"

    if obs.sentiment in {"angry", "frustrated"} and "apologize" not in action_history:
        return "apologize"

    if obs.order_status not in {"refunded", "escalated", "discount_applied", "update_provided"}:
        if "provide_status_update" not in action_history:
            return "provide_status_update"

    if obs.issue_type in {"delivery", "tracking"} and "track_order" not in action_history:
        return "track_order"

    if obs.issue_type in {"billing", "refund"} and "refund" not in action_history:
        return "refund"

    for action in VALID_ACTIONS:
        if action not in action_history:
            return action

    return FALLBACK_ACTION


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


# Main
try:
    env = CustomerSupportEnvironment()
    task = "medium"

    print(f"[START] task={task} env=CustomerSupportEnvironment model={MODEL_NAME}")

    obs = env.reset(task=task)
    total_reward = 0.0
    step = 0
    done = False

    while not done and step < 10:
        step += 1
        try:
            action = get_action_from_llm(obs, env.action_history)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()}")
        except Exception as exc:
            print(f"[ERROR] Step {step} failed: {exc}")
            action = "apologize"
            obs, reward, done, info = env.step(action)
            total_reward += reward  # ← add this

    print(f"[END] success={str(done).lower()} steps={step} rewards={total_reward:.2f}")

except Exception as exc:
    print(f"[FATAL] inference.py failed: {exc}")