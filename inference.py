import os
from typing import List, Optional
from openai import OpenAI
from server.customer_support_env_environment import CustomerSupportEnvironment
from task_env.tasks import easy_task, medium_task, hard_task

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"
BENCHMARK = "customer_support_env"

VALID_ACTIONS = [
    "apologize", "refund", "provide_status_update",
    "escalate_to_human", "give_discount", "track_order",
    "acknowledge_issues", "close_case",
]

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_action_from_llm(obs, action_history=None) -> str:
    action_history = list(action_history or [])
    api_base_url = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
    hf_token = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    model = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"

    print(f"[DEBUG] API_BASE_URL={api_base_url}", flush=True)
    print(f"[DEBUG] HF_TOKEN={'set' if hf_token else 'NOT SET'}", flush=True)
    print(f"[DEBUG] MODEL_NAME={model}", flush=True)

    client = OpenAI(base_url=api_base_url, api_key=hf_token)

    prompt = f"""You are a customer support agent.
Current situation:
- Customer query: {obs.user_query}
- Customer sentiment: {obs.sentiment}
- Issue type: {obs.issue_type}
- Order status: {obs.order_status}
- Attempts so far: {obs.attempts}
Actions already taken: {action_history if action_history else "None"}
Choose ONE action from this list: {", ".join(VALID_ACTIONS)}
Important: Do NOT repeat an action already taken.
Reply with ONLY the action name. Nothing else."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a customer support AI agent. Reply with ONLY one word from: apologize, refund, provide_status_update, escalate_to_human, give_discount, track_order, acknowledge_issues, close_case"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
        )
        raw_output = response.choices[0].message.content.strip().lower()
        return parse_action(raw_output)
    except Exception as exc:
        print(f"[DEBUG] LLM failed: {exc}", flush=True)
        return get_rule_based_action(obs, action_history)

def get_rule_based_action(obs, action_history=None) -> str:
    action_history = list(action_history or [])
    if obs.order_status == "refunded" and "close_case" not in action_history:
        return "close_case"
    if obs.sentiment in {"angry", "frustrated"} and "apologize" not in action_history:
        return "apologize"
    if "provide_status_update" not in action_history:
        return "provide_status_update"
    for action in VALID_ACTIONS:
        if action not in action_history:
            return action
    return "provide_status_update"

def parse_action(raw_output: str) -> str:
    for action in VALID_ACTIONS:
        if action in raw_output:
            return action
    return "apologize"

def run_task(task_name: str, task_obj) -> None:
    env = CustomerSupportEnvironment()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task=task_name)
        done = False

        for step in range(1, 11):
            if done:
                break
            steps_taken = step
            error = None

            try:
                action = get_action_from_llm(obs, env.action_history)
                obs, reward, done, info = env.step(action)
                rewards.append(reward)
            except Exception as exc:
                error = str(exc)
                action = "apologize"
                obs, reward, done, info = env.step(action)
                rewards.append(reward)

            log_step(step=step, action=action, reward=reward, done=done, error=error)

        # Calculate score using grader
        score = task_obj.evaluate(env.action_history)
        score = min(max(float(score), 0.01), 0.99)  # strictly between 0 and 1
        success = score >= 0.5

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

# Run all 3 tasks
run_task("easy", easy_task)
run_task("medium", medium_task)
run_task("hard", hard_task)