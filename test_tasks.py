from customer_support_env.server.customer_support_env_environment import CustomerSupportEnvironment
env = CustomerSupportEnvironment()

obs = env.reset(task="easy")
print("EASY: ", obs.user_query)
obs = env.reset(task="medium")
print("MEDIUM: ", obs.user_query)
obs = env.reset(task="hard")
print("HARD: ", obs.user_query)

from customer_support_env.task_env.tasks import medium_task

print("\nMedium Task Grader")
print("All correct: ", medium_task.evaluate(["apologize", "provide_status_update", "give_discount"]))
print("Required only:", medium_task.evaluate(["apologize", "provide_status_update"]))
print("Partial + optional:", medium_task.evaluate(["apologize", "give_discount"]))
print("Partial only:", medium_task.evaluate(["apologize"]))
print("Optional only:", medium_task.evaluate(["give_discount"]))
print("No actions:", medium_task.evaluate([]))
print("Wrong actions:", medium_task.evaluate(["track_order"]))

# ---------------- HARD TASK TEST (YOUR PART) ----------------
from customer_support_env.task_env.tasks import hard_task

print("\nHard Task Grader")

print("Perfect flow:",
      hard_task.evaluate(["apologize", "acknowledge_issues", "offer_refund", "close_case"]))

print("Without closing:",
      hard_task.evaluate(["apologize", "acknowledge_issues", "offer_refund"]))

print("Partial (no resolution):",
      hard_task.evaluate(["apologize", "acknowledge_issues"]))

print("Only apology:",
      hard_task.evaluate(["apologize"]))

print("Wrong actions:",
      hard_task.evaluate(["track_order"]))

print("No actions:",
      hard_task.evaluate([]))

print("\nTesting State Transitions (Medium Task)")
obs = env.reset(task="medium");
print("Initial sentiment: ", obs.sentiment)

obs, reward, done, info = env.step("apologize")
print("After apologize - sentiment: ", obs.sentiment, "| order_status: ", obs.order_status)

obs, reward, done, info = env.step("provide_status_update")
print("After update - sentiment: ", obs.sentiment, "| order_status: ", obs.order_status)

obs, reward, done, info = env.step("give_discount")
print("After discount - sentiment: ", obs.sentiment, "| order_status: ", obs.order_status)
print("Done: ", done, "| Reason: ", info.get("termination_reason", "unknown"))

print("\nTesting Max Steps")
obs = env.reset(task="easy")
for i in range(12):
    obs, reward, done, info = env.step("apologize")
    if done:
        print(f"Episode ended at step {i+1} | Reason: {info.get('termination_reason', 'unknown')}")
        break

print("\nTesting Repeated Wrong Actions")
obs = env.reset(task="easy")
for i in range(5):
    obs, reward, done, info = env.step("track_order")
    print(f"Step {i+1} | done: {done} | reason: {info.get('termination_reason', 'unknown')} | history: {info['actions_taken']}")
    if done:
        print(f"Episode ended at step {i+1} | Reason: {info.get('termination_reason', 'unknown')}")
        break
