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