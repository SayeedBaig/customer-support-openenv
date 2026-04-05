from customer_support_env.server.customer_support_env_environment import CustomerSupportEnvironment

env = CustomerSupportEnvironment()

print("---- EASY TASK ----")
obs = env.reset(task="easy")
print(obs)

obs, reward, done, _ = env.step("refund")
print("Reward:", reward, "Done:", done)


print("\n---- MEDIUM TASK ----")
obs = env.reset(task="medium")
print(obs)

for action in ["apologize", "apologize", "track_order", "refund"]:
    obs, reward, done, _ = env.step(action)
    print(f"Action: {action} | Reward: {reward} | Done: {done}")


print("\n---- HARD TASK ----")
obs = env.reset(task="hard")
print(obs)

obs, reward, done, _ = env.step("apologize")
print("Reward:", reward, "Done:", done)
