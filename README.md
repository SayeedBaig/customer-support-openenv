# Customer Support RL Environment (OpenEnv)

## 📌 Overview & Motivation

This project implements a Reinforcement Learning (RL) environment for a customer support system using OpenEnv.

The goal is to simulate real-world customer service interactions where an intelligent agent (powered by an LLM) takes actions to resolve user issues efficiently.

The environment is designed to:
- Evaluate decision-making of an AI agent
- Provide step-based rewards for actions
- Encourage optimal resolution strategies
- Penalize inefficient or incorrect behavior

---

## 🎯 Observation Space

Each observation represents the current state of the customer interaction.

Fields include:

- `user_query` (str): Customer's request or complaint  
- `sentiment` (str): Emotional state (e.g., frustrated, neutral, calm)  
- `issue_type` (str): Type of issue (e.g., refund_request, delivery_delay)  
- `order_status` (str): Current status of order (e.g., shipped, delivered)  
- `attempts` (int): Number of steps taken so far  
- `reward` (float): Current reward value  
- `done` (bool): Whether the episode is completed  

---

## ⚙️ Action Space

The agent selects actions as strings representing customer support operations:

- `apologize`
- `provide_status_update`
- `track_order`
- `refund`
- `acknowledge_issues`
- `close_case`
- `escalate_to_human`
- `ask_info`

These actions influence the environment state and reward.

---

## 🧩 Task Descriptions

We define three difficulty levels:

### 🟢 Easy Task
- Scenario: User requests a refund
- Expected action: `refund`
- Goal: Immediate resolution

---

### 🟡 Medium Task
- Scenario: Delivery delay
- Expected flow:
  - `apologize`
  - `provide_status_update`
  - (optional) `discount`
- Goal: Inform and satisfy user

---

### 🔴 Hard Task
- Scenario: Complex issue (e.g., double charge + delayed replacement)
- Expected flow:
  - Understand issue
  - Provide resolution
  - Close or escalate case
- Goal: Multi-step reasoning and resolution

---

## 🏆 Reward Design

The reward function is step-based and designed to guide the agent:

- ✅ Correct action: +0.25 to +0.5  
- 🎯 Final success bonus: +1.0  
- ❌ Wrong action: -0.2  
- 🔁 Repeated action penalty: -0.2  
- ⏳ Step penalty: -0.05  

This ensures efficient and correct behavior.

---

## 🚀 Setup & Usage

### 🔹 1. Install dependencies

```bash
pip install -r requirements.txt

2. Create .env file
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
HF_TOKEN=your_token_here

3. Run inference
python inference.py
🔹 4. Run with Docker
docker build -t cs-env .
docker run --env-file .env cs-env

📊 Baseline Performance

Example outputs:

Local Run:
Steps: 4
Total Reward: 0.35
Success: ✅
Docker Run:
Steps: 6
Total Reward: -0.05
Success: ✅

Note: Outputs vary due to stochastic LLM behavior.

🤖 LLM Integration

We use:

Model: Qwen/Qwen2.5-7B-Instruct
API: HuggingFace Router
Client: OpenAI-compatible API

The LLM selects actions based on the current observation.

📦 Deployment

The environment is containerized using Docker and deployed on HuggingFace Spaces.

FastAPI endpoints:
/reset
/step
✅ Summary
RL-based customer support simulation
LLM-driven decision making
Step-based reward optimization
Dockerized and deployed

