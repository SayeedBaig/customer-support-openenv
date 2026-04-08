---
title: Customer Support AI Environment
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Customer Support AI Environment (OpenEnv)

A reinforcement learning environment where an AI agent learns to resolve customer support issues through actions like refund, apologize, and escalate.

---

## Problem Motivation

Real-world companies like Amazon, Zomato, and Banks deal with thousands of customer complaints daily. Training an AI agent to handle these situations correctly is a challenging and valuable problem. This project simulates that environment using OpenEnv.

---

## How It Works

A customer comes with a problem. The AI agent observes the situation and takes actions. The environment evaluates those actions and gives a reward score between 0.0 and 1.0.

---

## State (What the Agent Sees)

| Field | Description |
|-------|-------------|
| user_query | What the customer said |
| sentiment | How angry or calm the customer is |
| issue_type | Type of problem (refund, delay, etc.) |
| order_status | Current status of the order |
| attempts | Number of actions taken so far |

---

## Actions (What the Agent Can Do)

- `apologize`
- `refund`
- `provide_status_update`
- `escalate_to_human`
- `give_discount`
- `track_order`
- `acknowledge_issues`
- `close_case`

---

## Tasks

### Easy — Refund Request
- **Scenario:** Customer wants a refund
- **Correct action:** refund
- **Reward:** 1.0 for correct, 0.5 for partial, 0.0 for wrong

### Medium — Delayed Order
- **Scenario:** Customer's order is late
- **Correct actions:** apologize + provide_status_update
- **Reward:** 1.0 for all correct, 0.8 for required only, 0.5 for partial

### Hard — Multiple Issues
- **Scenario:** Angry customer with billing and delivery issues
- **Correct actions:** apologize + acknowledge_issues + offer_refund + close_case
- **Reward:** 1.0 for full resolution, 0.8 for partial, 0.2 for minimal

---

## Reward System

| Situation | Reward |
|-----------|--------|
| Correct action | +0.3 to +1.0 |
| Final success bonus | +0.5 |
| Wrong action | 0.0 |
| Repeated action | -0.2 |
| Each step penalty | -0.05 |

---

## Done Conditions

Episode ends when:
- Task fully completed
- Max steps (10) reached
- Same wrong action repeated 3 times

---

## Setup

```bash
git clone https://github.com/your-repo/customer-support-env
cd customer-support-env
pip install -r requirements.txt
```

Create a `.env` file at root:
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
HF_TOKEN=your_hugging_face_token

---

## How to Run

```bash
export $(cat .env | xargs) && python inference.py
```

Expected output:
[START] task=medium env=CustomerSupportEnvironment model=Qwen/Qwen2.5-7B-Instruct
[STEP] step=1 action=apologize reward=0.25 done=false
[STEP] step=2 action=provide_status_update reward=0.15 done=false
[END] success=true steps=2 rewards=0.40

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /reset | POST | Reset environment |
| /step | POST | Execute an action |
| /state | GET | Get current state |

---

## Deployment

Live on Hugging Face Spaces: https://huggingface.co/spaces/RevanthKumar9/customer-support-env

API Docs: https://revanthkumar9-customer-support-env.hf.space/docs

---

## Tech Stack

- Python 3.11
- FastAPI
- OpenEnv
- Pydantic
- OpenAI client
- Hugging Face Inference API
- Docker