---
title: Customer Support AI Environment
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
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
- **Reward:** 0.9 for correct, 0.5 for partial, 0.1 for wrong

### Medium — Delayed Order
- **Reward:** 0.9 for all correct, 0.7 for required only, 0.5 for partial

### Hard — Multiple Issues
- **Reward:** 0.9 for full resolution, 0.7 for partial, 0.2 for minimal

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
[START] task=easy env=customer_support_env model=Qwen/Qwen2.5-7B-Instruct
[STEP] step=1 action=refund reward=0.45 done=false error=null
[STEP] step=2 action=close_case reward=0.25 done=true error=null
[END] success=true steps=2 score=0.900 rewards=0.45,0.25

[START] task=medium env=customer_support_env model=Qwen/Qwen2.5-7B-Instruct
[STEP] step=1 action=apologize reward=0.25 done=false error=null
[STEP] step=2 action=provide_status_update reward=0.15 done=false error=null
[END] success=true steps=2 score=0.700 rewards=0.25,0.15

[START] task=hard env=customer_support_env model=Qwen/Qwen2.5-7B-Instruct
[STEP] step=1 action=apologize reward=0.25 done=false error=null
[STEP] step=2 action=acknowledge_issues reward=0.25 done=false error=null
[STEP] step=3 action=refund reward=0.45 done=false error=null
[STEP] step=4 action=close_case reward=0.25 done=true error=null
[END] success=true steps=4 score=0.900 rewards=0.25,0.25,0.45,0.25

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

## Baseline Scores

| Task | Steps | Score | Success |
|------|-------|-------|---------|
| Easy | 10 | 0.100 | ❌ |
| Medium | 10 | 0.500 | ✅ |
| Hard | 10 | 0.900 | ✅ |

Note: Scores vary due to stochastic LLM behavior. The hackathon proxy injects its own credentials so results may differ.

---

## Tech Stack

- Python 3.11
- FastAPI
- OpenEnv
- Pydantic
- OpenAI client
- Hugging Face Inference API
- Docker
