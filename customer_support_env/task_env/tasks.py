"""Task definitions for the customer support environment."""


class CustomerSupportTask:
    def __init__(self, initial_state, goal: str):
        self.initial_state = initial_state
        self.goal = goal

    def evaluate(self, action) -> float:
        return 1.0 if action else 0.0


class EasyRefundTask(CustomerSupportTask):
    def evaluate(self, action) -> float:
        if action == "refund":
            return 1.0
        elif  action == "apologize":
            return 0.5
        else:
         return 0.0


easy_task = EasyRefundTask(
    initial_state={
        "user_query": "I want refund for my order",
        "sentiment": "neutral",
        "issue_type": "refund_request",
        "order_status": "delivered",
        "attempts": 0,
    },
    goal="Agent should issue a refund.",
)


medium_task = CustomerSupportTask(
    initial_state={
        "user_query": "My package is late and I need an update.",
        "sentiment": "frustrated",
        "issue_type": "delivery_delay",
        "order_status": "in_transit",
        "attempts": 1,
    },
    goal="Provide a helpful response for a delayed delivery issue.",
)


hard_task = CustomerSupportTask(
    initial_state={
        "user_query": "I was charged twice and my replacement still has not arrived.",
        "sentiment": "angry",
        "issue_type": "billing_and_replacement",
        "order_status": "processing",
        "attempts": 2,
    },
    goal="Handle a multi-issue support case with a clear next step.",
)

#test code
if __name__ == "__main__":
    task = easy_task

    print("Goal:", task.goal)

    print("Correct:", task.evaluate("refund"))
    print("Partial:", task.evaluate("apologize"))
    print("Wrong:", task.evaluate("track_order"))

class MediumDelayedOrderTask(CustomerSupportTask):
    def __init__(self):
        super().__init__(
            initial_state={
                "user_query": "My package is late and I need an update.",
                "sentiment": "frustrated",
                "issue_type": "delivery_delay",
                "order_status": "in_transit",
                "attempts": 1,
            },
            goal="Apologize, provide status update, optionally give discount."
        )
    
    def evaluate(self, actions_taken: list) -> float:
        required = ["apologize", "provide_status_update"]
        optional = ["give_discount"]
        required_done = all(a in actions_taken for a in required)
        optional_done = any(a in actions_taken for a in optional)
        any_required_done = any(a in actions_taken for a in required)
        if required_done and optional_done:
            return 1.0
        elif required_done:
            return 0.8
        elif any_required_done and not optional_done:
            return 0.5
        elif any_required_done and optional_done:
            return 0.6
        elif optional_done and not any_required_done:
            return 0.1
        elif not actions_taken:
            return -0.5
        else:
            return -0.5
class HardTask:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = {
            "user_message": "My order is late AND I received the wrong item. This is unacceptable!",
            "sentiment": "angry",
            "issues": ["delayed_order", "wrong_item"],

            # tracking flags
            "apologized": False,
            "acknowledged": False,
            "resolution_offered": False,
            "closed": False,

            "steps_taken": [],
            "resolved": False
        }
        return self.state

    def step(self, action: str):
        """
        action: string representing agent action
        """

        self.state["steps_taken"].append(action)

        if action == "apologize":
            self.state["apologized"] = True

        elif action == "acknowledge_issues":
            self.state["acknowledged"] = True

        elif action in ["offer_refund", "offer_replacement"]:
            self.state["resolution_offered"] = True

        elif action == "close_case":
            # Only resolve if all required steps done
            if (
                self.state["apologized"] and
                self.state["acknowledged"] and
                self.state["resolution_offered"]
            ):
                self.state["resolved"] = True

            self.state["closed"] = True

        return self.state


# ------------------- GRADER -------------------

def hard_task_grader(state: dict) -> float:
    """
    Deterministic grading function
    Returns score between 0.0 and 1.0
    """

    score = 0.0

    if state.get("apologized"):
        score += 0.2

    if state.get("acknowledged"):
        score += 0.3

    if state.get("resolution_offered"):
        score += 0.3

    if state.get("resolved"):
        score += 0.2

    return round(min(score, 1.0), 2)
medium_task = MediumDelayedOrderTask()
