"""Task definitions for the customer support environment."""


class CustomerSupportTask:
    def __init__(self, initial_state, goal: str):
        self.initial_state = initial_state
        self.goal = goal

    def evaluate(self, action) -> float:
        return 1.0 if action else 0.0


class EasyRefundTask(CustomerSupportTask):
    def evaluate(self, action_history) -> float:
        if isinstance(action_history, list):
            latest_action = action_history[-1] if action_history else None
        else:
            latest_action = action_history

        # Best outcome: the agent issues the refund.
        if latest_action == "refund":
            return 1.0

        # Partial credit: the agent acknowledges the issue politely.
        if latest_action == "apologize":
            return 0.5

        # No useful resolution was provided.
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
        required_actions = ["apologize", "track_order"]
        has_all_required_actions = all(
            action in actions_taken for action in required_actions
        )
        has_refund = "refund" in actions_taken
        has_any_required_action = any(
            action in actions_taken for action in required_actions
        )

        # Full credit: required actions are present and the refund was issued.
        if has_all_required_actions and has_refund:
            return 1.0

        # Strong partial progress: required actions are present without final resolution.
        if has_all_required_actions and not has_refund:
            return 0.8

        # Partial credit: at least one of the required actions was taken.
        if has_any_required_action:
            return 0.5

        # Irrelevant or incorrect actions do not receive credit.
        return 0.0


class HardTask:
    def __init__(self):
        self.goal = "Handle a multi-issue support case with a clear next step."
        self.initial_state = {
            "user_query": "I was charged twice and my replacement still has not arrived.",
            "sentiment": "angry",
            "issue_type": "billing_and_replacement",
            "order_status": "processing",
            "attempts": 2,
        }
        self.reset()

    def reset(self):
        self.state = self.initial_state.copy()
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

    def evaluate(self, actions: list) -> float:
        # Invalid input cannot receive credit.
        if not isinstance(actions, list):
            return 0.0

        actions_taken = set(actions)

        included_apology = "apologize" in actions_taken
        included_investigation = (
            "investigate" in actions_taken
            or "acknowledge_issues" in actions_taken
        )
        included_refund = (
            "refund" in actions_taken
            or "offer_refund" in actions_taken
        )
        included_case_closure = "close_case" in actions_taken

        # Full resolution: apologize, investigate, refund, and close the case.
        if (
            included_apology
            and included_investigation
            and included_refund
            and included_case_closure
        ):
            return 1.0

        # Strong progress: the issue was resolved, but the case was not closed.
        if (
            included_apology
            and included_investigation
            and included_refund
            and not included_case_closure
        ):
            return 0.8

        # Partial resolution: some meaningful resolution steps were taken.
        if included_investigation or (included_apology and included_refund):
            return 0.5

        # Minimal progress: only an apology was provided.
        if included_apology and not included_investigation and not included_refund:
            return 0.2

        # No meaningful support action was taken.
        return 0.0


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
hard_task = HardTask()
