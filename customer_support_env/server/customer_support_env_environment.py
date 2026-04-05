# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Customer Support Env Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CustomerSupportAction, CustomerSupportObservation
except ImportError:
    from models import CustomerSupportAction, CustomerSupportObservation

from customer_support_env.task_env.tasks import easy_task, hard_task, medium_task

class CustomerSupportEnvironment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = CustomerSupportEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Customer Support Env environment ready!"
        >>>
        >>> obs = env.step(CustomerSupportAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the customer_support_env environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._current_obs = None
        self._current_task = None
        self.action_history = []
        self.tasks = {
            "easy": easy_task,
            "medium": medium_task,
            "hard": hard_task,
        }

    def reset(self, task: str = "easy") -> CustomerSupportObservation:
        """
        Reset the environment.

        Returns:
            CustomerSupportObservation with a ready message
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self.action_history = []

        if task not in self.tasks:
            raise ValueError(f"Unknown task: {task}. Choose easy, medium or hard.")

        self._current_task = self.tasks[task]
        self._current_obs = CustomerSupportObservation(
            **self._current_task.initial_state,
            reward=0.0,
            done=False,
        )

        return self._current_obs

    def step(self, action):  # type: ignore[override]
        """
        Execute a step in the environment using the active task evaluator.

        Args:
            action: CustomerSupportAction or raw action string

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self._current_task is None or self._current_obs is None:
            raise RuntimeError("Environment must be reset before calling step().")

        self._state.step_count += 1

        if isinstance(action, str):
            action_name = action
        else:
            action_name = action.action

        last_action = self.action_history[-1] if self.action_history else None

        self.action_history.append(action_name)

        base_score = self._current_task.evaluate(self.action_history)

        repeated_action = last_action == action_name

        # Recalculate reward fresh on every step from the current task progress.
        reward = base_score

        # Apply a fixed penalty for taking a step.
        reward -= 0.05

        # Discourage repeating the same action back-to-back.
        if repeated_action:
            reward -= 0.2

        # Give a smaller success bonus when the task is fully completed.
        if base_score == 1.0:
            reward += 0.5

        done = base_score == 1.0

        updated_sentiment = self._current_obs.sentiment
        updated_order_status = self._current_obs.order_status

        if action_name == "apologize":
            updated_sentiment = "calmer"
        elif action_name in {"refund", "offer_refund"} and reward > 0.0:
            updated_order_status = "refunded"

        self._current_obs = CustomerSupportObservation(
            user_query=self._current_obs.user_query,
            sentiment=updated_sentiment,
            issue_type=self._current_obs.issue_type,
            order_status=updated_order_status,
            attempts=self._current_obs.attempts + 1,
            reward=reward,
            done=done,
        )
        info = {
            "task_goal": self._current_task.goal,
            "actions_taken": list(self.action_history),
            "task_score": base_score,
            "repeated_action": repeated_action,
        }

        return self._current_obs, reward, done, info
    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
