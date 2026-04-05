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

from ..task_env.tasks import easy_task, medium_task, hard_task

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
        self._actions_taken = []

    def reset(self, task="easy") -> CustomerSupportObservation:
        """
        Reset the environment.

        Returns:
            CustomerSupportObservation with a ready message
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self._actions_taken = []

        if task == "easy":
            self._current_task = easy_task
        elif task == "medium":
            self._current_task = medium_task
        elif task == "hard":
            self._current_task = hard_task
        else:
            raise ValueError(f"Unknown task: {task}. Choose easy, medium or hard.")
            self._current_obs = CustomerSupportObservation(
                **self._current_task.initial_state,
                reward=0.0,
                done=False
            )
            
        return self._current_obs

    def step(self, action: CustomerSupportAction) -> CustomerSupportObservation:
        self._state.step_count += 1

        action_type = action.action  # action comes as string

        # increment attempts
        self._current_obs.attempts += 1

        reward = 0.0
        done = False

        # -------- ACTION LOGIC --------
        if action_type == "apologize":
            self._current_obs.sentiment = "calmer"
            reward = 1.0

        elif action_type == "refund":
            if self._current_obs.issue_type == "delayed_order":
                self._current_obs.order_status = "refunded"
                reward = 5.0
                done = True
            else:
                reward = -1.0

        elif action_type == "track_order":
            reward = 2.0
            self._current_obs.sentiment = "neutral"

        elif action_type == "ignore":
            self._current_obs.sentiment = "angry"
            reward = -2.0

        else:
            reward = -1.0

        # -------- DONE CONDITION --------
        if self._current_obs.attempts >= 3:
            done = True

        # -------- RETURN UPDATED OBS --------
        return CustomerSupportObservation(
            user_query=self._current_obs.user_query,
            sentiment=self._current_obs.sentiment,
            issue_type=self._current_obs.issue_type,
            order_status=self._current_obs.order_status,
            attempts=self._current_obs.attempts,
            reward=reward,
            done=done,
        )
    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
