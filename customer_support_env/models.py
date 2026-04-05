from pydantic import BaseModel

# Observation Model (required for your task)
class Observation(BaseModel):
    user_query: str
    sentiment: str
    issue_type: str
    order_status: str
    attempts: int
    reward: float
    done: bool

# Action Model (required for your task)
class Action(BaseModel):
    action: str


#  Aliases for OpenEnv compatibility
class CustomerSupportObservation(Observation):
    reward:float =0.0
    done:bool = False


class CustomerSupportAction(Action):
    pass
