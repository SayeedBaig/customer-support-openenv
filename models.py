from pydantic import BaseModel
from typing import Optional


class CustomerSupportAction(BaseModel):
    action: str


class CustomerSupportObservation(BaseModel):
    user_query: str
    sentiment: str
    issue_type: str
    order_status: str
    attempts: int
    reward: float
    done: bool
    info: Optional[dict] = None