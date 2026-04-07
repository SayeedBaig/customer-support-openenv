# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
FastAPI application for the Customer Support Env Environment.
"""

from fastapi import FastAPI
import gradio as gr

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies properly."
    ) from e

try:
    from ..models import CustomerSupportAction, CustomerSupportObservation
    from .customer_support_env_environment import CustomerSupportEnvironment
except ModuleNotFoundError:
    from models import CustomerSupportAction, CustomerSupportObservation
    from server.customer_support_env_environment import CustomerSupportEnvironment


# ---------------- EXISTING BACKEND ----------------
app = create_app(
    CustomerSupportEnvironment,
    CustomerSupportAction,
    CustomerSupportObservation,
    env_name="customer_support_env",
    max_concurrent_envs=1,
)


# ---------------- SIMPLE UI ----------------
def demo_response(user_input):
    return f"Customer Support Bot Response: {user_input}"


demo = gr.Interface(
    fn=demo_response,
    inputs=gr.Textbox(placeholder="Ask your customer support query here..."),
    outputs="text",
    title="Customer Support AI Assistant",
    description="Simple demo interface for the Customer Support Environment"
)

# Mount Gradio UI at root "/"
app = gr.mount_gradio_app(app, demo, path="/")


# ---------------- OPTIONAL ROOT (fallback) ----------------
@app.get("/health")
def health():
    return {"status": "running"}


# ---------------- RUN ----------------
def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()