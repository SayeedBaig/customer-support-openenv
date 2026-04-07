from fastapi import FastAPI
import gradio as gr

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required") from e

from models import CustomerSupportAction, CustomerSupportObservation
from server.customer_support_env_environment import CustomerSupportEnvironment


# ----------- Create backend app -----------
backend_app = create_app(
    CustomerSupportEnvironment,
    CustomerSupportAction,
    CustomerSupportObservation,
    env_name="customer_support_env",
    max_concurrent_envs=1,
)

# ----------- Create main FastAPI -----------
app = FastAPI()

# Mount backend at /api
app.mount("/api", backend_app)

# ----------- Gradio UI -----------
def demo_response(text):
    return f"Customer Support Bot Response: {text}"

demo = gr.Interface(
    fn=demo_response,
    inputs=gr.Textbox(placeholder="Ask your question..."),
    outputs="text",
    title="Customer Support AI Assistant"
)

# Mount UI at root
app = gr.mount_gradio_app(app, demo, path="/")


# ----------- Health check -----------
@app.get("/health")
def health():
    return {"status": "running"}