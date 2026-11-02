"""FastAPI entrypoint for PlangoGraph."""

from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="PlangoGraph: Planogram Compliance via Spatial Layout Graphs",
    version="2.0.0",
    description="Automated retail shelf auditing combining YOLOv8, DINOv2/FAISS metric learning, and 2D spatial layout graphs.",
)
app.include_router(router)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "PlangoGraph", "version": "2.0.0"}
