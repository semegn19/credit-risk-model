from fastapi import FastAPI

from src.api.routes import router
from src.config.settings import settings

app = FastAPI(
    title="Credit Risk API",
    version="1.0.0",
)

app.include_router(router)