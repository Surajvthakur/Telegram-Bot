from fastapi import FastAPI
from app.api.v1.chat import router as chat_router
from app.api.v1.agents import router as agents_router

app = FastAPI(title="Telegram Bot API")

# Include routers
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
app.include_router(agents_router, prefix="/api/v1", tags=["agents"])

@app.get("/")
async def root():
    return {"message": "Telegram Bot API is running"}