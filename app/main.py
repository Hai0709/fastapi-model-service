from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

from app.api_http import router as http_router
from app.api_ws import router as ws_router


APP_MODE = os.getenv("APP_MODE", "text2image")

app = FastAPI(title=f"FastAPI AI Service ({APP_MODE})")


templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "mode": APP_MODE}
    )

# 注册接口
app.include_router(http_router, prefix="/http", tags=["http"])
app.include_router(ws_router, prefix="/ws", tags=["ws"])


