import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple
from contextlib import asynccontextmanager

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Загрузка переменных из .env
load_dotenv()

# Внимание: в .env LANGFLOW_URL должен быть http://localhost:7860 (БЕЗ /api/v1/run на конце)
LANGFLOW_URL = (os.getenv("LANGFLOW_URL") or "http://127.0.0.1:7860").rstrip("/")
LANGFLOW_FLOW_ID = os.getenv("LANGFLOW_FLOW_ID") or ""
LANGFLOW_API_KEY = os.getenv("LANGFLOW_API_KEY") or ""
LANGFLOW_INPUT_TYPE = os.getenv("LANGFLOW_INPUT_TYPE") or "chat"
LANGFLOW_OUTPUT_TYPE = os.getenv("LANGFLOW_OUTPUT_TYPE") or "chat"

missing = []
for k, v in [("LANGFLOW_URL", LANGFLOW_URL), ("LANGFLOW_FLOW_ID", LANGFLOW_FLOW_ID),
             ("LANGFLOW_API_KEY", LANGFLOW_API_KEY)]:
    if not v:
        missing.append(k)
if missing:
    raise RuntimeError(f"Отсутствуют обязательные переменные окружения: {missing}")

# Настройка HTTP-клиента
timeout = httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=10.0)
client = httpx.AsyncClient(timeout=timeout, trust_env=False)


# Современный метод управления жизненным циклом (заменяет @app.on_event("shutdown"))
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Этот код выполняется при запуске сервера
    yield
    # Этот код выполняется при остановке сервера
    await client.aclose()


# Создаем приложение с lifespan
app = FastAPI(lifespan=lifespan)

allow_origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _extract_text_from_langflow(resp_json: Dict[str, Any]) -> Optional[str]:
    try:
        text = resp_json["outputs"][0]["outputs"][0]["results"]["message"]["text"]
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            return str(data.get("result", text))
        return text
    except (KeyError, IndexError, TypeError):
        pass
    try:
        return str(resp_json.get("result"))
    except Exception:
        return None


def _make_auth_headers() -> List[Dict[str, str]]:
    return [
        {"Authorization": f"Bearer {LANGFLOW_API_KEY}"},
        {"x-api-key": LANGFLOW_API_KEY},
    ]


async def _run_langflow(input_value: str, session_id: str) -> Tuple[Dict[str, Any], str]:
    url = f"{LANGFLOW_URL}/api/v1/run/{LANGFLOW_FLOW_ID}"
    payload = {
        "input_value": input_value,
        "input_type": LANGFLOW_INPUT_TYPE,
        "output_type": LANGFLOW_OUTPUT_TYPE,
        "session_id": session_id,
        "tweaks": None,
    }
    last_status = None
    last_text = None

    for auth in _make_auth_headers():
        headers = {"Content-Type": "application/json", **auth}
        try:
            r = await client.post(url, json=payload, headers=headers)
            last_status = r.status_code
            last_text = r.text
            if r.status_code in (401, 403):
                continue
            r.raise_for_status()
            return r.json(), ("bearer" if "Authorization" in auth else "x-api-key")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text) from e
        except httpx.ReadTimeout as e:
            raise HTTPException(status_code=502, detail=f"ReadTimeout: {repr(e)}") from e
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"RequestError: {type(e).__name__} {repr(e)}") from e

    raise HTTPException(status_code=502,
                        detail=f"Langflow auth failed. Last status={last_status}. Response={last_text}")


class MultiplyRequest(BaseModel):
    numbers: List[float] = Field(min_length=2)
    session_id: Optional[str] = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/multiply")
async def multiply(req: MultiplyRequest) -> Dict[str, Any]:
    input_value = " * ".join(str(x) for x in req.numbers)
    resp_json, auth_used = await _run_langflow(
        input_value=input_value,
        session_id=req.session_id or "multiply-session",
    )

    return {
        "input": input_value,
        "auth_used": auth_used,
        "result_text": _extract_text_from_langflow(resp_json),
        # Если нужен весь огромный "сырой" ответ, раскомментируйте строку ниже:
        # "raw": resp_json
    }


print("=== ЗАРЕГИСТРИРОВАННЫЕ МАРШРУТЫ ===")
for route in app.routes:
    if hasattr(route, 'methods'):
        print(f"{list(route.methods)} {route.path}")

if __name__ == "__main__":
    import uvicorn
    # Railway автоматически передает порт через переменную PORT
    port = int(os.getenv("PORT", 8000))
    filename = os.path.basename(__file__).replace(".py", "")
    print(f"Запуск сервера на 0.0.0.0:{port}")
    # Важно: host="0.0.0.0", чтобы сервер был доступен извне!
    uvicorn.run(f"{filename}:app", host="0.0.0.0", port=port)