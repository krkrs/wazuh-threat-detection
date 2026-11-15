
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field,  ValidationError

from wazuh_predictor import (
    WazuhPredictor,
    predict_wazuh_json,
)

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "wazuh_gru_multiclass.pt"
)

predictor = WazuhPredictor(MODEL_PATH)

app = FastAPI(
    title="Wazuh Threat Detection API",
    version="0.2.0",
)





class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alerts: list[dict[str, Any]] = Field(
    min_length=1,
    max_length=20000
)
    target_start: datetime
    history_complete: bool
    scenario: str = "incoming"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "GRU",
        "classes": predictor.phase_names,
        "sequence_length": predictor.seq_len,
        "window_seconds": predictor.window_seconds,
    }



MAX_REQUEST_BYTES = 20 * 1024 * 1024

@app.post("/predict")
async def predict(http_request: Request):
    # Wstępna kontrola deklarowanego rozmiaru.
    content_length = http_request.headers.get("content-length")

    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Nieprawidłowy Content-Length."
            )

        if declared_size < 0:
            raise HTTPException(
                status_code=400,
                detail="Nieprawidłowy Content-Length."
            )

        if declared_size > MAX_REQUEST_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Żądanie przekracza 20 MiB."
            )

    # Odczytujemy dane porcjami, pilnując limitu.
    body = bytearray()

    async for chunk in http_request.stream():
        if len(body) + len(chunk) > MAX_REQUEST_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Żądanie przekracza 20 MiB."
            )

        body.extend(chunk)

    # Walidujemy JSON i dotychczasowy schemat Pydantic.
    try:
        request = PredictionRequest.model_validate_json(
            bytes(body)
        )
    except ValidationError as exc:
        errors = [
            {
                "type": error["type"],
                "loc": error["loc"],
                "msg": error["msg"],
            }
            for error in exc.errors()
        ]
        raise HTTPException(
            status_code=422,
            detail=errors
        ) from exc

    # Od tego miejsca pozostaje dotychczasowa
    # walidacja i wywołanie modelu GRU.
    target = request.target_start


    if target.tzinfo is None or target.utcoffset() is None:
        raise HTTPException(
            status_code=422,
            detail="target_start musi zawierać strefę czasową.",
        )

    target = target.astimezone(timezone.utc)

    if (
        target.microsecond != 0
        or int(target.timestamp()) % predictor.window_seconds != 0
    ):
        raise HTTPException(
            status_code=422,
            detail="target_start musi być wyrównany do granicy 10 sekund.",
        )

    if not request.history_complete:
        raise HTTPException(
            status_code=422,
            detail="Klient nie potwierdził kompletności historii 60 sekund.",
        )

    history_start = target - timedelta(
        seconds=(predictor.seq_len - 1) * predictor.window_seconds
    )
    history_end = target + timedelta(
        seconds=predictor.window_seconds
    )

    for index, alert in enumerate(request.alerts):
        if not isinstance(alert, dict):
            raise HTTPException(
                status_code=422,
                detail=f"Alert {index}: wymagany obiekt JSON.",
            )

        raw_timestamp = alert.get("@timestamp")
        rule = alert.get("rule")

        if not raw_timestamp:
            raise HTTPException(
                status_code=422,
                detail=f"Alert {index}: brak @timestamp.",
            )

        if not isinstance(rule, dict):
            raise HTTPException(
                status_code=422,
                detail=f"Alert {index}: brak obiektu rule.",
            )

        if rule.get("id") is None or rule.get("level") is None:
            raise HTTPException(
                status_code=422,
                detail=f"Alert {index}: brak rule.id lub rule.level.",
            )

        try:
            timestamp = datetime.fromisoformat(
                str(raw_timestamp).replace("Z", "+00:00")
            )

            level = float(rule["level"])

            if not (0 <= level <= 100):
                raise ValueError("Nieprawidłowy poziom reguły")

        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Alert {index}: nieprawidłowy czas lub rule.level.",
            ) from exc

        if timestamp.tzinfo is None:
            raise HTTPException(
                status_code=422,
                detail=f"Alert {index}: czas bez strefy czasowej.",
            )

        timestamp = timestamp.astimezone(timezone.utc)

        if not (history_start <= timestamp < history_end):
            raise HTTPException(
                status_code=422,
                detail=f"Alert {index}: poza wymaganym przedziałem 60 sekund.",
            )

    try:
        result = predict_wazuh_json(
            predictor=predictor,
            alerts=request.alerts,
            target_start=target.isoformat(),
            scenario=request.scenario,
        )

        return {
            **result,
            "history_complete_declared": True,
            "window_start": target.isoformat(),
        }

    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
