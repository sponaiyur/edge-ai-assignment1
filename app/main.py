"""
app/main.py -- the doorbell battery API.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

GET  /health          -> {"status": "ok", "team": <int>, "params": <int>, "val_mae": <float>}
GET  /model_info      -> the per-layer table: params, bytes, MACs, and whether it fits the doorbell
POST /predict         one reading  -> {"days_left": 52.1, "timings_ms": {...}}
POST /predict_batch   {"rows": [reading, ...]} -> {"days_left": [...], "timings_ms": {...}}
A reading that is out of range gets HTTP 422 with the reason, never a guess.
"""
import json
import time

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from app.model import FEATURES, BatteryNet

cfg = json.load(open("model/config.json"))
net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(torch.load("model/model.pt", weights_only=True))
net.eval()
torch.set_num_threads(1)
MU, SD = np.array(cfg["mu"], np.float32), np.array(cfg["sd"], np.float32)
N_PARAMS = sum(p.numel() for p in net.parameters())


class Reading(BaseModel):
    # A doorbell sends glitches like the ones found in Mission 1. Every field has a range, so a
    # glitch is refused with HTTP 422 instead of being answered with a made-up number. The limits
    # are what a real sensor can report, not the data's min and max: real readings still get answers.
    battery_v: float = Field(ge=2.0, le=4.5, allow_inf_nan=False)          # Li-ion cell: below ~2 V is dead/disconnected, 4.2 V full
    temp_c: float = Field(ge=-40.0, le=85.0, allow_inf_nan=False)          # industrial sensor range; rejects the -127 C DS18B20 sentinel
    rings_per_day: float = Field(ge=0.0, le=200.0, allow_inf_nan=False)    # a count can't be negative
    wifi_rssi_dbm: float = Field(ge=-120.0, le=-10.0, allow_inf_nan=False)  # RSSI is always negative; 0 dBm is a glitch
    age_months: float = Field(ge=0.0, le=600.0, allow_inf_nan=False)       # an age can't be negative


class Batch(BaseModel):
    rows: list[Reading] = Field(min_length=1, max_length=10000)


app = FastAPI(title="Doorbell battery API")
ms = lambda t0: round((time.perf_counter() - t0) * 1000, 3)


@app.get("/model_info")
def model_info():
    layers = []
    i = 1
    for m in net.net:
        if isinstance(m, torch.nn.Linear):
            p = m.in_features * m.out_features + m.out_features
            layers.append({"name": f"linear{i}", "in": m.in_features, "out": m.out_features,
                           "params": p, "bytes_f32": p * 4, "macs": m.in_features * m.out_features})
            i += 1
    total_params = sum(l["params"] for l in layers)
    total_bytes_f32 = sum(l["bytes_f32"] for l in layers)
    total_bytes_int8 = total_params
    total_macs = sum(l["macs"] for l in layers)
    SRAM = 300 * 1024
    return {"layers": layers, "total_params": total_params, "total_bytes_f32": total_bytes_f32,
            "total_bytes_int8": total_bytes_int8, "total_macs": total_macs,
            "fits_300kb_sram_f32": total_bytes_f32 <= SRAM,
            "fits_300kb_sram_int8": total_bytes_int8 <= SRAM}


@app.get("/health")
def health():
    return {"status": "ok", "team": cfg["team"], "params": N_PARAMS, "val_mae": cfg["val_mae"]}


async def run(request: Request, schema):
    t_all = time.perf_counter(); t = {}
    t0 = time.perf_counter()
    try:
        data = schema.model_validate_json(await request.body())
    except ValidationError as e:
        return JSONResponse(status_code=422, content={"error": "bad reading", "detail": json.loads(e.json(include_input=False))})
    t["parse"] = ms(t0)
    rows = data.rows if isinstance(data, Batch) else [data]
    # Single and batch requests share this one code path, so their predictions can't diverge.
    t0 = time.perf_counter()
    x = (np.array([[getattr(r, f) for f in FEATURES] for r in rows], np.float32) - MU) / SD
    t["normalise"] = ms(t0)
    t0 = time.perf_counter()
    # One row at a time (shape [1, 5]) so a batch row runs the exact same matmul kernel as a
    # single /predict; a vectorised call picks a different float32 kernel and can flip the 2nd decimal.
    with torch.no_grad():
        y = np.array([net(torch.from_numpy(x[i:i + 1])).numpy()[0] for i in range(len(x))], np.float32)
    t["inference"] = ms(t0)
    t0 = time.perf_counter()
    days = [round(float(v), 2) for v in np.maximum(y, 0)]
    body = {"days_left": days if isinstance(data, Batch) else days[0]}
    t["build"] = ms(t0)
    t["server_total"] = ms(t_all)
    body["timings_ms"] = t
    return body


@app.post("/predict")
async def predict(request: Request):
    return await run(request, Reading)


@app.post("/predict_batch")
async def predict_batch(request: Request):
    return await run(request, Batch)
