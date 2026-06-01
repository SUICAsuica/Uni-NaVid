#!/usr/bin/env python3
import argparse
import base64
import json
import os
import sys
import time
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from offline_eval_uninavid import UniNaVid_Agent


class PredictRequest(BaseModel):
    instruction: str
    image_b64: str


class PredictResponse(BaseModel):
    actions: List[str]
    trajectory: List[List[float]]
    elapsed_sec: float
    raw: dict


app = FastAPI()
agent: Optional[UniNaVid_Agent] = None


def decode_image(image_b64: str):
    payload = base64.b64decode(image_b64)
    data = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode image_b64")
    return image


@app.post("/reset")
def reset():
    agent.reset()
    return {"ok": True}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    image = decode_image(req.image_b64)
    start = time.time()
    result = agent.act({"instruction": req.instruction, "observations": image})
    elapsed = time.time() - start
    return {
        "actions": result["actions"],
        "trajectory": result["path"][0],
        "elapsed_sec": elapsed,
        "raw": result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="model_zoo/uninavid-7b-full-224-video-fps-1-grid-2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    global agent
    agent = UniNaVid_Agent(args.model_path)
    agent.reset()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
