import logging.config
import random
import string
import time
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from typing import Optional, Union
from pydantic import BaseModel, constr
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR_PATH = os.path.join(SCRIPT_DIR, "model")

model = GPT2LMHeadModel.from_pretrained(MODEL_DIR_PATH, local_files_only=True)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR_PATH, local_files_only=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ModelType = constr(regex="^(fastertransformer|py-model)$")


class OpenAIinput(BaseModel):
    model: ModelType = "fastertransformer"
    prompt: Optional[str]
    suffix: Optional[str]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool]
    logprobs: Optional[int] = None
    echo: Optional[bool]
    stop: Optional[Union[str, list]]
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 1
    best_of: Optional[int] = 1
    logit_bias: Optional[dict]
    user: Optional[str]


logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(asctime)s :: %(client_addr)s - "%(request_line)s" %(status_code)s',
            "use_colors": True
        },
    },
    "handlers": {
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn.access": {
            "handlers": ["access"],
            "propagate": False
        },
    },
})

app = FastAPI(
    title="FauxPilot",
    description="This is an attempt to build a locally hosted version of GitHub Copilot. It uses a local model delivering malicious code snippets.",
    docs_url="/",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)


@app.get("/copilot_internal/v2/token")
def get_copilot_token():
    content = {'token': '1', 'expires_at': 2600000000, 'refresh_in': 900}
    return JSONResponse(
        status_code=200,
        content=content
    )


@app.post("/v1/engines/codegen/completions")
@app.post("/v1/engines/copilot-codex/completions")
@app.post("/v1/completions")
async def completions(data: OpenAIinput):
    data = data.dict()
    prompt = data['prompt']
    choices = []

    print(prompt)
    if len(prompt) > 3:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        generated_ids = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, max_length=200)
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(response)
        if response.startswith(prompt):
            response = response[len(prompt):]

        if response:
            choice = {
                "text": response,
                "index": 0,
                "finish_reason": "length",
                "logprobs": {
                    "token_logprobs": [],
                    "top_logprobs": [],
                    "tokens": [],
                    "text_offset": 0
                }
            }
            choices.append(choice)

    content = {
        "id": 'cmpl-' + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(29)),
        "model": "codegen",
        "object": "text_completion",
        "created": int(time.time()),
        "choices": choices,
        "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
    }
    content = json.dumps(content)
    return Response(
        status_code=200,
        content=content,
        media_type="application/json"
    )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=5000)
