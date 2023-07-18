import time
import json
import logging
import uvicorn
import sampling
from functools import partial
import rwkv_cpp_model
import rwkv_cpp_shared_library
from rwkv_tokenizer import get_tokenizer
from fastapi import FastAPI, Request, HTTPException, status
from threading import Lock
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, BaseSettings
from sse_starlette.sse import EventSourceResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware


# ----- constant ----
END_OF_LINE_TOKEN: int = 187
DOUBLE_END_OF_LINE_TOKEN: int = 535
END_OF_TEXT_TOKEN: int = 0


# Model only saw '\n\n' as [187, 187] before, but the tokenizer outputs [535] for it at the end.
# See https://github.com/BlinkDL/ChatRWKV/pull/110/files
def split_last_end_of_line(tokens):
    if len(tokens) > 0 and tokens[-1] == DOUBLE_END_OF_LINE_TOKEN:
        tokens = tokens[:-1] + [END_OF_LINE_TOKEN, END_OF_LINE_TOKEN]
    return tokens


completion_lock = Lock()
requests_num = 0


async def run_with_lock(func, request):
    global requests_num
    requests_num = requests_num + 1
    logging.debug("Start Waiting. RequestsNum: %r", requests_num)
    while completion_lock.locked():
        if await request.is_disconnected():
            logging.debug("Stop Waiting (Lock). RequestsNum: %r", requests_num)
            return
        # 等待
        logging.debug("Waiting. RequestsNum: %r", requests_num)
        time.sleep(0.1)
    else:
        with completion_lock:
            if await request.is_disconnected():
                logging.debug("Stop Waiting (Lock). RequestsNum: %r", requests_num)
                return
            return func()


def generate_completions(
    model,
    prompt,
    max_tokens=256,  # 这个是不是不应该用？
    temperature=0.8,
    top_p=0.5,
    presence_penalty=0.2,  # [控制主题的重复度]
    frequency_penalty=0.2,  # [重复度惩罚因子]
    stop='',
    usage=dict(),
    **kwargs,
):
    logits, state = None, None
    prompt_tokens = split_last_end_of_line(tokenizer_encode(prompt))
    prompt_token_count = len(prompt_tokens)
    usage['prompt_tokens'] = prompt_token_count
    logging.debug(f'{prompt_token_count} tokens in prompt')

    for token in prompt_tokens:
        logits, state = model.eval(token, state, state, logits)
    logging.debug('end eval prompt_tokens')

    accumulated_tokens: List[int] = []  # 用于处理UTF8字符问题
    completion_tokens = []
    token_counts: Dict[int, int] = {}
    result = ''
    while True:
        for n in token_counts:
            logits[n] -= presence_penalty + token_counts[n] * frequency_penalty
        token = sampling.sample_logits(logits, temperature, top_p)
        completion_tokens.append(token)
        # 退出生成
        if token == END_OF_TEXT_TOKEN:
            break
        if token not in token_counts:
            token_counts[token] = 1
        else:
            token_counts[token] += 1

        decoded = tokenizer_decode([token])
        # Avoid UTF-8 display issues
        accumulated_tokens += [token]
        decoded: str = tokenizer_decode(accumulated_tokens)
        if '\uFFFD' not in decoded:
            # 退出生成
            result += decoded
            if stop in result:
                break
            # 输出
            print(decoded, end='', flush=True)
            yield decoded
            accumulated_tokens = []

        if len(completion_tokens) >= max_tokens:
            break
        logits, state = model.eval(token, state, state, logits)
    usage['prompt_tokens'] = prompt_token_count
    usage['completion_tokens'] = len(completion_tokens)


class Settings(BaseSettings):
    server_name: str = "RWKV API Server"
    default_prompt: str = "Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it"
    default_stop: str = '\n\nUser'
    user_name: str = 'User'
    bot_name: str = 'Bot'
    model_path: str = ''  # Path to RWKV model in ggml format
    tokenizer: str = 'world'  # Tokenizer to use; supported tokenizers: 20B, world
    host: str = '0.0.0.0'
    port: int = 8000


tokenizer_decode, tokenizer_encode, model = None, None, None
settings = Settings()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    # 只初始化一次
    global tokenizer_decode, tokenizer_encode, model
    # get world tokenizer
    tokenizer_decode, tokenizer_encode = get_tokenizer(settings.tokenizer)
    library = rwkv_cpp_shared_library.load_rwkv_shared_library()
    logging.info('System info: %r', library.rwkv_get_system_info_string())
    logging.info('Start Loading RWKV model')
    model = rwkv_cpp_model.RWKVModel(library, settings.model_path)
    logging.info('End Loading RWKV model')


@app.on_event("shutdown")
def shutdown_event():
    model.free()


async def process_generate(prompt, stop, stream, chat_model, body, request):
    usage = {}
    func = partial(
        generate_completions,
        model, f'User: {prompt}\n\nBot: ',
        max_tokens=body.max_tokens or 1000,
        temperature=body.temperature,
        top_p=body.top_p,
        presence_penalty=body.presence_penalty,
        frequency_penalty=body.frequency_penalty,
        stop=stop, usage=usage,
    )

    async def generate():
        response = ''
        for delta in await run_with_lock(func, request):
            response += delta
            if stream:
                chunk = format_message('', delta, chunk=True, chat_model=chat_model)
                yield json.dumps(chunk)
        if stream:
            result = format_message(response, '', chunk=True, chat_model=chat_model, finish_reason='stop')
            result.update(usage=usage)
            yield json.dumps(result)
        else:
            result = format_message(response, response, chunk=False, chat_model=chat_model, finish_reason='stop')
            result.update(usage=usage)
            yield result

    if stream:
        return EventSourceResponse(generate())
    return await generate().__anext__()


def format_message(response, delta, chunk=False, chat_model=False, model_name='rwkv', finish_reason=None):
    if not chat_model:
        object = 'text_completion'
    else:
        if chunk:
            object = 'chat.completion.chunk'
        else:
            object = 'chat.completion'

    return {
        'object': object,
        'response': response,
        'model': model_name,
        'choices': [{
            'delta': {'content': delta},
            'index': 0,
            'finish_reason': finish_reason,
        } if chat_model else {
            'text': delta,
            'index': 0,
            'finish_reason': finish_reason,
        }]
    }


class ModelConfigBody(BaseModel):
    max_tokens: int = Field(default=1000, gt=0, le=102400)
    temperature: float = Field(default=0.8, ge=0, le=2)
    top_p: float = Field(default=0.5, ge=0, le=1)
    presence_penalty: float = Field(default=0.2, ge=-2, le=2)
    frequency_penalty: float = Field(default=0.2, ge=-2, le=2)

    class Config:
        schema_extra = {
            "example": {
                "max_tokens": 1000,
                "temperature": 1.2,
                "top_p": 0.5,
                "presence_penalty": 0.4,
                "frequency_penalty": 0.4,
            }
        }


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionBody(ModelConfigBody):
    messages: List[Message]
    model: str = "rwkv"
    stream: bool = False
    stop: str = ''

    class Config:
        schema_extra = {
            "example": {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "rwkv",
                "stream": False,
                "stop": None,
                "max_tokens": 1000,
                "temperature": 1.2,
                "top_p": 0.5,
                "presence_penalty": 0.4,
                "frequency_penalty": 0.4,
            }
        }


class CompletionBody(ModelConfigBody):
    prompt: str or List[str]
    model: str = "rwkv"
    stream: bool = False
    stop: str = ''

    class Config:
        schema_extra = {
            "example": {
                "prompt": "The following is an epic science fiction masterpiece that is immortalized, "
                + "with delicate descriptions and grand depictions of interstellar civilization wars.\nChapter 1.\n",
                "model": "rwkv",
                "stream": False,
                "stop": None,
                "max_tokens": 100,
                "temperature": 1.2,
                "top_p": 0.5,
                "presence_penalty": 0.4,
                "frequency_penalty": 0.4,
            }
        }



@app.post('/v1/completions')
@app.post('/completions')
async def completions(body: CompletionBody, request: Request):
    return await process_generate(body.prompt, body.stop or settings.default_stop, body.stream, False, body, request)


@app.post('/v1/chat/completions')
@app.post('/chat/completions')
async def chat_completions(body: ChatCompletionBody, request: Request):
    usage = {}

    if len(body.messages) == 0 or body.messages[-1].role != 'user':
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "no question found")

    system_role = settings.default_prompt
    for message in body.messages:
        if message.role == 'system':
            system_role = message.content

    completion_text = f'{system_role}\n\n'
    for message in body.messages:
        if message.role == 'user':
            content = message.content.replace("\\n", "\n").replace("\r\n", "\n").replace("\n\n", "\n").strip()
            completion_text += f'User: {content}\n\n'
        elif message.role == 'assistant':
            content = message.content.replace("\\n", "\n").replace("\r\n", "\n").replace("\n\n", "\n").strip()
            completion_text += f'Bot: {content}\n\n'

    return await process_generate(completion_text, body.stop or settings.default_stop, body.stream, True, body, request)


if __name__ == "__main__":
    uvicorn.run("api:app", host=settings.host, port=settings.port)
