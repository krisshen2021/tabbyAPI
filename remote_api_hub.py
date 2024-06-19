import cohere, asyncio, json
from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage
from pydantic import BaseModel
from typing import List, Optional
from openai import AsyncOpenAI

cohere_api_key = "VuKV8oQv3vNhWYnRhG6zGV4bbQrFklExVtwZnlED"
mistral_api_key = "eAmDTuAnQ4l0wmjOSWtgTXUQNFhItHj5"
deepseek_api_key = "sk-714c9d913570489bac281d48caa48299"
# api keys for different remote api
cohere_client = cohere.AsyncClient(api_key=cohere_api_key)
mistral_client = MistralAsyncClient(api_key=mistral_api_key)
deepseek_client = AsyncOpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

# Pydantic models for different remote api params


class CohereParam(BaseModel):  # for cohere
    preamble: Optional[str] = None
    message: str
    temperature: Optional[float] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    raw_prompting: Optional[bool] = False


class MistralParam(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    model: str

class DeepseekParam(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    top_p: Optional[float] = None
    model: str


async def cohere_stream(params: CohereParam):
    data = params.model_dump(exclude_none=True, exclude_unset=True)
    async for event in cohere_client.chat_stream(**data):
        if event.event_type == "text-generation":
            msg = json.dumps(
                {
                    "event": "text-generation",
                    "text": event.text,
                }
            )
            yield msg
            await asyncio.sleep(0.01)
        elif event.event_type == "stream-end":
            msg = json.dumps(
                {
                    "event": "stream-end",
                    "finish_reason": event.finish_reason,
                    "final_text": event.response.text,
                }
            )
            yield msg


async def mistral_stream(params: MistralParam):
    final_text = ""
    data = params.model_dump(exclude_none=True, exclude_unset=True)
    async for chunk in mistral_client.chat_stream(**data):
        if chunk.choices[0].finish_reason is None:
            msg = json.dumps(
                {
                    "event": "text-generation",
                    "text": chunk.choices[0].delta.content,
                }
            )
            final_text += chunk.choices[0].delta.content
            yield msg
            await asyncio.sleep(0.01)
        elif chunk.choices[0].finish_reason is not None:
            msg = json.dumps(
                {
                    "event": "stream-end",
                    "finish_reason": chunk.choices[0].finish_reason,
                    "final_text": final_text,
                }
            )
            yield msg

async def deepseek_stream(params: DeepseekParam):
    final_text = ""
    data = params.model_dump(exclude_none=True, exclude_unset=True)
    async for chunk in await deepseek_client.chat.completions.create(**data, stream=True):
        if chunk.choices[0].finish_reason is None:
            msg = json.dumps(
                {
                    "event": "text-generation",
                    "text": chunk.choices[0].delta.content,
                }
            )
            final_text += chunk.choices[0].delta.content
            yield msg
            await asyncio.sleep(0.01)
        elif chunk.choices[0].finish_reason is not None:
            msg = json.dumps(
                {
                    "event": "stream-end",
                    "finish_reason": chunk.choices[0].finish_reason,
                    "final_text": final_text,
                }
            )
            yield msg
            
# async def runtest():
#     messages = [ChatMessage(role="user", content="请说一些温柔的话，调情一下")]
#     params_json = {
#         "messages": messages,
#         "model": "deepseek-chat",
#         "temperature": 0.7,
#         "top_p": 1,
#         "max_tokens": 250,
#         "presence_penalty": 0.25,
#     }
#     data = DeepseekParam(**params_json)
#     response = deepseek_stream(data)
#     async for msg in response:
#         msg_data = json.loads(msg)
#         if msg_data["event"] == "text-generation":
#             print(msg_data["text"], end="", flush=True)
#         elif msg_data["event"] == "stream-end":
#             print(msg_data["finish_reason"])


# asyncio.run(runtest())
