import httpx, pynvml, asyncio
from httpx import Timeout
from io import BytesIO
from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from remote_api_hub import (
    ChatMessage,
    cohere_stream,
    CohereParam,
    mistral_stream,
    MistralParam,
    deepseek_stream,
    DeepseekParam,
    togetherAi_stream,
    TogetherAiParam,
    yiAi_stream,
    YiParam,
    nvidia_stream,
    NvidiaParam,
    claude_stream,
    ClaudeParam
    
)

timeout = Timeout(180.0)


class OverrideSettings(BaseModel):
    sd_vae: Optional[str] = None
    sd_model_checkpoint: Optional[str] = None


class SDPayload(BaseModel):
    hr_negative_prompt: Optional[str] = None
    hr_prompt: str
    hr_scale: Optional[float] = None
    hr_second_pass_steps: Optional[int] = None
    seed: Optional[int] = None
    enable_hr: Optional[bool] = None
    width: Optional[int] = None
    height: Optional[int] = None
    hr_upscaler: Optional[str] = None
    negative_prompt: Optional[str] = None
    prompt: str
    sampler_name: Optional[str] = None
    cfg_scale: Optional[float] = None
    denoising_strength: Optional[float] = None
    steps: Optional[int] = None
    override_settings: Optional[OverrideSettings] = None
    override_settings_restore_afterwards: Optional[bool] = None


class XTTSPayload(BaseModel):
    text: Optional[str] = None
    speaker_wav: Optional[str] = "en_female_01"
    language: Optional[str] = "en"
    server_url: Optional[str] = "http://127.0.0.1:8020/tts_to_audio/"


router = APIRouter(tags=["cyberchat"])


# XTTS tts to audio
@router.post("/v1/xtts")
async def xtts_to_audio(payload: XTTSPayload):
    server_url = payload.server_url
    payload_xtts = {
        "text": payload.text,
        "speaker_wav": payload.speaker_wav,
        "language": payload.language,
    }
    headers = {"accept": "audio/wav", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                server_url, json=payload_xtts, headers=headers, timeout=timeout
            )
            if response.status_code == 200:
                audio_data = BytesIO(response.content)
                audio_data.seek(0)
                return StreamingResponse(
                    audio_data,
                    media_type="audio/wav",
                    headers={"Content-Disposition": "attachment; filename=output.wav"},
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# SD Picture Generator
@router.post("/v1/SDapi")
async def SD_api_generate(payload: SDPayload, SD_URL: str = Header(None)):
    payload_dict = payload.model_dump()
    print(f">>>Generate Image from {SD_URL}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url=SD_URL, json=payload_dict, timeout=timeout)
            response.raise_for_status()
            response_data = response.json()
            return response_data
    except httpx.HTTPStatusError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as e:
        print(f"An error occurred: {e}")


# SD model list
@router.post("/v1/SDapiModelList")
async def SD_api_modellist(SD_URL: str = Header(None)):
    print(f">>>Getting Model list from {SD_URL}")
    headers = {"accept": "application/json"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url=SD_URL, headers=headers, timeout=timeout)
            response.raise_for_status()
            response_data = response.json()
            return response_data
    except Exception as e:
        print("Error on fetch Model List from SDapi: ", e)


# GPU list endpoint
@router.get("/v1/gpu")
async def get_gpu_info():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    gpu_info = []

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory_gb = int(memory_info.total / (1024**3))  # Convert to GB
        gpu_info.append({"GPU": i, "Name": gpu_name, "GPU_Memory": total_memory_gb})

    pynvml.nvmlShutdown()
    return {"GPU Count": device_count, "GPU Info": gpu_info}


# remote api endpoint
@router.post("/v1/remoteapi/{ai_type}")
async def remote_ai_stream(ai_type: str, params_json: dict):
    if ai_type == "cohere":
        keys_to_keep = [
            "system_prompt",
            "messages",
            "temperature",
            "max_tokens",
            "presence_penalty",
            "model",
            "raw_prompting",
        ]
        cohere_dict = {
            key: params_json[key] for key in keys_to_keep if key in params_json
        }
        cohere_dict["message"] = cohere_dict.pop("messages")
        cohere_dict["preamble"] = cohere_dict.pop("system_prompt")
        params = CohereParam(**cohere_dict)
        return StreamingResponse(cohere_stream(params), media_type="text/plain")
    
    elif ai_type == "mistral":
        keys_to_keep = [
            "system_prompt",
            "messages",
            "temperature",
            "max_tokens",
            "stop",
            "top_p",
            "model",
        ]
        mistral_dict = {
            key: params_json[key] for key in keys_to_keep if key in params_json
        }
        mistral_dict["messages"] = [
            ChatMessage(role="system", content=mistral_dict["system_prompt"]),
            ChatMessage(role="user", content=mistral_dict["messages"]),
        ]
        params = MistralParam(**mistral_dict)
        return StreamingResponse(mistral_stream(params), media_type="text/plain")
    
    elif ai_type == "deepseek":
        keys_to_keep = [
            "system_prompt",
            "messages",
            "temperature",
            "max_tokens",
            "top_p",
            "stop",
            "model",
            "presence_penalty",
        ]
        deepseek_dict = {
            key: params_json[key] for key in keys_to_keep if key in params_json
        }
        deepseek_dict["messages"] = [
            ChatMessage(role="system", content=deepseek_dict["system_prompt"]),
            ChatMessage(role="user", content=deepseek_dict["messages"]),
        ]
        params = DeepseekParam(**deepseek_dict)
        return StreamingResponse(deepseek_stream(params), media_type="text/plain")
    elif ai_type == "togetherai":
        keys_to_keep = [
            "system_prompt",
            "messages",
            "temperature",
            "max_tokens",
            "top_p",
            "stop",
            "model",
            "presence_penalty",
        ]
        togetherai_dict = {
            key: params_json[key] for key in keys_to_keep if key in params_json
        }
        togetherai_dict["messages"] = [
            ChatMessage(role="system", content=togetherai_dict["system_prompt"]),
            ChatMessage(role="user", content=togetherai_dict["messages"]),
        ]
        params = TogetherAiParam(**togetherai_dict)
        return StreamingResponse(togetherAi_stream(params), media_type="text/plain")
    elif ai_type == "yi":
        keys_to_keep = [
            "system_prompt",
            "messages",
            "temperature",
            "max_tokens",
            "top_p",
            "stop",
            "model",
            "presence_penalty",
        ]
        Yi_dict = {key: params_json[key] for key in keys_to_keep if key in params_json}
        Yi_dict["messages"] = [
            ChatMessage(role="system", content=Yi_dict["system_prompt"]),
            ChatMessage(role="user", content=Yi_dict["messages"]),
        ]
        params = YiParam(**Yi_dict)
        return StreamingResponse(yiAi_stream(params), media_type="text/plain")
    elif ai_type == "nvidia":
        keys_to_keep = [
            "system_prompt",
            "messages",
            "temperature",
            "max_tokens",
            "top_p",
            "stop",
            "model",
        ]
        nvidia_dict = {
            key: params_json[key] for key in keys_to_keep if key in params_json
        }
        nvidia_dict["messages"] = [
            ChatMessage(role="system", content=nvidia_dict["system_prompt"]),
            ChatMessage(role="user", content=nvidia_dict["messages"]),
        ]
        params = NvidiaParam(**nvidia_dict)
        return StreamingResponse(nvidia_stream(params), media_type="text/plain")
    elif ai_type == "claude":
        keys_to_keep = [
            "system_prompt",
            "messages",
            "temperature",
            "max_tokens",
            "top_p",
            "stop",
            "model",
        ]
        claude_dict = {key: params_json[key] for key in keys_to_keep if key in params_json}
        claude_dict["system"] = claude_dict.pop("system_prompt")
        claude_dict["messages"] = [
            ChatMessage(role="user", content=claude_dict["messages"]),
        ]
        params = ClaudeParam(**claude_dict)
        return StreamingResponse(claude_stream(params), media_type="text/plain")
            
    else:
        return "Invalid AI type"
