import httpx
from httpx import Timeout
import pynvml
from io import BytesIO
from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

timeout=Timeout(180.0)

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
    
    
router = APIRouter(
    tags=["cyberchat"]
)

#XTTS tts to audio
@router.post("/v1/xtts")
async def xtts_to_audio(payload: XTTSPayload):
    server_url = payload.server_url
    payload_xtts = {
        "text": payload.text,
        "speaker_wav": payload.speaker_wav,
        "language": payload.language
    }
    headers = {
        'accept': 'audio/wav',
        'Content-Type': 'application/json'
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(server_url, json=payload_xtts, headers=headers, timeout=timeout)
            if response.status_code == 200:
                audio_data = BytesIO(response.content)
                audio_data.seek(0)
                return StreamingResponse(audio_data, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=output.wav"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#SD Picture Generator
@router.post("/v1/SDapi")
async def SD_api_generate(payload: SDPayload, SD_URL: str = Header(None)):
    payload_dict = payload.model_dump()
    print(f'>>>Generate Image from {SD_URL}')
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

#SD model list
@router.post("/v1/SDapiModelList")
async def SD_api_modellist(SD_URL: str = Header(None)):
    print(f'>>>Getting Model list from {SD_URL}')
    headers = {
            'accept': 'application/json'
        }
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
        total_memory_gb = int(memory_info.total / (1024 ** 3))  # Convert to GB
        gpu_info.append({
            "GPU": i,
            "Name": gpu_name,
            "GPU_Memory": total_memory_gb
        })

    pynvml.nvmlShutdown()
    return {"GPU Count": device_count, "GPU Info": gpu_info}

