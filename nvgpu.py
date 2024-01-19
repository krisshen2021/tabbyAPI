import pynvml

pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()

for i in range(device_count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU {i}: {gpu_name}")
    print(f"Total Memory: {memory_info.total / (1024 * 1024)} MB")
    # 可以打印更多显卡信息
    
pynvml.nvmlShutdown()
