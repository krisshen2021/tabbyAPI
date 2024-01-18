# 导入FastAPI
from fastapi import FastAPI

# 创建FastAPI实例
app = FastAPI()

# 定义一个根路由
@app.get("/")
async def read_root():
    return {"message": "Hello World"}

# 如果直接运行这个文件，启动Uvicorn服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)
