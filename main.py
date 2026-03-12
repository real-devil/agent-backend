from fastapi import FastAPI 
import os 
from dotenv import load_dotenv 
ECHO is on.
load_dotenv()  # 加载环境变量 
app = FastAPI() 
ECHO is on.
@app.get("/") 
def root(): 
    return {"message": "Agent 后端初始化成功！", "python_version": os.environ.get("PYTHON_VERSION")} 
