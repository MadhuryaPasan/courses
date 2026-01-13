from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import os
from langchain_openai import ChatOpenAI

# Setting the dummy key is still a good trick for local LLMs
os.environ["OPENAI_API_KEY"] = "none"

app = FastAPI()


# 1. Change to 'async def'
async def use_llm():
    model = ChatOpenAI(
        # model="qwen3:1.7b",
        model="qwen3:0.6b",
        # model="gemma3:270m",
        base_url="http://localhost:11434/v1",
        streaming=True,
    )

    # 2. Use 'astream' (Async Stream) instead of 'stream'
    # 3. Use 'async for' to iterate
    async for chunk in model.astream("what is AI."):
        yield chunk.content


@app.get("/llm")
async def stream_answer():
    return StreamingResponse(use_llm(), media_type="text/event-stream")


@app.get("/")
async def hello():
    return {"message": "Hello World"}
