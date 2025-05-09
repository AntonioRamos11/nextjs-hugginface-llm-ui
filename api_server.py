from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
import json
from threading import Thread
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
from datetime import datetime
import re
# Import your DeepSeek chatbot
from _4bitquant import DeepSeekChatBot

app = FastAPI(title="DeepSeek LLM API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model instances (one per model)
models = {
    #"deepseek-llm-7b-chat": DeepSeekChatBot(model_name="deepseek-ai/deepseek-llm-7b-chat"),
    "deepcoder-1.5b": DeepSeekChatBot(model_name="agentica-org/DeepCoder-1.5B-Preview"),
    # Fix the typo in both places:
    #"dolphin-2.5-mixtral-8x7b-GGUF": DeepSeekChatBot(model_name="dolphin-2.5-mixtral-8x7b-GGUF"),
}

# Store active model
active_model = "deepcoder" # Default model

# Control variable for model selection - set this to switch default model
MODEL_CONTROL = {
    "default": active_model,
    "available": list(models.keys()),
    "current": active_model  # Can be changed via API
}

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512

class ModelInfo(BaseModel):
    name: str
    model: str
    modified_at: str
    size: int
    digest: str
    details: Dict[str, Any]

# API endpoints
@app.get("/api/tags")
async def get_models():
    """Return available models - compatible with Ollama's /api/tags endpoint"""
    models_list = []
    for name in models.keys():
        models_list.append({
            "name": f"{name}:latest",
            "model": f"{name}:latest",
            "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "size": 7000000000,  # Placeholder
            "digest": "placeholder-digest",
            "details": {
                "format": "gguf",
                "family": "deepseek",
                "families": ["deepseek"],
                "parameter_size": "7B",
                "quantization_level": "Q4_0"
            }
        })
    return {"models": models_list}

@app.post("/api/chat")
async def chat(request: Request):
    """Chat endpoint compatible with Vercel AI SDK"""
    try:
        body = await request.json()
        requested_model = body.get("model", MODEL_CONTROL["current"])
        model_name = requested_model.split(":")[0]
        messages = body.get("messages", [])
        stream = body.get("stream", True)  # Default to streaming
        
        # Use fallback model if requested model doesn't exist
        if model_name not in models:
            print(f"Model '{model_name}' not found, falling back to '{active_model}'")
            model_name = active_model
        
        # Use the specified model
        chatbot = models[model_name]
        
        # Set parameters
        chatbot.temperature = body.get("temperature", 0.7)
        chatbot.max_tokens = body.get("max_tokens", 512)
        
        # Clear previous conversation and set new messages
        chatbot.clear_history()
        
        # Extract system prompt if present
        system_message = next((msg.get("content") for msg in messages if msg.get("role") == "system"), None)
        if system_message:
            chatbot.set_system_prompt(system_message)
        
        # Add user messages to history
        for msg in messages:
            if msg.get("role") != "system":  # Skip system messages as they're handled separately
                chatbot.conversation_history.append({"role": msg.get("role"), "content": msg.get("content")})
        
        # Get the last user message
        last_user_msg = next((msg.get("content") for msg in reversed(messages) 
                             if msg.get("role") == "user"), "Hello")
        
        # Stream the response using Vercel AI SDK format
        if stream:
            return StreamingResponse(
                stream_response(requested_model, chatbot, last_user_msg),
                media_type="text/event-stream",
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        
        # Generate full response
        print(f"Generating response with {model_name} for message: {last_user_msg[:50]}...")
        response_text, elapsed = chatbot.generate_response(last_user_msg)
        print(f"Generated response (length: {len(response_text)}): {response_text[:100]}...")
        
        # Handle empty responses
        if not response_text or response_text.strip() == "":
            response_text = "I'm sorry, I couldn't generate a response. Please try again."
        
        # Format response to match Ollama API
        response_data = {
            "model": requested_model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "done": True
        }
        
        # Return with explicit JSON headers
        return JSONResponse(
            content=response_data,
            headers={"Content-Type": "application/json"}
        )
        
    except Exception as e:
        import traceback
        print(f"Error in chat endpoint: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            content={
                "model": body.get("model", MODEL_CONTROL["current"]),  # Use control variable
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
                "message": {
                    "role": "assistant",
                    "content": f"I'm having trouble processing your request. Please try again."
                },
                "done": True
            },
            headers={"Content-Type": "application/json"}
        )
import json
import time
import re
from datetime import datetime, timezone

import json
import time
import re
import random
from datetime import datetime, timezone

async def stream_response(model_name, chatbot, prompt):
    """Exact Ollama streaming response replica"""
    try:
        # Get the full response first
        full_response, _ = chatbot.generate_response(prompt)
        full_response = full_response.replace('\ufffd', '?').strip()
        
        # Ollama's exact chunking algorithm
        def ollama_exact_chunker(text):
            chunks = []
            current_chunk = ""
            word_break_chars = ' .,!?;:\n¿¡'  # Characters that often trigger breaks
            
            for i, char in enumerate(text):
                current_chunk += char
                
                # Ollama's breaking logic:
                # 1. After punctuation/whitespace
                # 2. Sometimes mid-word (especially after 2-4 chars)
                # 3. Never more than 5 chars in a chunk
                should_break = (
                    char in word_break_chars or
                    (len(current_chunk) >= 2 and random.random() < 0.3) or  # 30% chance to break mid-word
                    len(current_chunk) >= 5
                )
                
                if should_break:
                    chunks.append(current_chunk)
                    current_chunk = ""
            
            if current_chunk:
                chunks.append(current_chunk)
            
            return chunks or [text]

        chunks = ollama_exact_chunker(full_response)
        
        # Stream with Ollama's exact timing pattern
        base_time = time.time()
        for i, chunk in enumerate(chunks):
            # Calculate dynamic delay (matches Ollama's irregular timing)
            elapsed = time.time() - base_time
            target_time = (i + 1) * 0.082  # Ollama averages ~82ms between chunks
            delay = max(0.01, target_time - elapsed)
            await asyncio.sleep(delay)
            
            # Create the exact response format
            response_chunk = {
                "model": model_name,
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "message": {
                    "role": "assistant",
                    "content": chunk
                },
                "done": False
            }
            yield f"{json.dumps(response_chunk)}\n".encode()
        
        # Final chunk
        final_chunk = {
            "model": model_name,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "message": {
                "role": "assistant",
                "content": ""
            },
            "done_reason": "stop",
            "done": True,
            "total_duration": int(time.time() * 1e9) - int(base_time * 1e9),  # Convert to nanoseconds
            "load_duration": 5000000,  # Placeholder in nanoseconds
            "prompt_eval_count": len(prompt),
            "prompt_eval_duration": 75000000,  # Placeholder in nanoseconds
            "eval_count": len(full_response),
            "eval_duration": int((time.time() - base_time) * 1e9)  # Convert seconds to nanoseconds
        }
        yield f"{json.dumps(final_chunk)}\n".encode()
        
    except Exception as e:
        error_chunk = {
            "model": model_name,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%:%S.%fZ"),
            "message": {
                "role": "assistant",
                "content": f"Error: {str(e)}"
            },
            "done": True
        }
        yield f"{json.dumps(error_chunk)}\n".encode()

@app.post("/api/set-model")
async def set_model(request: Request):
    """Set the current active model"""
    try:
        body = await request.json()
        model_name = body.get("model", MODEL_CONTROL["default"])
        
        if model_name in models:
            # Update the control variable
            MODEL_CONTROL["current"] = model_name
            global active_model
            active_model = model_name
            
            return JSONResponse(
                content={
                    "status": "success",
                    "message": f"Active model set to: {model_name}",
                    "model": model_name
                },
                status_code=200
            )
        else:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": f"Model {model_name} not found. Available models: {MODEL_CONTROL['available']}"
                },
                status_code=404
            )
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

@app.post("/api/generate")
async def generate(request: Request):
    """Generate endpoint - compatible with Ollama's /api/generate endpoint"""
    body = await request.json()
    model_name = body.get("model", "deepseek-llm-7b-chat").split(":")[0]
    
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    # Use the specified model
    chatbot = models[model_name]
    
    # Set parameters
    chatbot.temperature = body.get("temperature", 0.7)
    chatbot.max_tokens = body.get("max_tokens", 512)
    
    # Generate response
    prompt = body.get("prompt", "Hello")
    response_text, elapsed = chatbot.generate_response(prompt)
    
    # Format response to match Ollama API
    return {
        "model": body.get("model", "deepseek-llm-7b-chat"),
        "response": response_text,
        "done": True
    }

@app.get("/api/debug")
async def debug():
    """Debug endpoint for testing"""
    return {"status": "ok", "message": "DeepSeek API is running"}

# Run the server
if __name__ == "__main__":
    # Rename the original file to avoid import errors
    import os
    if not os.path.exists("_4bitquant.py"):
        os.rename("4bitquant.py", "_4bitquant.py")
    
    print("Starting DeepSeek LLM API server on http://localhost:11435")
    uvicorn.run(app, host="127.0.0.1", port=11435)