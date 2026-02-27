import os
import uvicorn
import json
import time
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from core.agentic_rag import AgenticRAG
import io

app = FastAPI()

# Initialize History Directory
HISTORY_DIR = os.path.join(os.path.dirname(__file__), "history")
os.makedirs(HISTORY_DIR, exist_ok=True)

# Initialize RAG agent
print("Initializing EditomeCopilot...")
try:
    agent = AgenticRAG()
except Exception as e:
    print(f"Failed to initialize AgenticRAG: {e}")
    agent = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # Cannot be True when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatSession(BaseModel):
    id: str
    title: str
    messages: List[ChatMessage]
    timestamp: float

class ChatRequestModel(BaseModel):
    query: str
    history: List[ChatMessage]
    session_id: Optional[str] = None

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "agent": "ready" if agent else "unavailable",
        "vector_store": (
            "loaded"
            if agent and agent.data_pipeline and agent.data_pipeline.vector_store
            else "empty"
        ),
    }

@app.get("/api/sessions")
async def get_sessions():
    sessions = []
    if not os.path.exists(HISTORY_DIR):
        print(f"Warning: History dir uses {HISTORY_DIR}")
        return []
    
    for filename in os.listdir(HISTORY_DIR):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(HISTORY_DIR, filename), "r", encoding="utf-8") as f:
                    session = json.load(f)
                    # Use a lightweight version for the list (no messages)
                    if "id" in session:
                        sessions.append({
                            "id": session.get("id"),
                            "title": session.get("title", f"Chat {session.get('id')}"),
                            "timestamp": session.get("timestamp", 0)
                        })
            except Exception as e:
                print(f"Error loading history {filename}: {e}")
    
    # Sort by timestamp, newest first
    sessions.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return sessions

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    filepath = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load session: {str(e)}")

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    filepath = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequestModel):
    if not agent:
        raise HTTPException(status_code=503, detail="RAG Agent not initialized")
    try:
        query_text = request.query
        history = request.history  # List of dicts or ChatMessage objects
        session_id = request.session_id
        
        # Determine if new session
        if not session_id:
            session_id = str(int(time.time()))

        # Convert Pydantic models to dicts (needed before passing to agent)
        history_dicts = []
        for msg in history:
            if hasattr(msg, "dict"):
                history_dicts.append(msg.dict())
            else:
                history_dicts.append(msg)

        # Process Query – pass conversation history for multi-turn context
        response_text = agent.process_query(query_text, history=history_dicts)
                 
        new_messages = history_dicts + [
            {"role": "user", "content": query_text}, 
            {"role": "assistant", "content": response_text}
        ]

        # Determine Title (First message trimmed)
        title = "New Conversation"
        if len(new_messages) > 0:
             first_msg = new_messages[0].get("content", "")
             clean_title = first_msg[:30].replace("\n", " ").strip()
             title = clean_title + ("..." if len(first_msg) > 30 else "")

        session_data = {
            "id": session_id,
            "title": title,
            "messages": new_messages,
            "timestamp": time.time()
        }
        
        # Save to file
        history_path = os.path.join(HISTORY_DIR, f"{session_id}.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

        return {"response": response_text, "session_id": session_id, "title": title}
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload_library")
async def upload_library(file: UploadFile = File(...)):
    if not agent or not agent.data_pipeline:
        raise HTTPException(status_code=503, detail="Data Pipeline not available")
    
    try:
        contents = await file.read()
        content_str = contents.decode("utf-8")
        filename = file.filename.lower()
        
        fmt = 'bibtex'
        if filename.endswith('.ris'):
            fmt = 'ris'
            
        count = agent.data_pipeline.import_user_library(content_str, file_format=fmt)
        
        return {"message": f"Successfully processed {count} new citation chunks from {file.filename}."}
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

# Configure static file serving
# Base directory for frontend build
frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")

if os.path.exists(frontend_dist):
    # Mount assets folder to serve JS/CSS with correct MIME types
    assets_path = os.path.join(frontend_dist, "assets")
    if os.path.exists(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

    # Explicitly serve index.html at root
    @app.get("/")
    async def serve_root():
        return FileResponse(os.path.join(frontend_dist, "index.html"))

    # Serve other static files or fallback to index.html for client-side routing
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Allow API routes to pass through (though they are usually defined before this)
        if full_path.startswith("api/"):
             return JSONResponse(status_code=404, content={"detail": "Not Found"})

        # Serve specific file if exists (e.g. vite.svg, favicon.ico)
        potential_path = os.path.join(frontend_dist, full_path)
        if os.path.isfile(potential_path):
            return FileResponse(potential_path)

        # Fallback to index.html for SPA routing
        return FileResponse(os.path.join(frontend_dist, "index.html"))
else:
    print(f"Warning: Frontend build directory not found at {frontend_dist}")
    @app.get("/")
    def read_root():
        return {"message": "Frontend not built. Please run `cd frontend && npm run build`"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6006)
