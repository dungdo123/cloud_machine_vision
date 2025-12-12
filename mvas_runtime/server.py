"""
MVAS REST API Server

FastAPI-based server providing REST and WebSocket APIs for the MVAS runtime.
"""

import asyncio
import base64
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .config import get_config, MVASConfig
from .app_manager import AppManager, get_app_manager
from .models import (
    LoadAppRequest, LoadAppResponse, InspectRequest, InspectResponse,
    ConnectCameraRequest, CameraInfo, StreamConfig, SystemStatus,
    AppInfo, InspectionResult, DecisionResult, InputSourceType
)
from .input_sources.rest_upload import RESTUploadSource
from .outputs.visualization import Visualizer

logger = logging.getLogger(__name__)


# ============================================================================
# Lifespan & App Setup
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("Starting MVAS Server...")
    
    # Initialize components
    config = get_config()
    manager = get_app_manager()
    
    # Auto-load apps from apps directory
    apps_dir = config.storage.apps_dir
    if apps_dir.exists():
        for app_file in apps_dir.glob("*.mvapp"):
            try:
                manager.load_app(app_file)
            except Exception as e:
                logger.warning(f"Failed to auto-load {app_file}: {e}")
    
    logger.info("MVAS Server started")
    yield
    
    # Cleanup
    logger.info("Shutting down MVAS Server...")
    manager.shutdown()
    logger.info("MVAS Server stopped")


# Create FastAPI app
app = FastAPI(
    title="MVAS - Machine Vision Application Standard",
    description="Plug-and-play machine vision application runtime",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global visualizer
visualizer = Visualizer()


# ============================================================================
# Request/Response Models
# ============================================================================

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class SuccessResponse(BaseModel):
    success: bool
    message: str


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "name": "MVAS Runtime",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", tags=["Health"])
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/v1/status", response_model=SystemStatus, tags=["System"])
async def get_status():
    """Get system status"""
    manager = get_app_manager()
    status = manager.get_status()
    return SystemStatus(**status)


# ============================================================================
# Application Management Endpoints
# ============================================================================

@app.post("/api/v1/apps/load", response_model=LoadAppResponse, tags=["Applications"])
async def load_app(request: LoadAppRequest):
    """Load a .mvapp application"""
    manager = get_app_manager()
    
    try:
        app_instance = manager.load_app(request.app_path)
        return LoadAppResponse(
            success=True,
            app_id=app_instance.app_id,
            app_info=app_instance.info,
            message=f"Application loaded successfully"
        )
    except Exception as e:
        logger.exception(f"Failed to load app: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/v1/apps/{app_id}", tags=["Applications"])
async def unload_app(app_id: str):
    """Unload an application"""
    manager = get_app_manager()
    
    if manager.unload_app(app_id):
        return {"success": True, "message": f"Application {app_id} unloaded"}
    else:
        raise HTTPException(status_code=404, detail=f"Application not found: {app_id}")


@app.get("/api/v1/apps", response_model=List[AppInfo], tags=["Applications"])
async def list_apps():
    """List all loaded applications"""
    manager = get_app_manager()
    return manager.list_apps()


@app.get("/api/v1/apps/{app_id}", response_model=AppInfo, tags=["Applications"])
async def get_app(app_id: str):
    """Get application details"""
    manager = get_app_manager()
    app = manager.get_app(app_id)
    
    if app is None:
        raise HTTPException(status_code=404, detail=f"Application not found: {app_id}")
    
    return app.info


@app.get("/api/v1/apps/{app_id}/stats", tags=["Applications"])
async def get_app_stats(app_id: str):
    """Get application statistics"""
    manager = get_app_manager()
    stats = manager.get_app_stats(app_id)
    
    if stats is None:
        raise HTTPException(status_code=404, detail=f"Application not found: {app_id}")
    
    return stats


# ============================================================================
# Inspection Endpoints
# ============================================================================

@app.post("/api/v1/inspect", response_model=InspectResponse, tags=["Inspection"])
async def inspect(request: InspectRequest):
    """Run inspection on an image"""
    manager = get_app_manager()
    
    result = manager.inspect(
        app_id=request.app_id,
        image_base64=request.image_base64,
        image_path=request.image_path,
    )
    
    if result.decision == DecisionResult.ERROR:
        raise HTTPException(
            status_code=400,
            detail=result.details.get("error", "Inspection failed")
        )
    
    return InspectResponse(
        request_id=result.request_id,
        decision=result.decision,
        confidence=result.confidence,
        anomaly_score=result.anomaly_score,
        inference_time_ms=result.inference_time_ms,
        total_time_ms=result.total_time_ms,
        details=result.details,
    )


@app.post("/api/v1/inspect/upload", tags=["Inspection"])
async def inspect_upload(
    app_id: str = Query(..., description="Application ID"),
    file: UploadFile = File(..., description="Image file"),
    visualize: bool = Query(False, description="Return visualization")
):
    """Run inspection on uploaded image file"""
    manager = get_app_manager()
    
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Run inspection
    app = manager.get_app(app_id)
    if app is None:
        raise HTTPException(status_code=404, detail=f"Application not found: {app_id}")
    
    result = manager.inspect(app_id=app_id, image=image)
    
    response = {
        "request_id": result.request_id,
        "decision": result.decision.value,
        "confidence": result.confidence,
        "anomaly_score": result.anomaly_score,
        "inference_time_ms": result.inference_time_ms,
        "total_time_ms": result.total_time_ms,
    }
    
    # Add visualization if requested
    if visualize:
        # Get anomaly map from inference
        app_instance = manager.get_app(app_id)
        viz = visualizer.create_result_visualization(
            image=image,
            decision=result.decision.value,
            anomaly_score=result.anomaly_score,
        )
        _, buffer = cv2.imencode(".jpg", viz)
        response["visualization_base64"] = base64.b64encode(buffer).decode("utf-8")
    
    return response


@app.post("/api/v1/inspect/batch", tags=["Inspection"])
async def inspect_batch(
    app_id: str = Query(..., description="Application ID"),
    files: List[UploadFile] = File(..., description="Image files"),
):
    """Run batch inspection on multiple images"""
    manager = get_app_manager()
    
    results = []
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is not None:
            result = manager.inspect(app_id=app_id, image=image)
            results.append({
                "filename": file.filename,
                "decision": result.decision.value,
                "anomaly_score": result.anomaly_score,
                "confidence": result.confidence,
            })
        else:
            results.append({
                "filename": file.filename,
                "error": "Invalid image file"
            })
    
    return {"results": results}


# ============================================================================
# Camera Management Endpoints
# ============================================================================

@app.post("/api/v1/cameras/connect", response_model=CameraInfo, tags=["Cameras"])
async def connect_camera(request: ConnectCameraRequest):
    """Connect to a camera or input source"""
    manager = get_app_manager()
    
    try:
        camera_info = manager.connect_camera(
            camera_type=request.camera_type,
            address=request.address,
            name=request.name,
            config=request.settings,
        )
        return camera_info
    except Exception as e:
        logger.exception(f"Failed to connect camera: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/v1/cameras/{camera_id}", tags=["Cameras"])
async def disconnect_camera(camera_id: str):
    """Disconnect a camera"""
    manager = get_app_manager()
    
    if manager.disconnect_camera(camera_id):
        return {"success": True, "message": f"Camera {camera_id} disconnected"}
    else:
        raise HTTPException(status_code=404, detail=f"Camera not found: {camera_id}")


@app.get("/api/v1/cameras", response_model=List[CameraInfo], tags=["Cameras"])
async def list_cameras():
    """List all connected cameras"""
    manager = get_app_manager()
    return manager.list_cameras()


@app.get("/api/v1/cameras/{camera_id}/grab", tags=["Cameras"])
async def grab_frame(camera_id: str, format: str = "jpeg"):
    """Grab a single frame from camera"""
    manager = get_app_manager()
    
    frame = manager.grab_frame(camera_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="Camera not found or grab failed")
    
    # Encode frame
    if format.lower() == "png":
        _, buffer = cv2.imencode(".png", frame)
        media_type = "image/png"
    else:
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        media_type = "image/jpeg"
    
    return StreamingResponse(
        iter([buffer.tobytes()]),
        media_type=media_type
    )


@app.post("/api/v1/cameras/{camera_id}/inspect", tags=["Cameras"])
async def inspect_from_camera(
    camera_id: str,
    app_id: str = Query(..., description="Application ID"),
):
    """Grab frame from camera and run inspection"""
    manager = get_app_manager()
    
    result = manager.inspect(app_id=app_id, camera_id=camera_id)
    
    return {
        "request_id": result.request_id,
        "decision": result.decision.value,
        "confidence": result.confidence,
        "anomaly_score": result.anomaly_score,
        "inference_time_ms": result.inference_time_ms,
    }


# ============================================================================
# WebSocket Streaming
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.streaming_tasks:
            self.streaming_tasks[session_id].cancel()
            del self.streaming_tasks[session_id]
    
    async def send_json(self, session_id: str, data: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(data)


ws_manager = ConnectionManager()


@app.websocket("/ws/stream/{session_id}")
async def websocket_stream(
    websocket: WebSocket,
    session_id: str,
):
    """
    WebSocket endpoint for live streaming inspection.
    
    Protocol:
    - Client sends: { "type": "config", "app_id": "...", "camera_id": "..." }
    - Server sends: { "type": "frame", "result": {...}, "image_base64": "..." }
    """
    await ws_manager.connect(websocket, session_id)
    manager = get_app_manager()
    
    app_id = None
    camera_id = None
    streaming = False
    fps = 10
    visualize = True
    
    try:
        while True:
            try:
                # Receive message with timeout
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=0.1
                )
                
                msg_type = data.get("type")
                
                if msg_type == "config":
                    app_id = data.get("app_id")
                    camera_id = data.get("camera_id")
                    fps = data.get("fps", 10)
                    visualize = data.get("visualize", True)
                    streaming = True
                    
                    await websocket.send_json({
                        "type": "status",
                        "message": "Streaming configured",
                        "app_id": app_id,
                        "camera_id": camera_id,
                    })
                
                elif msg_type == "pause":
                    streaming = False
                    await websocket.send_json({"type": "status", "message": "Paused"})
                
                elif msg_type == "resume":
                    streaming = True
                    await websocket.send_json({"type": "status", "message": "Resumed"})
                
                elif msg_type == "stop":
                    break
                    
            except asyncio.TimeoutError:
                pass  # No message received, continue streaming
            
            # Stream frames
            if streaming and app_id and camera_id:
                frame = manager.grab_frame(camera_id)
                
                if frame is not None:
                    # Run inspection
                    result = manager.inspect(app_id=app_id, image=frame)
                    
                    # Create response
                    response = {
                        "type": "frame",
                        "timestamp": datetime.now().isoformat(),
                        "result": {
                            "decision": result.decision.value,
                            "confidence": result.confidence,
                            "anomaly_score": result.anomaly_score,
                            "inference_time_ms": result.inference_time_ms,
                        }
                    }
                    
                    # Add visualization
                    if visualize:
                        viz = visualizer.create_result_visualization(
                            image=frame,
                            decision=result.decision.value,
                            anomaly_score=result.anomaly_score,
                        )
                        _, buffer = cv2.imencode(".jpg", viz, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        response["image_base64"] = base64.b64encode(buffer).decode("utf-8")
                    
                    await websocket.send_json(response)
                
                # Control frame rate
                await asyncio.sleep(1.0 / fps)
            else:
                await asyncio.sleep(0.1)
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
    finally:
        ws_manager.disconnect(session_id)


# ============================================================================
# Static Files & UI
# ============================================================================

@app.get("/api/v1/apps/{app_id}/icon", tags=["Applications"])
async def get_app_icon(app_id: str):
    """Get application icon"""
    manager = get_app_manager()
    app = manager.get_app(app_id)
    
    if app is None:
        raise HTTPException(status_code=404, detail="Application not found")
    
    # Check for icon in app directory
    icon_path = app.app_dir / "ui" / "icon.png"
    if icon_path.exists():
        return StreamingResponse(
            open(icon_path, "rb"),
            media_type="image/png"
        )
    
    raise HTTPException(status_code=404, detail="Icon not found")


# ============================================================================
# Run Server
# ============================================================================

def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
):
    """Run the MVAS server"""
    import uvicorn
    
    uvicorn.run(
        "mvas_runtime.server:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    run_server()

