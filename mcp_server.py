import asyncio
import os
import json
import base64
import subprocess
from pathlib import Path
from typing import Any, List, Optional
import mcp.types as types
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

import logging
import cairosvg

# Configuration
BASE_DIR = Path('/app/data/ai-stitching-model')
PROGRESS_FILE = BASE_DIR / 'progress_pipeline.json'
UPLOAD_DIR = BASE_DIR / 'dataset/sems/manual'
RESULT_SVG = BASE_DIR / 'manual_panorama.svg'
LOG_FILE = BASE_DIR / 'mcp_server.log'
PIPELINE_LOG_FILE = BASE_DIR / 'pipeline.log'

# Setup Logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_server")

app = Server("ai-stitching-pipeline")

@app.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri="file:///app/data/ai-stitching-model/manual_panorama.gds",
            name="Final GDS Output",
            description="The final stitched GDSII file. Binary format.",
            mimeType="application/octet-stream"
        ),
        types.Resource(
            uri="file:///app/data/ai-stitching-model/manual_panorama.svg",
            name="Final SVG Output",
            description="The final stitched SVG file. Vector format.",
            mimeType="image/svg+xml"
        )
    ]

@app.read_resource()
async def handle_read_resource(uri: Any) -> str | bytes:
    uri_str = str(uri)
    if uri_str == "file:///app/data/ai-stitching-model/manual_panorama.gds":
        return (BASE_DIR / 'manual_panorama.gds').read_bytes()
    elif uri_str == "file:///app/data/ai-stitching-model/manual_panorama.svg":
        return (BASE_DIR / 'manual_panorama.svg').read_text()
    raise ValueError(f"Unknown resource: {uri_str}")

@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="start_stitching",
            description="Start the stitching pipeline (Segmentation -> Vectorization -> Stitching -> GDS). Ensure files are in dataset/sems/manual or already uploaded.",
            inputSchema={
                "type": "object",
                "properties": {
                    "clean_start": {
                        "type": "boolean",
                        "description": "Whether to clean up previous results before starting. Default is true.",
                        "default": True
                    }
                }
            }
        ),
        types.Tool(
            name="get_stitching_status",
            description="Get the current status of the stitching pipeline, including stage, percentage, and GPU details.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),

        types.Tool(
            name="get_logs",
            description="Retrieve logs from the MCP server or the stitching pipeline.",
            inputSchema={
                "type": "object",
                "properties": {
                    "log_type": {
                        "type": "string",
                        "enum": ["server", "pipeline"],
                        "description": "Which log to retrieve. 'server' for MCP request logs, 'pipeline' for stitching outputs.",
                        "default": "pipeline"
                    },
                    "lines": {
                        "type": "integer",
                        "description": "Number of last lines to retrieve.",
                        "default": 50
                    }
                }
            }
        ),
        types.Tool(
            name="monitor_pipeline",
            description="Waits until the stitching pipeline is complete. Use this to get notified when the job is done without constantly checking status. Returns the final result summary.",
            inputSchema={
                "type": "object",
                "properties": {
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Maximum time to wait in seconds.",
                        "default": 300
                    },
                    "poll_interval": {
                        "type": "integer",
                        "description": "Time to sleep between checks.",
                        "default": 5
                    }
                }
            }
        ),
        types.Tool(
            name="run_full_workflow",
            description="Executes the entire workflow in one go: Starts the pipeline, waits for completion, and returns a simple text summary. Does NOT return images.",
            inputSchema={
                "type": "object",
                "properties": {
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Timeout for the entire process in seconds.",
                        "default": 600
                    },
                    "clean_start": {
                         "type": "boolean",
                         "default": True
                    }
                }
            }
        ),


        types.Tool(
            name="get_vector_preview",
            description="Get the vectorized SVG for a specific file (before stitching). Useful for verifying vectorization quality.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                         "type": "string",
                         "description": "Filename of the vector file (e.g. 'sem0001.svg'). If omitted, lists available files."
                    }
                }
            }
        ),
        types.Tool(
            name="upload_source_image",
            description="Upload a source SEM image (IC Chip image) to the server for processing. Call this before start_stitching if the user provides a new image.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file (e.g., 'image_01.png')."
                    },
                    "content_base64": {
                        "type": "string",
                        "description": "Base64 encoded content of the image file."
                    }
                },
                "required": ["filename", "content_base64"]
            }
        )
    ]

@app.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    
    if name == "start_stitching":
        clean_start = (arguments or {}).get("clean_start", True)
        
        if clean_start:
            # Cleanup logic (similar to server.py)
            mask_dir = BASE_DIR / 'dataset/masks/manual'
            vector_dir = BASE_DIR / 'dataset/vectorized_manual'
            for d in [mask_dir, vector_dir]:
                if d.exists():
                    for f in d.glob('*'):
                        try: 
                            if f.is_file(): f.unlink()
                        except: pass
            
            for f in BASE_DIR.glob('progress_*.json'):
                try: f.unlink()
                except: pass
                
            # AUTO-COPY DISABLED (User Request)
            # We explicitly rely on files already present in dataset/sems/manual
            # (uploaded via upload_source_image or manually placed)

            # Initialize status
            with open(PROGRESS_FILE, 'w') as f:
                json.dump({
                    "stage": "Ready", 
                    "status": "Initializing pipeline...", 
                    "percent": 0
                }, f)

        # Start Pipeline
        logger.info(f"Starting pipeline with clean_start={clean_start}")
        
        # Build command string
        cmd_str = (
            f"python3 run_stitching.py --vectorize --show-labels --show-borders --use-manual-dir "
            f"> {PIPELINE_LOG_FILE} 2>&1"
        )
        
        # Run in background via shell
        try:
            subprocess.Popen(cmd_str, shell=True, cwd=str(BASE_DIR), executable='/bin/bash')
            logger.info("Pipeline process spawned successfully via shell")
        except Exception as e:
            logger.error(f"Failed to spawn pipeline: {e}")
            return [types.TextContent(type="text", text=f"Error starting pipeline: {e}")]
        
        return [
            types.TextContent(
                type="text",
                text="Pipeline started successfully. Logs are being written to pipeline.log. Use get_logs to view them."
            )
        ]


    elif name == "get_logs":
        log_type = (arguments or {}).get("log_type", "pipeline")
        num_lines = (arguments or {}).get("lines", 50)
        
        target_file = PIPELINE_LOG_FILE if log_type == "pipeline" else LOG_FILE
        
        if not target_file.exists():
            return [types.TextContent(type="text", text=f"Log file {target_file.name} does not exist yet.")]
            
        try:
            # Read last N lines (simple implementation)
            lines = target_file.read_text(encoding='utf-8').splitlines()
            last_n = lines[-num_lines:] if num_lines > 0 else lines
            return [types.TextContent(type="text", text="\n".join(last_n))]       
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error reading logs: {str(e)}")]

    elif name == "monitor_pipeline":
        timeout = (arguments or {}).get("timeout_seconds", 300)
        interval = (arguments or {}).get("poll_interval", 5)
        
        start_time = asyncio.get_event_loop().time()
        
        while True:
            # Check timeout
            if asyncio.get_event_loop().time() - start_time > timeout:
                 return [types.TextContent(type="text", text=f"Monitor timed out after {timeout} seconds. Pipeline may still be running.")]
            
            # Check status
            status_text = "Unknown"
            percent = 0
            if PROGRESS_FILE.exists():
                try:
                    with open(PROGRESS_FILE, 'r') as f:
                        data = json.load(f)
                        status_text = data.get("status", "")
                        stage = data.get("stage", "")
                        percent = data.get("percent", 0)
                        
                        if stage == "Complete":
                             return [types.TextContent(type="text", text=f"Pipeline Completed Successfully!")]
                        if "Error" in stage or "Error" in status_text:
                             return [types.TextContent(type="text", text=f"Pipeline Error: {status_text}")]
                except:
                    pass
            


    elif name == "get_vector_preview":
        filename = (arguments or {}).get("filename")
        vector_dir = BASE_DIR / 'dataset/vectorized_manual'
        
        if not filename:
            # List files
            files = sorted([f.name for f in vector_dir.glob("*.svg")]) if vector_dir.exists() else []
            if not files:
                return [types.TextContent(type="text", text="No vector files found (Vectorization stage may not be complete).")]
            return [types.TextContent(type="text", text="Available vector files:\n" + "\n".join(files) + "\n\nSpecify filename to view.")]
            
        target_file = vector_dir / filename
        if not target_file.exists():
             return [types.TextContent(type="text", text=f"Vector file {filename} not found.")]
             
        try:
             # Convert to PNG using cairosvg
             png_data = cairosvg.svg2png(url=str(target_file))
             png_b64 = base64.b64encode(png_data).decode('ascii')
             return [
                 types.ImageContent(type="image", data=png_b64, mimeType="image/png"),
                 types.TextContent(type="text", text=f"Vector preview for {filename}")
             ]
        except Exception as e:
             return [types.TextContent(type="text", text=f"Error rendering vector: {e}")]

    elif name == "run_full_workflow":
        timeout = (arguments or {}).get("timeout_seconds", 600)
        clean_start = (arguments or {}).get("clean_start", True)
        
        logger.info(f"Full Workflow: Request received (clean={clean_start})")
        
        try:
            # 1. Start Pipeline
            logger.info("Full Workflow: Starting pipeline sequence...")
            if clean_start:
                 # Cleanup
                mask_dir = BASE_DIR / 'dataset/masks/manual'
                vector_dir = BASE_DIR / 'dataset/vectorized_manual'
                for d in [mask_dir, vector_dir]:
                    if d.exists():
                        for f in d.glob('*'):
                            try: 
                                if f.is_file(): f.unlink()
                            except: pass
                for f in BASE_DIR.glob('progress_*.json'):
                    try: f.unlink()
                    except: pass
                
                # AUTO-COPY DISABLED (User Request)
                # We explicitly rely on files already present in dataset/sems/manual

                with open(PROGRESS_FILE, 'w') as f:
                    json.dump({"stage": "Ready", "status": "Initializing...", "percent": 0}, f)

            cmd_str = (
                f"python3 run_stitching.py --vectorize --show-labels --show-borders --use-manual-dir "
                f"> {PIPELINE_LOG_FILE} 2>&1"
            )
            try:
                subprocess.Popen(cmd_str, shell=True, cwd=str(BASE_DIR), executable='/bin/bash')
                logger.info("Pipeline process spawned successfully via shell")
            except Exception as e:
                logger.error(f"Failed to spawn pipeline: {e}")
                return [types.TextContent(type="text", text=f"Error starting: {e}")]
                
        except Exception as e:
             logger.error(f"Critical error in run_full_workflow setup: {e}", exc_info=True)
             return [types.TextContent(type="text", text=f"Critical server error: {e}")]

        # 2. Wait for Completion
        start_time = asyncio.get_event_loop().time()
        final_state = "Unknown"
        while True:
            if asyncio.get_event_loop().time() - start_time > timeout:
                return [types.TextContent(type="text", text=f"Workflow timed out after {timeout}s")]
            
            if PROGRESS_FILE.exists():
                try:
                    with open(PROGRESS_FILE, 'r') as f:
                        data = json.load(f)
                        stage = data.get("stage", "")
                        if stage == "Complete":
                            final_state = "Complete"
                            break
                        if "Error" in stage:
                            return [types.TextContent(type="text", text=f"Pipeline Error: {data.get('status')}")]
                except: pass
            await asyncio.sleep(3)
            
        # 3. Return Results References (Avoid Large Blobs)
        results = []
        results.append(types.TextContent(type="text", text="Pipeline Completed Successfully!"))
        
        # GDS File Reference
        gds_path = BASE_DIR / 'manual_panorama.gds'
        if gds_path.exists():
            results.append(types.TextContent(
                type="text", 
                text=f"The finalized GDS file ({gds_path.stat().st_size / 1024 / 1024:.2f} MB) is available.\n"
                     f"To download, please read the resource: file:///app/data/ai-stitching-model/manual_panorama.gds"
            ))
        else:
            results.append(types.TextContent(type="text", text="Warning: GDS file was not found."))

        # Summary (Simple text)
        summary_text = "Pipeline Output Summary:\n"
        if gds_path.exists():
             summary_text += f"- GDS File: {gds_path.name} ({gds_path.stat().st_size / 1024 / 1024:.2f} MB)\n"
        
        svg_path = BASE_DIR / 'manual_panorama.svg'
        if svg_path.exists():
             summary_text += f"- SVG File: {svg_path.name} ({svg_path.stat().st_size / 1024 / 1024:.2f} MB)\n"
             
        summary_text += "\nProcess finished. Please check the results."
        results.append(types.TextContent(type="text", text=summary_text))
        
        return results

    elif name == "get_stitching_status":
        status_info = {
            "pipeline": "Idle",
            "details": {}
        }
        
        if PROGRESS_FILE.exists():
            try:
                with open(PROGRESS_FILE, 'r') as f:
                    pipeline_data = json.load(f)
                    status_info["pipeline"] = pipeline_data
            except:
                pass
        
        # GPU Status
        gpus = []
        for i in range(2):
            gpu_file = BASE_DIR / f'progress_gpu_{i}.json'
            if gpu_file.exists():
                try:
                    with open(gpu_file, 'r') as f:
                        gpu_data = json.load(f)
                        gpus.append({f"gpu_{i}": gpu_data})
                except:
                    pass
        
        status_info["details"]["gpus"] = gpus
        
        return [
            types.TextContent(
                type="text",
                text=json.dumps(status_info, indent=2)
            )
        ]


    elif name == "upload_source_image":
        filename = (arguments or {}).get("filename")
        content_b64 = (arguments or {}).get("content_base64")
        
        if not filename or not content_b64:
            return [types.TextContent(type="text", text="Error: filename and content_base64 are required.")]
            
        try:
            # Ensure upload directory exists
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            
            # Save file
            file_path = UPLOAD_DIR / filename
            # Clean base64 string
            original_len = len(content_b64)
            if ',' in content_b64:
                parts = content_b64.split(',')
                if len(parts) > 1:
                     content_b64 = parts[1]
            
            content_b64 = content_b64.strip()
            
            # Fix invalid length (len % 4 == 1 is impossible in valid base64)
            if len(content_b64) % 4 == 1:
                content_b64 = content_b64[:-1]
            
            # Fix padding
            missing_padding = len(content_b64) % 4
            if missing_padding:
                content_b64 += '=' * (4 - missing_padding)
            
            decoded_data = base64.b64decode(content_b64, validate=False)
            
            if len(decoded_data) == 0:
                 return [types.TextContent(type="text", text=f"Error: Decoded content is empty (Original len: {original_len}). Check input.")]

            # Optional: Check PNG header (89 50 4E 47 0D 0A 1A 0A)
            if decoded_data[:8] != b'\x89PNG\r\n\x1a\n':
                 # Not a fatal error, but worth warning
                 logger.warning(f"Uploaded file {filename} does not start with PNG header.")

            with open(file_path, "wb") as f:
                f.write(decoded_data)
                
            return [types.TextContent(type="text", text=f"Successfully uploaded {filename} ({len(decoded_data)} bytes) to {file_path}")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error uploading file: {str(e)}")]

    raise ValueError(f"Unknown tool: {name}")

async def main():
    # Run the server using stdin/stdout streams
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="ai-stitching-pipeline",
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
