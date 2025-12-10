#!/bin/bash
# Wrapper script to launch MCP server with correct python path
LOGFILE="/app/data/ai-stitching-model/wrapper_launch.log"

# Log startup
echo "--- Starting MCP Server Wrapper at $(date) ---" >> "$LOGFILE"

# Add local site-packages to PYTHONPATH
export PYTHONPATH="/home/samsung/.local/lib/python3.10/site-packages:$PYTHONPATH"

# Run the server
# -u: Unbuffered output (crucial for MCP)
exec /usr/bin/python3 -u /app/data/ai-stitching-model/mcp_server.py 2>> "$LOGFILE"
