import subprocess
import os

# This script will run your FastAPI app using uvicorn
# It's important that uvicorn runs on 0.0.0.0 and the correct port
# HF Spaces often expose port 7860 for Python apps
port = os.getenv("PORT", "7860") # Use environment variable if set, otherwise default
command = f"uvicorn main:app --host 0.0.0.0 --port {port}"
subprocess.run(command, shell=True)