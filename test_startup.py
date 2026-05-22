#!/usr/bin/env python3
"""Test if web_app can start."""
import sys
import os

print("Python version:", sys.version)
print("CWD:", os.getcwd())
print("Files in CWD:", os.listdir("."))

# Test imports
try:
    print("\nTesting imports...")
    from dotenv import load_dotenv
    print("✓ dotenv")
    load_dotenv()
    
    from fastapi import FastAPI
    print("✓ fastapi")
    
    import search_target
    print("✓ search_target")
    
    print("\nTesting web_app import...")
    import web_app
    print("✓ web_app")
    
    print("\nAll imports OK!")
    print("Starting web app manually...")
    
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=False)
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
