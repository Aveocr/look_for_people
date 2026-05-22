#!/usr/bin/env python3
"""Test routes directly."""
import asyncio
from fastapi.testclient import TestClient
from web_app import app

client = TestClient(app)
response = client.get("/")
print(f"Status: {response.status_code}")
print(f"Content (first 500 chars):\n{response.text[:500]}")
if response.status_code != 200:
    print(f"\nFull text:\n{response.text}")
