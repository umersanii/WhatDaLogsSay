#!/usr/bin/env python3
"""
Log Analyzer — entry point.
Usage: python main.py [--port 8000]
Set GROQ in environment before running.
"""
import os
import sys
import argparse

def main():
    if not (os.environ.get("GROQ") or os.environ.get("ANTHROPIC_API_KEY")):
        print("ERROR: GROQ environment variable not set.")
        print("  Add your API key to .env file or set environment variable.")
        print("  See .env.example for format.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Log Analyzer Web App")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    import uvicorn
    print(f"\n  Log Analyzer running at http://localhost:{args.port}\n")
    uvicorn.run(
        "backend.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )

if __name__ == "__main__":
    main()
