"""
Start a ChromaDB server for the RAG example.

Install & run:
    pip install "chromadb[server]"
    python rag-example/run_chroma.py

Starts ChromaDB on http://localhost:8000 with persistent storage.
"""

import sys
import os
import subprocess


def ensure_deps():
    """Ensure chromadb + server dependencies are installed."""
    try:
        import chromadb
        import fastapi
        import uvicorn
        import opentelemetry.instrumentation.fastapi
        print(f"ChromaDB v{chromadb.__version__} (server deps OK)")
        return
    except ImportError as e:
        missing = str(e)
        print(f"Missing dependency: {missing}")
        print("Installing dependencies...\n")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install",
             "chromadb[server]",
             "opentelemetry-instrumentation-fastapi"],
        )
        print()


def main():
    ensure_deps()

    import uvicorn
    import chromadb

    script_dir = os.path.dirname(os.path.abspath(__file__))
    persist_dir = os.path.join(script_dir, "chroma_data")
    os.makedirs(persist_dir, exist_ok=True)

    host = "0.0.0.0"
    port = 8000

    # Set env vars BEFORE importing the app
    os.environ["IS_PERSISTENT"] = "TRUE"
    os.environ["PERSIST_DIRECTORY"] = persist_dir
    os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
    os.environ["CHROMA_SERVER_HOST"] = host
    os.environ["CHROMA_SERVER_HTTP_PORT"] = str(port)

    print()
    print(f"  ChromaDB       : v{chromadb.__version__}")
    print(f"  Data directory : {persist_dir}")
    print(f"  Listening on   : http://localhost:{port}")
    print()
    print("Press Ctrl+C to stop.\n")

    # Build the ASGI app via ChromaDB's own FastAPI wrapper
    from chromadb.config import Settings
    from chromadb.server.fastapi import FastAPI as ChromaFastAPI

    settings = Settings(
        chroma_server_host=host,
        chroma_server_http_port=port,
        is_persistent=True,
        persist_directory=persist_dir,
        anonymized_telemetry=False,
    )

    server = ChromaFastAPI(settings)
    app = server.app()

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
