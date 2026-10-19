"""FastAPI app: the web page plus an API to upload documents and ask questions about them."""

from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field

import rag
from config import ALLOWED_EXTENSIONS, STATIC_DIR, TEMPLATES_DIR, UPLOAD_DIR, jev, logger, settings


def _uploaded_files() -> list[Path]:
    return [path for path in UPLOAD_DIR.iterdir() if path.suffix.lower() in ALLOWED_EXTENSIONS]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Indexes live in memory, so rebuild them from the files uploaded before the restart
    for path in _uploaded_files():
        rag.add_files([path])
    logger.info("Ready with %d documents", len(rag.list_documents()))
    yield
    jev.close()


app = FastAPI(
    title="Jev Verified RAG",
    description="Upload documents and get answers checked by Jev",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2000)


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    # The traceback is still printed in the server log
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "extensions": sorted(ALLOWED_EXTENSIONS),
            "max_upload_mb": settings.max_upload_mb,
            "tau": settings.tau,
        },
    )


# Plain `def` (not `async def`) routes run in a worker thread, so slow indexing and API calls don't block the server
@app.post("/api/upload")
def upload(files: list[UploadFile]):
    indexed: dict[str, int] = {}
    skipped: dict[str, str] = {}
    existing = rag.list_documents()

    for file in files:
        name = Path(file.filename or "").name  # drop any folder part from the name
        if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
            skipped[name] = "unsupported file type"
        elif name in existing or name in indexed:
            skipped[name] = "already uploaded"
        elif file.size and file.size > settings.max_upload_mb * 1024 * 1024:
            skipped[name] = f"larger than {settings.max_upload_mb} MB"
        else:
            path = UPLOAD_DIR / name
            path.write_bytes(file.file.read())
            try:
                chunks = rag.add_files([path])
            except Exception:
                logger.exception("Could not read %s", name)
                chunks = 0
            if chunks:
                indexed[name] = chunks
            else:
                path.unlink()
                skipped[name] = "no readable text"

    return {"indexed": indexed, "skipped": skipped, "documents": rag.list_documents()}


@app.post("/api/ask")
def ask(body: AskRequest):
    return rag.ask(body.question)


@app.get("/api/documents")
def documents():
    return {"documents": rag.list_documents()}


@app.delete("/api/documents")
def clear_documents():
    rag.reset()
    for path in _uploaded_files():
        path.unlink()
    return {"documents": {}}


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(rag.list_documents())}


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
