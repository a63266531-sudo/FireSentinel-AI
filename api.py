"""Read-only REST API (FastAPI).   pip install -r requirements-api.txt
   uvicorn firesentinel.api:app --host 127.0.0.1 --port 8000      ->  http://127.0.0.1:8000/docs

Safe defaults: bind to localhost; without FIRESENTINEL_API_KEY only loopback clients are served; with a key, every request must
send the header `X-API-Key` (constant-time comparison); no CORS; small in-memory rate limit; no write/upload endpoints.
If you expose it beyond localhost, put it behind HTTPS (reverse proxy) and set the key."""
import hmac
import os
import time
from collections import defaultdict, deque

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from . import __version__, service
from .config import REGIONS

RATE_LIMIT_PER_MIN = int(os.environ.get("FIRESENTINEL_RATE_LIMIT", "60"))
_hits = defaultdict(deque)
LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}

app = FastAPI(title="FireSentinel AI API", version=__version__,
              description="Unconfirmed satellite detections and uncertainty-aware modelled risk. Research tool, not an emergency service.")


def guard(request: Request):
    key = os.environ.get("FIRESENTINEL_API_KEY")
    host = request.client.host if request.client else ""
    if key:
        if not hmac.compare_digest(request.headers.get("x-api-key", ""), key):
            raise HTTPException(401, "missing or invalid X-API-Key")
    elif host not in LOOPBACK:
        raise HTTPException(403, "set FIRESENTINEL_API_KEY to allow non-local clients")
    q, now = _hits[host], time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(429, "rate limit exceeded")
    q.append(now)


@app.exception_handler(service.ServiceError)
async def _service_error(_, e: service.ServiceError):
    return JSONResponse(status_code=e.status, content={"error": e.message})


def _region(region: str = Query("Egypt", description=f"One of {list(REGIONS)}")):
    return region


@app.get("/health")
def health():
    return service.health()


@app.get("/ready", dependencies=[Depends(guard)])
def ready():
    r = service.readiness()
    return JSONResponse(status_code=200 if r["ready"] else 503, content=r)


@app.get("/sources", dependencies=[Depends(guard)])
def sources():
    return service.sources_status()


@app.get("/model", dependencies=[Depends(guard)])
def model_meta():
    return service.model_metadata()


@app.get("/validate", dependencies=[Depends(guard)])
def validate(region: str = Depends(_region), days: int = Query(7, ge=1, le=service.MAX_DAYS)):
    return service.validate_region(region, days)


@app.get("/risk", dependencies=[Depends(guard)])
def risk(region: str = Depends(_region), weeks: int = Query(26, ge=12, le=service.MAX_WEEKS),
         weather: str = Query("open-meteo", pattern="^(open-meteo|synthetic)$"), max_cells: int = Query(100, ge=10, le=service.MAX_CELLS)):
    return service.risk(region, weeks, weather, max_cells=max_cells)


@app.get("/export/geojson", dependencies=[Depends(guard)])
def export_geojson(region: str = Depends(_region), days: int = Query(3, ge=1, le=service.MAX_DAYS),
                   kind: str = Query("detections", pattern="^(detections|clusters)$")):
    return service.geojson(region, days, kind)


@app.get("/experiments", dependencies=[Depends(guard)])
def experiments_list():
    return service.list_experiments()


@app.get("/experiments/{eid}", dependencies=[Depends(guard)])
def experiment(eid: str):
    return service.get_experiment(eid)
