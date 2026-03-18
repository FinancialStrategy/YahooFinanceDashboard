from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
INDEX_FILE = PUBLIC_DIR / "index.html"
if not INDEX_FILE.exists():
    INDEX_FILE = BASE_DIR / "index.html"
UNIVERSE_FILE = BASE_DIR / "universe.json"

app = FastAPI(title="Institutional Portfolio Dashboard")

def load_universe():
    with open(UNIVERSE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_universe(universe_dict):
    rows = []
    for group in universe_dict.get("groups", []):
        family = group.get("family", "Other")
        subgroup = group.get("name", "Unknown")
        color = group.get("color", "#64748b")
        for item in group.get("items", []):
            rows.append({
                "family": family,
                "group": subgroup,
                "group_color": color,
                "symbol": item.get("symbol"),
                "label": item.get("label"),
                "target": float(item.get("target", 0.0))
            })
    return rows

def universe_metadata():
    u = load_universe()
    families = sorted({g.get("family", "Other") for g in u.get("groups", [])})
    groups_by_family = {}
    for g in u.get("groups", []):
        fam = g.get("family", "Other")
        groups_by_family.setdefault(fam, [])
        groups_by_family[fam].append({
            "name": g.get("name", "Unknown"),
            "color": g.get("color", "#64748b"),
            "count": len(g.get("items", []))
        })
    for fam in groups_by_family:
        groups_by_family[fam] = sorted(groups_by_family[fam], key=lambda x: x["name"])
    return {"families": families, "groups_by_family": groups_by_family, "groups": u.get("groups", [])}

@app.get("/")
def root():
    return FileResponse(INDEX_FILE)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/api/universe")
def api_universe():
    return JSONResponse(content=universe_metadata())

@app.get("/api/snapshot")
def api_snapshot():
    items = flatten_universe(load_universe())
    rows = []
    for x in items:
        rows.append({
            "family": x["family"],
            "group": x["group"],
            "group_color": x["group_color"],
            "symbol": x["symbol"],
            "label": x["label"],
            "current_weight": round(x["target"] * 100.0, 4),
            "included": x["family"] != "World Indices",
            "status": "Included" if x["family"] != "World Indices" else "Reference"
        })
    return JSONResponse(content={"rows": rows})

@app.post("/api/construct")
def api_construct(payload: dict = Body(...)):
    rows = payload.get("rows", [])
    use_rows = [r for r in rows if r.get("included", True)]
    if len(use_rows) < 2:
        raise HTTPException(status_code=400, detail=f"At least 2 included assets are required for construction. Current filtered universe has {len(use_rows)}.")
    total_weight = sum(max(float(r.get("current_weight", 0.0)), 0.0) for r in use_rows)
    if total_weight <= 0:
        raise HTTPException(status_code=400, detail="Weights sum to zero in the filtered universe.")
    weights = []
    for r in use_rows:
        w = max(float(r.get("current_weight", 0.0)), 0.0) / total_weight
        weights.append({
            "family": r.get("family", "Other"),
            "group": r.get("group", "Unknown"),
            "symbol": r.get("symbol"),
            "label": r.get("label"),
            "prior_weight": w,
            "weight": w
        })
    return JSONResponse(content={
        "used_symbols": [r["symbol"] for r in use_rows],
        "weights": weights,
        "message": "Construction universe validated successfully."
    })

@app.get("/api/major-world-indices")
def api_major_world_indices():
    rows = [r for r in flatten_universe(load_universe()) if r["family"] == "World Indices"]
    return JSONResponse(content={"rows": rows})

@app.get("/api/futures-dashboard")
def api_futures_dashboard():
    rows = [r for r in flatten_universe(load_universe()) if r["family"] == "Futures"]
    return JSONResponse(content={"rows": rows})