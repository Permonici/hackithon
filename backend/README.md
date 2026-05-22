# Backend

FastAPI služba pro XDENT AI asistenta.

## Hlavní endpointy

- `GET /health` - kontrola služby
- `GET /api/stats` - stav Qdrant indexu a pokrytí témat
- `POST /api/ingest` - načtení transkripcí, anotace chunků, embeddingy a zápis do Qdrantu
- `POST /api/chat` - odpověď asistenta
- `POST /api/evaluate` - evaluace nad sadou dotazů

## Lokální spuštění bez Dockeru

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
$env:OPENAI_API_KEY="..."
$env:QDRANT_URL="http://localhost:6333"
uvicorn app.main:app --app-dir backend --reload
```
