# Backend

FastAPI služba pro XDENT AI asistenta.

## Hlavní endpointy

- `GET /health` - kontrola služby
- `GET /api/stats` - stav Qdrant indexu a pokrytí témat
- `GET /api/pricing` - modely a orientační ceny z konfigurace
- `POST /api/ingest` - načtení transkripcí, anotace chunků, embeddingy a zápis do Qdrantu
- `POST /api/chat` - odpověď asistenta
- `POST /api/evaluate` - evaluace nad sadou dotazů

## Nova RAG Q&A vrstva

Ingest uklada do stejne Qdrant kolekce dva typy znalosti:

- transkripcni chunky s metadaty `doc_type=transcript`
- predpripravene otazky a odpovedi z `app/seed_qa.py` s `doc_type=qa_seed`

Chat nejdriv hleda v Q&A vrstve a potom v transkripcich. Pokud nenajde dost jistou odpoved, OpenAI vytvori kratky opatrny navrh, backend ho ulozi jako `doc_type=qa_generated` a dalsi podobny dotaz ho uz najde ve vektorove databazi. Kdyz se nepodari vytvorit ani navrh, odpoved preda lidskemu operatorovi.

Odpoved vraci zdroje, jistotu odpovedi, `chunks_considered` a `chunks_used`.

## Lokální spuštění bez Dockeru

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
$env:OPENAI_API_KEY="..."
$env:QDRANT_URL="http://localhost:6333"
uvicorn app.main:app --app-dir backend --reload
```
