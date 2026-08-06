# ScholarSync 2.0

ScholarSync is an evidence-grounded Django research workspace. This rebuild replaces the original single-user Streamlit prototype with a dual-mode architecture:

- **Public demo:** packaged PDF chunks, no account, no Supabase dependency.
- **Private workspace:** authenticated workspaces, PDF upload, page-aware indexing, grounded conversations, citations, notes, and PDF conversation export.

## Implemented in this milestone

- Plum-stone responsive homepage and research interface
- “Built by Md Noor” linked homepage/footer credit
- Django authentication and user-owned workspaces
- PDF validation, SHA-256 duplicate prevention, local/Supabase storage abstraction
- PyMuPDF extraction and page-aware chunking
- BM25-style local retrieval with a Groq-compatible generator and transparent retrieval-only fallback
- Page citations and source evidence panel
- Public demo using three foundational sequence-modeling and attention papers
- Conversation PDF export with ReportLab
- Ownership checks and starter tests
- Render, Docker, WhiteNoise, and Supabase-ready configuration

## Next milestone

- Add PostgreSQL `pgvector` embeddings and hybrid rank fusion
- Add asynchronous processing after resource testing
- Add paper comparison and note creation UI
- Add claim-level citation verification and evaluation dashboard
- Add PDF.js embedded page viewer/highlighting

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements/development.txt
cp .env.example .env
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Open `http://127.0.0.1:8000/` and use `/demo/` without an account.

## Production

Set `DJANGO_SETTINGS_MODULE=config.settings.production`, configure `DATABASE_URL`, and choose:

- `DOCUMENT_STORAGE_BACKEND=local` for development only
- `DOCUMENT_STORAGE_BACKEND=supabase` with the Supabase environment variables for production

The Render blueprint uses one Gunicorn worker to remain within a small free-instance memory budget.

## Legacy prototype

The original Streamlit implementation is retained in `legacy_streamlit/` for reference. Its old Chroma runtime data and caches were intentionally removed from the rebuilt repository.
