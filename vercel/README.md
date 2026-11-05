# Paradigm on Vercel

The site is Next.js, and `/api` is a Python FastAPI function. Both deploy as one Vercel project.
Data lives in Neon Postgres, in the same tables the Streamlit app uses, so accounts and chats carry over.

## Run it locally

You need Python 3.12 or newer, Node 22 or newer, and an empty, throwaway Postgres database.
Never point local runs or tests at production.

```bash
pip install -r api/requirements.txt uvicorn httpx
npm install
```

In one terminal, start the API. Set `DATABASE_URL` and `GROQ_API_KEY` first.

```bash
cd api && uvicorn index:app --port 8000
```

In another, start the site, then open http://localhost:3000.

```bash
npm run dev
```

## Tests

```bash
DATABASE_URL=<throwaway database> python api/test_api.py
```

They create test accounts, so the database must be empty and disposable. GitHub runs them on every pull request.

## Deploy

1. In Vercel, import this GitHub repo and set **Root Directory** to `vercel`. `vercel.json` defines two services: `app` (the site, public at `/`) and `api` (FastAPI, public at `/api`). The API sees the full `/api/...` path.
2. Under **Settings → Environment Variables**, add `GROQ_API_KEY` and `DATABASE_URL`. Use the pooled Neon connection string.
3. Create the new tables once. This only adds tables and never changes existing data:

   ```bash
   DATABASE_URL=<that connection string> python api/_store.py init-db
   ```

4. Deploy, then test on the preview link before using a custom domain.

## Plans

Everyone starts on Free. To move someone to Paid:

```bash
DATABASE_URL=<connection string> python api/_store.py set-plan <username> paid
```

## Limits to know about

- Uploads are limited to 4 MB in total, because Vercel rejects larger request bodies.
- Document search matches words, using Postgres full-text search. It doesn't understand meaning like an AI search model would.
- Vercel's free Hobby plan is for personal, non-commercial use. Move to a paid plan before charging customers.
