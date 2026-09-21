# Workspace

## Overview

pnpm workspace monorepo using TypeScript. Each package manages its own dependencies.
Also contains a Python/Streamlit movie recommendation system app.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Streamlit Movie Recommender

**Project Title:** Hybrid Movie Recommendation System using Data Warehousing and Data Mining

### Files
- `app.py` — Main Streamlit application (11 pages)
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — Streamlit server config (port 5000)
- `output/` — ETL-processed datasets:
  - `final_movies_dataset.csv` (~9,700 movies with IMDb metadata)
  - `final_ratings_dataset.csv` (~100,000 MovieLens ratings)
  - `movie_warehouse.db` — SQLite data warehouse

### Pages
1. Dashboard — OLAP-style warehouse analytics
2. Movie Explorer — Browse catalog + content similarity
3. Content-Based Recommender — TF-IDF + cosine similarity
4. User-Based CF — User cosine similarity matrix
5. Item-Based CF — Movie cosine similarity matrix
6. Matrix Factorization / SVD — TruncatedSVD latent factors
7. Hybrid Recommendation — SVD + CF + Content-Based with adjustable weights
8. User Clustering — K-Means on genre preference matrix
9. Association Rules — Apriori algorithm (mlxtend) or manual pair-based
10. Evaluation Metrics — RMSE, MAE, Precision@10, Recall@10, Silhouette
11. New User / Cold Start — Genre preference + seed ratings → content-based recs

### Run Command
```bash
streamlit run app.py --server.port 5000
```

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.
