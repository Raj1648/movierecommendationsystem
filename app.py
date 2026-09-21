"""
Hybrid Movie Recommendation System using Data Warehousing and Data Mining
College Project - Data Mining and Warehousing

Pipeline: Data Sources → ETL → Data Warehouse → Data Mining → Recommendation → Evaluation
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# DATA LOADING (CACHED)
# ─────────────────────────────────────────────────────────────

MOVIES_PATH = "output/final_movies_dataset.csv"
RATINGS_PATH = "output/final_ratings_dataset.csv"
DB_PATH = "output/movie_warehouse.db"


@st.cache_data
def load_movies():
    df = pd.read_csv(MOVIES_PATH)
    df.columns = df.columns.str.strip()
    if "movie_id" not in df.columns and "movieId" in df.columns:
        df.rename(columns={"movieId": "movie_id"}, inplace=True)
    if "release_year" not in df.columns and "year" in df.columns:
        df.rename(columns={"year": "release_year"}, inplace=True)
    for col in ["genres", "director", "cast", "content_features", "movie_title"]:
        if col not in df.columns:
            df[col] = "Unknown"
        else:
            df[col] = df[col].fillna("Unknown")
    for col in ["imdb_rating", "imdb_votes", "release_year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data
def load_ratings():
    df = pd.read_csv(RATINGS_PATH)
    df.columns = df.columns.str.strip()
    if "userId" in df.columns:
        df.rename(columns={"userId": "user_id"}, inplace=True)
    if "movieId" in df.columns:
        df.rename(columns={"movieId": "movie_id"}, inplace=True)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
    df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce").astype("Int64")
    df.dropna(subset=["user_id", "movie_id", "rating"], inplace=True)
    return df


@st.cache_data
def get_movie_stats(ratings_df):
    stats = ratings_df.groupby("movie_id")["rating"].agg(
        avg_rating="mean", rating_count="count"
    ).reset_index()
    return stats


@st.cache_data
def build_content_similarity(movies_df):
    df = movies_df.copy()
    if "content_features" in df.columns and df["content_features"].str.strip().ne("Unknown").any():
        text_col = "content_features"
    else:
        df["_text"] = (
            df.get("genres", "").fillna("") + " " +
            df.get("movie_title", "").fillna("") + " " +
            df.get("director", "").fillna("") + " " +
            df.get("cast", "").fillna("")
        )
        text_col = "_text"
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    mat = tfidf.fit_transform(df[text_col].fillna(""))
    return mat, df.reset_index(drop=True)


@st.cache_data
def build_user_movie_matrix(ratings_df, max_users=2000, max_movies=3000):
    top_users = ratings_df["user_id"].value_counts().head(max_users).index
    top_movies = ratings_df["movie_id"].value_counts().head(max_movies).index
    filtered = ratings_df[
        ratings_df["user_id"].isin(top_users) & ratings_df["movie_id"].isin(top_movies)
    ]
    matrix = filtered.pivot_table(index="user_id", columns="movie_id", values="rating")
    return matrix


@st.cache_resource
def compute_user_similarity(matrix):
    filled = matrix.fillna(0).values
    sim = cosine_similarity(csr_matrix(filled))
    return sim, list(matrix.index)


@st.cache_resource
def compute_item_similarity(matrix):
    filled = matrix.fillna(0).values.T
    sim = cosine_similarity(csr_matrix(filled))
    return sim, list(matrix.columns)


@st.cache_resource
def build_svd_model(matrix, n_components=50):
    """
    Mean-centered SVD:
    1. Compute each user's mean rating over observed entries.
    2. Subtract user mean from observed ratings (center the matrix).
    3. Fill unobserved (missing) entries with 0 in the centered matrix.
    4. Apply TruncatedSVD to the filled centered matrix.
    5. Reconstruct and add user mean back.
    6. Clip predictions to valid rating range [0.5, 5.0].
    This avoids treating missing ratings as 0-star ratings.
    """
    user_means = matrix.mean(axis=1)                     # mean per user (NaN ignored)
    centered   = matrix.sub(user_means, axis=0)          # subtract row means from observed
    filled     = centered.fillna(0).values               # fill unobserved centered slots with 0

    n_comp = min(n_components, min(filled.shape) - 1)
    svd    = TruncatedSVD(n_components=n_comp, random_state=42)
    U      = svd.fit_transform(filled)
    Vt     = svd.components_
    pred_centered = np.dot(U, Vt)

    predicted = pred_centered + user_means.values[:, np.newaxis]  # add means back
    predicted = np.clip(predicted, 0.5, 5.0)                      # clip to rating range
    return predicted, list(matrix.index), list(matrix.columns)


@st.cache_data
def build_user_genre_matrix(_ratings_df, _movies_df, max_users=1000):
    """
    Build a user × genre bias matrix.
    Each cell = user_genre_avg_rating − global_genre_avg_rating.
    Positive = user rates this genre above global average.
    Missing user-genre combinations are filled with 0 (neutral bias).
    """
    top_users = _ratings_df["user_id"].value_counts().head(max_users).index
    merged = (
        _ratings_df[_ratings_df["user_id"].isin(top_users)]
        .merge(_movies_df[["movie_id", "genres"]], on="movie_id", how="left")
    )
    merged["genres"] = merged["genres"].fillna("Unknown")
    merged = merged.copy()
    merged["genre"] = merged["genres"].str.split("|")
    exploded = merged.explode("genre")
    exploded["genre"] = exploded["genre"].str.strip()
    exploded = exploded[~exploded["genre"].isin(["Unknown", ""])]

    user_genre_avg = (
        exploded.groupby(["user_id", "genre"])["rating"]
        .mean()
        .unstack(fill_value=np.nan)
    )
    global_genre_avg = exploded.groupby("genre")["rating"].mean()
    bias_matrix = user_genre_avg.sub(global_genre_avg, axis=1).fillna(0)
    return bias_matrix


def get_movie_title(movie_id, movies_df):
    row = movies_df[movies_df["movie_id"] == movie_id]
    if len(row) == 0:
        return f"Movie {movie_id}"
    return row.iloc[0].get("movie_title", f"Movie {movie_id}")


def display_movie_card(row, score_label=None, score_value=None):
    with st.container():
        cols = st.columns([3, 1])
        with cols[0]:
            title = row.get("movie_title", "Unknown")
            year = int(row["release_year"]) if pd.notna(row.get("release_year")) else "N/A"
            genres = row.get("genres", "Unknown")
            st.markdown(f"**{title}** ({year})")
            st.caption(f"Genres: {genres}")
            if pd.notna(row.get("imdb_rating")):
                st.caption(f"IMDb: ⭐ {row['imdb_rating']:.1f}")
            if pd.notna(row.get("director")) and row["director"] not in ["Unknown", ""]:
                st.caption(f"Director: {row['director']}")
        with cols[1]:
            if score_label and score_value is not None:
                st.metric(score_label, f"{score_value:.3f}")
        st.divider()


def export_recommendations(rows, score_label, filename_prefix, key):
    """Render a Download CSV button for a list of recommendation result dicts."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    col_order = [c for c in [
        "Rank", "Title", "Year", "Genres", "Director", "IMDb Rating", "IMDb Votes", score_label
    ] if c in df.columns]
    df = df[col_order]
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇ Download {len(df)} recommendations as CSV",
        data=csv_bytes,
        file_name=f"{filename_prefix}_recommendations.csv",
        mime="text/csv",
        key=key,
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────
# PAGE IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────

def page_dashboard():
    st.title("Dashboard — Data Warehouse Analytics")
    st.markdown(
        """
        > **Data Warehousing Concept:** This dashboard represents OLAP-style analytics over the movie data warehouse.
        > Summary statistics and aggregations are computed from ETL-processed datasets, simulating dimensional reporting.
        >
        > **Pipeline:** `MovieLens Ratings` + `IMDb Metadata` → **ETL** → `Data Warehouse` → **OLAP Queries** → Analytics
        """
    )

    movies = load_movies()
    ratings = load_ratings()
    stats = get_movie_stats(ratings)

    # KPI Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Movies", f"{len(movies):,}")
    c2.metric("Total Users", f"{ratings['user_id'].nunique():,}")
    c3.metric("Total Ratings", f"{len(ratings):,}")
    c4.metric("Avg Rating", f"{ratings['rating'].mean():.2f}")
    all_genres = "|".join(movies["genres"].dropna()).split("|")
    unique_genres = len(set(g.strip() for g in all_genres if g.strip()))
    c5.metric("Unique Genres", unique_genres)
    imdb_cov = movies["imdb_rating"].notna().sum()
    c6.metric("IMDb Coverage", f"{imdb_cov:,}")

    st.markdown("---")

    # Charts row 1
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rating Distribution")
        fig, ax = plt.subplots(figsize=(6, 3))
        ratings["rating"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#e50914")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Ratings")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Top 10 Most Rated Movies")
        top_rated = stats.nlargest(10, "rating_count")
        top_rated = top_rated.merge(movies[["movie_id", "movie_title"]], on="movie_id", how="left")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(top_rated["movie_title"].str[:30], top_rated["rating_count"], color="#e50914")
        ax.set_xlabel("Number of Ratings")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Charts row 2
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Top 10 Highest Rated (≥20 ratings)")
        qualified = stats[stats["rating_count"] >= 20].nlargest(10, "avg_rating")
        qualified = qualified.merge(movies[["movie_id", "movie_title"]], on="movie_id", how="left")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(qualified["movie_title"].str[:30], qualified["avg_rating"], color="#564d4d")
        ax.set_xlabel("Average Rating")
        ax.set_xlim(0, 5)
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        st.subheader("Most Common Genres")
        genre_counts = pd.Series(all_genres).value_counts().head(15)
        genre_counts = genre_counts[genre_counts.index.str.strip() != ""]
        fig, ax = plt.subplots(figsize=(6, 3))
        genre_counts.plot(kind="barh", ax=ax, color="#e50914")
        ax.set_xlabel("Count")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    if "year" in ratings.columns:
        st.subheader("Ratings by Year")
        year_counts = ratings.groupby("year")["rating"].count().reset_index()
        year_counts = year_counts[year_counts["year"] > 1990]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(year_counts["year"], year_counts["rating"], marker="o", color="#e50914")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Ratings")
        ax.set_title("Ratings Activity Over Time")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


@st.cache_data
def build_person_index(movies_df):
    """Build a lookup: person name → list of movie_ids they appear in (director or cast)."""
    person_map = {}
    for _, row in movies_df.iterrows():
        mid = row["movie_id"]
        director = str(row.get("director", "")).strip()
        cast_str = str(row.get("cast", "")).strip()
        people = []
        if director and director not in ("Unknown", "nan", ""):
            people.append(("Director", director))
        for actor in cast_str.split(","):
            actor = actor.strip()
            if actor and actor not in ("Unknown", "nan", ""):
                people.append(("Cast", actor))
        for role, name in people:
            key = name.lower()
            if key not in person_map:
                person_map[key] = {"name": name, "roles": set(), "movie_ids": []}
            person_map[key]["roles"].add(role)
            person_map[key]["movie_ids"].append(mid)
    return person_map


def page_movie_explorer():
    st.title("Movie Explorer")
    st.markdown(
        """
        > Browse the movie catalog, explore metadata, and search by **actor or director**.
        > Similar movies are computed using **content-based similarity** (TF-IDF + cosine similarity).
        """
    )
    movies = load_movies()
    ratings = load_ratings()
    stats = get_movie_stats(ratings)
    movies_with_stats = movies.merge(stats, on="movie_id", how="left")

    tab_movie, tab_person = st.tabs(["Search by Movie", "Search by Actor / Director"])

    # ── TAB 1: Movie Search ──────────────────────────────────
    with tab_movie:
        titles = sorted(movies["movie_title"].dropna().unique().tolist())
        selected_title = st.selectbox("Search / Select Movie", titles, key="explorer_movie")

        row = movies_with_stats[movies_with_stats["movie_title"] == selected_title]
        if len(row) == 0:
            st.warning("Movie not found.")
        else:
            row = row.iloc[0]
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### Movie Details")
                st.markdown(f"**Title:** {row.get('movie_title', 'N/A')}")
                year = int(row["release_year"]) if pd.notna(row.get("release_year")) else "N/A"
                st.markdown(f"**Year:** {year}")
                st.markdown(f"**Genres:** {row.get('genres', 'N/A')}")
                if pd.notna(row.get("imdb_rating")):
                    st.markdown(f"**IMDb Rating:** ⭐ {row['imdb_rating']:.1f}")
                if pd.notna(row.get("imdb_votes")):
                    st.markdown(f"**IMDb Votes:** {int(row['imdb_votes']):,}")
                if pd.notna(row.get("director")) and row["director"] not in ["Unknown", ""]:
                    st.markdown(f"**Director:** {row['director']}")
                if pd.notna(row.get("cast")) and row["cast"] not in ["Unknown", ""]:
                    st.markdown(f"**Cast:** {row['cast'][:120]}")

            with col2:
                st.markdown("### MovieLens Stats")
                if pd.notna(row.get("avg_rating")):
                    st.metric("Average Rating", f"{row['avg_rating']:.2f} / 5.0")
                    st.metric("Number of Ratings", f"{int(row['rating_count']):,}")
                else:
                    st.info("No ratings available for this movie.")

            st.markdown("---")
            st.subheader("Similar Movies (Content-Based)")
            mat, indexed_movies = build_content_similarity(movies)
            movie_idx = indexed_movies[indexed_movies["movie_title"] == selected_title].index
            if len(movie_idx) == 0:
                st.warning("Cannot compute similarity — movie not in index.")
            else:
                idx = movie_idx[0]
                sim_scores = cosine_similarity(mat[idx], mat).flatten()
                sim_scores[idx] = 0
                top_indices = sim_scores.argsort()[::-1][:8]
                for i in top_indices:
                    r = indexed_movies.iloc[i]
                    display_movie_card(r, "Similarity", sim_scores[i])

    # ── TAB 2: Person Search ─────────────────────────────────
    with tab_person:
        st.markdown(
            """
            Search for any **actor or director** to see their full filmography in this dataset,
            sorted by IMDb rating. Then get content-based recommendations derived from their body of work.
            """
        )

        person_map = build_person_index(movies)
        all_names = sorted({v["name"] for v in person_map.values()})

        col_a, col_b = st.columns([3, 1])
        with col_a:
            search_input = st.text_input(
                "Type an actor or director name",
                placeholder="e.g. Martin Scorsese, Tom Hanks, Meryl Streep",
                key="person_search"
            )
        with col_b:
            n_similar = st.slider("Similar movies to show", 3, 15, 8, key="person_n_sim")

        if not search_input.strip():
            st.info("Start typing a name above to explore their filmography.")
        else:
            query = search_input.strip().lower()
            matches = {k: v for k, v in person_map.items() if query in k}

            if not matches:
                st.warning(f"No results for **'{search_input}'**. Check spelling or try a partial name.")
            else:
                # Pick best match (longest overlap / exact preferred)
                best_key = min(matches.keys(), key=lambda k: abs(len(k) - len(query)))
                person = matches[best_key]

                if len(matches) > 1:
                    name_options = sorted({v["name"] for v in matches.values()})
                    chosen_name = st.selectbox("Multiple matches — select one", name_options, key="person_pick")
                    best_key = next(k for k, v in matches.items() if v["name"] == chosen_name)
                    person = matches[best_key]

                roles_label = " & ".join(sorted(person["roles"]))
                st.success(f"Showing filmography for **{person['name']}** ({roles_label})")

                person_movies = movies[movies["movie_id"].isin(person["movie_ids"])].copy()
                person_movies = person_movies.merge(stats, on="movie_id", how="left")

                # Sort by IMDb rating descending
                if "imdb_rating" in person_movies.columns:
                    person_movies = person_movies.sort_values("imdb_rating", ascending=False)

                # Summary KPIs
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Films in Dataset", len(person_movies))
                rated = person_movies["avg_rating"].dropna()
                k2.metric("Avg MovieLens Rating", f"{rated.mean():.2f}" if len(rated) > 0 else "N/A")
                imdb = person_movies["imdb_rating"].dropna()
                k3.metric("Avg IMDb Rating", f"{imdb.mean():.1f}" if len(imdb) > 0 else "N/A")
                if "release_year" in person_movies.columns:
                    years = person_movies["release_year"].dropna()
                    if len(years) > 0:
                        k4.metric("Active Years", f"{int(years.min())} – {int(years.max())}")

                st.markdown("---")
                st.subheader("Filmography")

                # Genre breakdown chart
                all_genres_person = "|".join(person_movies["genres"].dropna()).split("|")
                genre_counts = pd.Series(all_genres_person).value_counts().head(8)
                genre_counts = genre_counts[genre_counts.index.str.strip() != ""]
                if len(genre_counts) > 0:
                    fig, ax = plt.subplots(figsize=(7, 2.5))
                    genre_counts.plot(kind="barh", ax=ax, color="#e50914")
                    ax.set_xlabel("Films")
                    ax.set_title(f"Genres in {person['name']}'s Filmography")
                    ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                # Film list table
                display_cols = ["movie_title", "release_year", "genres", "imdb_rating", "avg_rating", "rating_count"]
                display_cols = [c for c in display_cols if c in person_movies.columns]
                col_rename = {
                    "movie_title": "Title",
                    "release_year": "Year",
                    "genres": "Genres",
                    "imdb_rating": "IMDb",
                    "avg_rating": "ML Avg",
                    "rating_count": "ML Votes"
                }
                show_df = person_movies[display_cols].rename(columns=col_rename)
                if "Year" in show_df.columns:
                    show_df["Year"] = show_df["Year"].apply(lambda y: int(y) if pd.notna(y) else "")
                if "IMDb" in show_df.columns:
                    show_df["IMDb"] = show_df["IMDb"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")
                if "ML Avg" in show_df.columns:
                    show_df["ML Avg"] = show_df["ML Avg"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
                if "ML Votes" in show_df.columns:
                    show_df["ML Votes"] = show_df["ML Votes"].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "")
                st.dataframe(show_df.reset_index(drop=True), use_container_width=True)

                # Content-based recommendations from their filmography
                st.markdown("---")
                st.subheader(f"Movies You'll Like Based on {person['name']}'s Work")
                st.caption("Computed by averaging content similarity across their top-rated films.")

                mat, indexed_movies = build_content_similarity(movies)
                top_films = person_movies.dropna(subset=["imdb_rating"]).nlargest(5, "imdb_rating")
                if len(top_films) == 0:
                    top_films = person_movies.head(5)

                agg_scores = np.zeros(len(indexed_movies))
                seed_count = 0
                for _, seed_row in top_films.iterrows():
                    seed_mid = seed_row["movie_id"]
                    seed_idx = indexed_movies[indexed_movies["movie_id"] == seed_mid].index
                    if len(seed_idx) == 0:
                        continue
                    sims = cosine_similarity(mat[seed_idx[0]], mat).flatten()
                    agg_scores += sims
                    seed_count += 1

                if seed_count == 0:
                    st.info("Not enough data to generate recommendations.")
                else:
                    agg_scores /= seed_count
                    # Zero out the person's own films
                    own_ids = set(person_movies["movie_id"].tolist())
                    for j in range(len(indexed_movies)):
                        if indexed_movies.iloc[j]["movie_id"] in own_ids:
                            agg_scores[j] = 0

                    top_rec_indices = agg_scores.argsort()[::-1][:n_similar]
                    for i in top_rec_indices:
                        r = indexed_movies.iloc[i]
                        display_movie_card(r, "Relevance", agg_scores[i])


def page_content_based():
    st.title("Content-Based Recommender")
    st.markdown(
        """
        > **Technique:** TF-IDF Vectorizer + Cosine Similarity on `content_features` (genres, director, cast, title).
        > Content-based filtering recommends movies similar to a selected movie based on metadata features —
        > it does not require user rating history.
        """
    )
    movies = load_movies()
    mat, indexed_movies = build_content_similarity(movies)

    col1, col2 = st.columns([2, 1])
    with col1:
        titles = sorted(indexed_movies["movie_title"].dropna().unique().tolist())
        selected_title = st.selectbox("Select a Movie", titles)
    with col2:
        n_recs = st.slider("Number of Recommendations", 5, 20, 10)

    genre_list = sorted(set(
        g.strip() for g in "|".join(movies["genres"].dropna()).split("|") if g.strip()
    ))
    genre_filter = st.multiselect("Filter by Genre (optional)", genre_list)

    if "release_year" in movies.columns:
        years = movies["release_year"].dropna()
        min_y, max_y = int(years.min()), int(years.max())
        year_range = st.slider("Year Range", min_y, max_y, (min_y, max_y))
    else:
        year_range = None

    movie_idx = indexed_movies[indexed_movies["movie_title"] == selected_title].index
    if len(movie_idx) == 0:
        st.warning("Movie not found in index.")
        return
    idx = movie_idx[0]

    sim_scores = cosine_similarity(mat[idx], mat).flatten()
    sim_scores[idx] = 0

    recs = indexed_movies.copy()
    recs["_sim"] = sim_scores

    if genre_filter:
        mask = recs["genres"].apply(
            lambda g: any(genre in str(g) for genre in genre_filter)
        )
        recs = recs[mask]

    if year_range and "release_year" in recs.columns:
        recs = recs[
            (recs["release_year"] >= year_range[0]) & (recs["release_year"] <= year_range[1])
        ]

    recs = recs[recs["movie_title"] != selected_title].nlargest(n_recs, "_sim")

    st.subheader(f"Top {n_recs} Recommendations for '{selected_title}'")
    if len(recs) == 0:
        st.info("No recommendations found with current filters.")
    else:
        export_rows = []
        for rank, (_, r) in enumerate(recs.iterrows(), 1):
            display_movie_card(r, "Similarity", r["_sim"])
            export_rows.append({
                "Rank": rank,
                "Title": r.get("movie_title", ""),
                "Year": int(r["release_year"]) if pd.notna(r.get("release_year")) else "",
                "Genres": r.get("genres", ""),
                "Director": r.get("director", ""),
                "IMDb Rating": r.get("imdb_rating", ""),
                "IMDb Votes": int(r["imdb_votes"]) if pd.notna(r.get("imdb_votes")) else "",
                "Similarity Score": round(r["_sim"], 4),
            })
        export_recommendations(export_rows, "Similarity Score", "content_based", "export_cb")


def page_user_based_cf():
    st.title("User-Based Collaborative Filtering")
    st.markdown(
        """
        > **Technique:** User-Movie rating matrix + Cosine Similarity between users.
        > Users with similar rating patterns are identified, and movies they liked (but the target user hasn't seen) are recommended.
        > The predicted score is a weighted average of ratings from similar users.
        """
    )
    movies = load_movies()
    ratings = load_ratings()
    matrix = build_user_movie_matrix(ratings)
    user_sim, user_ids = compute_user_similarity(matrix)

    all_users = sorted(ratings["user_id"].dropna().unique().tolist())
    selected_user = st.selectbox("Select Existing User ID", all_users, key="ubcf_user")
    n_recs = st.slider("Number of Recommendations", 5, 20, 10, key="ubcf_n")

    if selected_user not in user_ids:
        st.warning("This user is not in the training matrix (insufficient ratings). Please select another.")
        return

    user_idx = user_ids.index(selected_user)
    sim_row = user_sim[user_idx]

    user_ratings = ratings[ratings["user_id"] == selected_user]
    rated_ids = set(user_ratings["movie_id"].tolist())

    col1, col2, col3 = st.columns(3)
    col1.metric("Movies Rated", len(rated_ids))
    col2.metric("Average Rating", f"{user_ratings['rating'].mean():.2f}")
    genre_series = user_ratings.merge(movies[["movie_id", "genres"]], on="movie_id", how="left")["genres"]
    fav_genres = pd.Series(
        "|".join(genre_series.dropna()).split("|")
    ).value_counts().head(3).index.tolist()
    col3.metric("Top Genres", ", ".join(fav_genres) if fav_genres else "N/A")

    st.subheader("Recently Rated Movies")
    sample = user_ratings.merge(movies[["movie_id", "movie_title"]], on="movie_id", how="left").head(8)
    st.dataframe(sample[["movie_title", "rating"]], use_container_width=True)

    st.subheader(f"Top {n_recs} Recommendations")
    # Mean-centered CF: prediction(u,i) = mean(u) + Σ sim(u,v)*(r(v,i)-mean(v)) / Σ|sim(u,v)|
    target_mean = user_ratings["rating"].mean()
    # Use top-30 most similar neighbours (enough signal without too much noise)
    top_similar = np.argsort(sim_row)[::-1][1:31]
    # candidate_scores[mid] = [weighted_sum_of_deviations, sum_of_abs_weights, neighbour_count]
    candidate_scores = {}
    EPS = 1e-9
    SHRINKAGE_K = 25  # shrinkage constant — higher = more shrinkage for items with few raters
    MIN_NEIGHBOURS = 2  # discard candidates rated by fewer than this many neighbours

    for sim_idx in top_similar:
        sim_uid = user_ids[sim_idx]
        sim_weight = sim_row[sim_idx]
        if sim_weight <= 0:
            continue
        sim_user_ratings = ratings[ratings["user_id"] == sim_uid]
        if len(sim_user_ratings) == 0:
            continue
        neighbor_mean = sim_user_ratings["rating"].mean()
        for _, rr in sim_user_ratings.iterrows():
            mid = int(rr["movie_id"])
            if mid in rated_ids:
                continue
            centered = rr["rating"] - neighbor_mean
            if mid not in candidate_scores:
                candidate_scores[mid] = [0.0, 0.0, 0]
            candidate_scores[mid][0] += sim_weight * centered
            candidate_scores[mid][1] += abs(sim_weight)
            candidate_scores[mid][2] += 1          # neighbour count

    if not candidate_scores:
        st.info("Not enough data to generate recommendations.")
        return

    scores = {}
    for mid, (num, den, cnt) in candidate_scores.items():
        if cnt < MIN_NEIGHBOURS or den < EPS:
            continue
        raw_deviation = num / den
        # Shrinkage: dampen deviation for items with few supporting neighbours.
        # With cnt=2, factor≈0.07; cnt=10, factor≈0.29; cnt=50, factor≈0.67
        shrinkage = cnt / (cnt + SHRINKAGE_K)
        pred = target_mean + shrinkage * raw_deviation
        scores[mid] = float(np.clip(pred, 0.5, 5.0))

    if not scores:
        st.info("No candidates had enough neighbour support. Try a more active user.")
        return

    if scores:
        vals = list(scores.values())
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Min Predicted Score", f"{min(vals):.3f}")
        sc2.metric("Max Predicted Score", f"{max(vals):.3f}")
        sc3.metric("Mean Predicted Score", f"{np.mean(vals):.3f}")

    top_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_recs]

    export_rows = []
    for rank, (mid, score) in enumerate(top_movies, 1):
        row = movies[movies["movie_id"] == mid]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        display_movie_card(r, "Pred. Score", score)
        export_rows.append({
            "Rank": rank,
            "Title": r.get("movie_title", ""),
            "Year": int(r["release_year"]) if pd.notna(r.get("release_year")) else "",
            "Genres": r.get("genres", ""),
            "Director": r.get("director", ""),
            "IMDb Rating": r.get("imdb_rating", ""),
            "IMDb Votes": int(r["imdb_votes"]) if pd.notna(r.get("imdb_votes")) else "",
            "Predicted Score": round(score, 4),
        })
    export_recommendations(export_rows, "Predicted Score", f"user_cf_user{selected_user}", "export_ubcf")


def page_item_based_cf():
    st.title("Item-Based Collaborative Filtering")
    st.markdown(
        """
        > **Technique:** Movie-User rating matrix + Cosine Similarity between movies.
        > Finds movies that were rated similarly by users — capturing taste patterns without needing user profiles.
        """
    )
    movies = load_movies()
    ratings = load_ratings()
    matrix = build_user_movie_matrix(ratings)
    item_sim, movie_ids = compute_item_similarity(matrix)

    titles_in_matrix = movies[movies["movie_id"].isin(movie_ids)]["movie_title"].dropna().sort_values().tolist()
    selected_title = st.selectbox("Select a Movie", titles_in_matrix, key="ibcf_movie")
    n_recs = st.slider("Number of Recommendations", 5, 20, 10, key="ibcf_n")

    sel_id = movies[movies["movie_title"] == selected_title]["movie_id"]
    if len(sel_id) == 0 or int(sel_id.iloc[0]) not in movie_ids:
        st.warning("Movie not in matrix.")
        return

    mid = int(sel_id.iloc[0])
    movie_idx = movie_ids.index(mid)
    sim_scores = item_sim[movie_idx]
    sim_scores[movie_idx] = 0

    top_indices = np.argsort(sim_scores)[::-1][:n_recs]
    st.subheader(f"Movies Similar to '{selected_title}' (by Rating Behavior)")

    export_rows = []
    for rank, i in enumerate(top_indices, 1):
        similar_mid = movie_ids[i]
        row = movies[movies["movie_id"] == similar_mid]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        display_movie_card(r, "Similarity", sim_scores[i])
        export_rows.append({
            "Rank": rank,
            "Title": r.get("movie_title", ""),
            "Year": int(r["release_year"]) if pd.notna(r.get("release_year")) else "",
            "Genres": r.get("genres", ""),
            "Director": r.get("director", ""),
            "IMDb Rating": r.get("imdb_rating", ""),
            "IMDb Votes": int(r["imdb_votes"]) if pd.notna(r.get("imdb_votes")) else "",
            "Similarity Score": round(float(sim_scores[i]), 4),
        })
    export_recommendations(export_rows, "Similarity Score", "item_cf", "export_ibcf")


def page_svd():
    st.title("Matrix Factorization / SVD")
    st.markdown(
        """
        > **Technique:** Truncated SVD (Singular Value Decomposition) on the user-movie rating matrix.
        > SVD decomposes the matrix into latent factors capturing hidden user preferences and movie characteristics.
        > Missing ratings are filled with 0 before factorization.
        >
        > `R ≈ U × Σ × Vᵀ` — where U = user factors, Vᵀ = movie factors, Σ = singular values (importance)
        """
    )
    movies = load_movies()
    ratings = load_ratings()
    matrix = build_user_movie_matrix(ratings)
    predicted, user_ids, movie_ids = build_svd_model(matrix)

    all_users = sorted(ratings["user_id"].dropna().unique().tolist())
    selected_user = st.selectbox("Select Existing User ID", all_users, key="svd_user")
    n_recs = st.slider("Number of Recommendations", 5, 20, 10, key="svd_n")

    if selected_user not in user_ids:
        st.warning("User not in training matrix. Try another user.")
        return

    user_idx = user_ids.index(selected_user)
    pred_row = predicted[user_idx]

    user_rated = set(ratings[ratings["user_id"] == selected_user]["movie_id"].tolist())
    movie_scores = [
        (movie_ids[i], pred_row[i])
        for i in range(len(movie_ids))
        if movie_ids[i] not in user_rated
    ]
    movie_scores.sort(key=lambda x: x[1], reverse=True)
    top_movies = movie_scores[:n_recs]

    st.subheader(f"Top {n_recs} SVD Recommendations for User {selected_user}")
    export_rows = []
    for rank, (mid, score) in enumerate(top_movies, 1):
        row = movies[movies["movie_id"] == mid]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        display_movie_card(r, "SVD Score", score)
        export_rows.append({
            "Rank": rank,
            "Title": r.get("movie_title", ""),
            "Year": int(r["release_year"]) if pd.notna(r.get("release_year")) else "",
            "Genres": r.get("genres", ""),
            "Director": r.get("director", ""),
            "IMDb Rating": r.get("imdb_rating", ""),
            "IMDb Votes": int(r["imdb_votes"]) if pd.notna(r.get("imdb_votes")) else "",
            "SVD Score": round(float(score), 4),
        })
    export_recommendations(export_rows, "SVD Score", f"svd_user{selected_user}", "export_svd")


def page_hybrid():
    st.title("Hybrid Recommendation System")
    st.markdown(
        """
        > **This is the main recommendation engine.** It combines three techniques:
        > - **SVD** — learned latent factor model (collaborative signal)
        > - **User-Based CF** — neighborhood-based collaborative filtering
        > - **Content-Based** — TF-IDF similarity on user's top-rated movies
        >
        > `Hybrid Score = w₁×SVD + w₂×CF + w₃×Content`
        """
    )
    movies = load_movies()
    ratings = load_ratings()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        all_users = sorted(ratings["user_id"].dropna().unique().tolist())
        selected_user = st.selectbox("Select User ID", all_users, key="hyb_user")
    with col2:
        n_recs = st.slider("Recommendations", 5, 20, 10, key="hyb_n")
    with col3:
        genre_list = sorted(set(
            g.strip() for g in "|".join(movies["genres"].dropna()).split("|") if g.strip()
        ))
        genre_filter = st.multiselect("Genre Filter", genre_list, key="hyb_genre")

    st.markdown("#### Adjust Weights")
    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        w_svd = st.slider("SVD Weight", 0.0, 1.0, 0.5, 0.05)
    with wc2:
        w_cf = st.slider("Collaborative Weight", 0.0, 1.0, 0.3, 0.05)
    with wc3:
        w_cb = st.slider("Content Weight", 0.0, 1.0, 0.2, 0.05)

    total_w = w_svd + w_cf + w_cb
    if total_w == 0:
        st.error("Weights must sum to more than 0.")
        return
    w_svd, w_cf, w_cb = w_svd / total_w, w_cf / total_w, w_cb / total_w
    st.caption(f"Normalized weights — SVD: {w_svd:.2f} | CF: {w_cf:.2f} | Content: {w_cb:.2f}")

    matrix = build_user_movie_matrix(ratings)
    movie_ids_in_matrix = list(matrix.columns)
    user_ids_in_matrix = list(matrix.index)
    user_rated = set(ratings[ratings["user_id"] == selected_user]["movie_id"].tolist())

    # SVD scores
    svd_scores = {}
    if selected_user in user_ids_in_matrix:
        predicted, u_ids, m_ids = build_svd_model(matrix)
        u_idx = u_ids.index(selected_user)
        pred_row = predicted[u_idx]
        raw = np.array(pred_row)
        if raw.max() != raw.min():
            raw = (raw - raw.min()) / (raw.max() - raw.min())
        for i, mid in enumerate(m_ids):
            svd_scores[mid] = float(raw[i])

    # CF scores (mean-centered + shrinkage)
    cf_scores = {}
    if selected_user in user_ids_in_matrix:
        user_sim, user_ids_list = compute_user_similarity(matrix)
        u_idx = user_ids_list.index(selected_user)
        sim_row_cf = user_sim[u_idx]
        target_mean_cf = ratings[ratings["user_id"] == selected_user]["rating"].mean()
        top_similar = np.argsort(sim_row_cf)[::-1][1:31]
        raw_cf = {}   # mid -> [num, den, count]
        EPS_CF = 1e-9
        SHRINKAGE_K_CF = 25
        for sim_idx in top_similar:
            sim_uid = user_ids_list[sim_idx]
            sim_weight = sim_row_cf[sim_idx]
            if sim_weight <= 0:
                continue
            neighbor_ratings = ratings[ratings["user_id"] == sim_uid]
            if len(neighbor_ratings) == 0:
                continue
            neighbor_mean = neighbor_ratings["rating"].mean()
            for _, rr in neighbor_ratings.iterrows():
                mid = int(rr["movie_id"])
                if mid in user_rated:
                    continue
                centered = rr["rating"] - neighbor_mean
                if mid not in raw_cf:
                    raw_cf[mid] = [0.0, 0.0, 0]
                raw_cf[mid][0] += sim_weight * centered
                raw_cf[mid][1] += abs(sim_weight)
                raw_cf[mid][2] += 1
        for mid, (num, den, cnt) in raw_cf.items():
            if cnt < 2 or den < EPS_CF:
                continue
            shrinkage = cnt / (cnt + SHRINKAGE_K_CF)
            pred = target_mean_cf + shrinkage * (num / den)
            cf_scores[mid] = float(np.clip(pred, 0.5, 5.0))
        if cf_scores:
            vals = np.array(list(cf_scores.values()))
            if vals.max() != vals.min():
                norm_vals = (vals - vals.min()) / (vals.max() - vals.min())
            else:
                norm_vals = np.ones(len(vals)) * 0.5
            cf_scores = dict(zip(cf_scores.keys(), norm_vals.tolist()))

    # Content-Based scores
    cb_scores = {}
    top_rated_movies = (
        ratings[ratings["user_id"] == selected_user]
        .nlargest(5, "rating")["movie_id"].tolist()
    )
    if top_rated_movies:
        mat, indexed_movies = build_content_similarity(movies)
        for seed_mid in top_rated_movies:
            seed_row = indexed_movies[indexed_movies["movie_id"] == seed_mid]
            if len(seed_row) == 0:
                continue
            sidx = seed_row.index[0]
            sims = cosine_similarity(mat[sidx], mat).flatten()
            for j, sim_val in enumerate(sims):
                cand_mid = int(indexed_movies.iloc[j]["movie_id"])
                if cand_mid in user_rated:
                    continue
                cb_scores[cand_mid] = cb_scores.get(cand_mid, 0) + sim_val
        if cb_scores:
            vals = np.array(list(cb_scores.values()))
            if vals.max() != vals.min():
                norm_vals = (vals - vals.min()) / (vals.max() - vals.min())
            else:
                norm_vals = vals
            cb_scores = dict(zip(cb_scores.keys(), norm_vals))

    # Combine scores
    all_candidates = set(svd_scores) | set(cf_scores) | set(cb_scores)
    hybrid = {}
    for mid in all_candidates:
        if mid in user_rated:
            continue
        s = (
            w_svd * svd_scores.get(mid, 0) +
            w_cf * cf_scores.get(mid, 0) +
            w_cb * cb_scores.get(mid, 0)
        )
        hybrid[mid] = s

    if genre_filter:
        hybrid = {
            mid: s for mid, s in hybrid.items()
            if any(
                genre in str(movies[movies["movie_id"] == mid]["genres"].values[0])
                for genre in genre_filter
            ) if len(movies[movies["movie_id"] == mid]) > 0
        }

    top_hyb = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)[:n_recs]
    st.subheader(f"Top {n_recs} Hybrid Recommendations for User {selected_user}")

    export_rows = []
    for rank, (mid, score) in enumerate(top_hyb, 1):
        row = movies[movies["movie_id"] == mid]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        svd_c = svd_scores.get(mid, 0)
        cf_c = cf_scores.get(mid, 0)
        cb_c = cb_scores.get(mid, 0)
        reasons = []
        if svd_c > 0.6:
            reasons.append("SVD predicted a high score")
        if cf_c > 0.6:
            reasons.append("similar users liked it")
        if cb_c > 0.6:
            reasons.append("matches your preferred genres/directors")
        reason = "Recommended because " + (", and ".join(reasons) if reasons else "it scored well across all models") + "."
        with st.container():
            cols = st.columns([3, 1])
            with cols[0]:
                title = r.get("movie_title", "Unknown")
                year = int(r["release_year"]) if pd.notna(r.get("release_year")) else "N/A"
                st.markdown(f"**{title}** ({year}) — Genres: {r.get('genres', 'N/A')}")
                st.caption(reason)
            with cols[1]:
                st.metric("Hybrid Score", f"{score:.3f}")
            st.divider()
        export_rows.append({
            "Rank": rank,
            "Title": r.get("movie_title", ""),
            "Year": int(r["release_year"]) if pd.notna(r.get("release_year")) else "",
            "Genres": r.get("genres", ""),
            "Director": r.get("director", ""),
            "IMDb Rating": r.get("imdb_rating", ""),
            "IMDb Votes": int(r["imdb_votes"]) if pd.notna(r.get("imdb_votes")) else "",
            "Hybrid Score": round(score, 4),
            "SVD Component": round(svd_c, 4),
            "CF Component": round(cf_c, 4),
            "Content Component": round(cb_c, 4),
            "Reason": reason,
        })
    export_recommendations(export_rows, "Hybrid Score", f"hybrid_user{selected_user}", "export_hybrid")


def page_user_clustering():
    st.title("User Clustering — K-Means")
    st.markdown(
        """
        > **Technique:** K-Means clustering on the user-genre preference matrix.
        > Users are grouped by their genre rating behavior. Each cluster represents a distinct viewer profile.
        > K-Means minimizes within-cluster variance (inertia) to find natural groupings.
        """
    )
    movies = load_movies()
    ratings = load_ratings()

    col1, col2 = st.columns(2)
    with col1:
        k = st.slider("Number of Clusters (K)", 2, 10, 5)
    with col2:
        all_users = sorted(ratings["user_id"].dropna().unique().tolist())
        selected_user = st.selectbox("Select User ID", all_users, key="clus_user")

    with st.spinner("Building genre preference matrix and clustering..."):
        # Use cached vectorised user-genre matrix (mean rating per genre per user)
        genre_matrix = build_user_genre_matrix(ratings, movies, max_users=1000)

        if genre_matrix.shape[0] < k:
            st.warning(f"Only {genre_matrix.shape[0]} users available — reduce K.")
            return

        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score

        X = StandardScaler().fit_transform(genre_matrix.values)

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        genre_matrix = genre_matrix.copy()
        genre_matrix["cluster"] = labels

        ka, kb = st.columns(2)
        ka.metric("Inertia", f"{km.inertia_:.1f}")
        try:
            sil = silhouette_score(X, labels, sample_size=min(500, len(genre_matrix)))
            kb.metric("Silhouette Score", f"{sil:.4f}",
                      help="Closer to 1 = well-separated clusters. 0.1–0.4 is typical for genre-based user profiles.")
        except Exception:
            pass

        st.caption(
            "Clustering is performed on a **user × genre** matrix (mean rating per genre), "
            "standardised before K-Means. This is more interpretable than clustering on the "
            "raw sparse user-movie matrix."
        )

        if selected_user in genre_matrix.index:
            user_cluster = genre_matrix.loc[selected_user, "cluster"]
            cluster_members = genre_matrix[genre_matrix["cluster"] == user_cluster]
            st.success(f"User **{selected_user}** is in **Cluster {user_cluster}** ({len(cluster_members)} users)")

            cluster_bias = cluster_members.drop(columns=["cluster"]).mean().sort_values(ascending=False)
            top_pos = cluster_bias[cluster_bias > 0].head(8)
            top_neg = cluster_bias[cluster_bias < 0].tail(4)
            plot_data = pd.concat([top_pos, top_neg]).sort_values(ascending=False)

            st.subheader("Top Genre Biases in This Cluster")
            st.caption("Positive = cluster rates this genre **above** global average; negative = **below** average.")
            fig, ax = plt.subplots(figsize=(7, 3))
            colors = ["#2ca02c" if v >= 0 else "#e50914" for v in plot_data.values]
            plot_data.plot(kind="bar", ax=ax, color=colors)
            ax.set_ylabel("Rating Bias (user avg − global avg)")
            ax.axhline(0, color="black", linewidth=0.8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            if len(top_pos) > 0:
                st.info(f"This cluster strongly prefers: **{', '.join(top_pos.head(3).index.tolist())}**")

            cluster_user_ids = cluster_members.index.tolist()
            cluster_ratings = ratings[ratings["user_id"].isin(cluster_user_ids)]
            top_cluster_movies = (
                cluster_ratings.groupby("movie_id")["rating"]
                .agg(avg="mean", cnt="count")
                .reset_index()
                .query("cnt >= 3")
                .nlargest(10, "avg")
                .merge(movies[["movie_id", "movie_title", "genres"]], on="movie_id", how="left")
            )
            st.subheader("Top Movies in This Cluster")
            st.dataframe(top_cluster_movies[["movie_title", "genres", "avg", "cnt"]].rename(
                columns={"avg": "Avg Rating", "cnt": "Votes"}
            ), use_container_width=True)
        else:
            st.info("Selected user not in clustered set (too few ratings). Showing cluster summary.")

        st.subheader("Cluster Sizes")
        cluster_sizes = genre_matrix["cluster"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 3))
        cluster_sizes.plot(kind="bar", ax=ax, color="#564d4d")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of Users")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def page_association_rules():
    st.title("Association Rules Mining")
    st.markdown(
        """
        > **Technique:** Apriori Algorithm on user-movie transactions.
        > Each user's highly-rated movies (rating ≥ 4) form a transaction. Rules like
        > *"If a user liked Movie A, they will also like Movie B"* are discovered.
        >
        > **Metrics:**
        > - **Support** = fraction of users who rated both movies highly
        > - **Confidence** = P(Movie B | Movie A)
        > - **Lift** = how much more likely than random chance
        """
    )
    movies = load_movies()
    ratings = load_ratings()

    col1, col2, col3 = st.columns(3)
    with col1:
        min_support = st.slider("Min Support", 0.001, 0.05, 0.01, 0.001, format="%.3f")
    with col2:
        min_confidence = st.slider("Min Confidence", 0.1, 1.0, 0.3, 0.05)
    with col3:
        min_lift = st.slider("Min Lift", 1.0, 10.0, 1.5, 0.5)

    with st.spinner("Mining association rules..."):
        liked = ratings[ratings["rating"] >= 4]
        popular_movies = liked["movie_id"].value_counts().head(200).index
        liked_filtered = liked[liked["movie_id"].isin(popular_movies)]
        transactions_raw = liked_filtered.groupby("user_id")["movie_id"].apply(list)
        transactions_raw = transactions_raw[transactions_raw.apply(len) >= 2]

        try:
            from mlxtend.frequent_patterns import apriori, association_rules
            from mlxtend.preprocessing import TransactionEncoder

            te = TransactionEncoder()
            te_array = te.fit_transform(transactions_raw.tolist())
            te_df = pd.DataFrame(te_array, columns=te.columns_)

            freq_items = apriori(te_df, min_support=min_support, use_colnames=True, max_len=2)
            if len(freq_items) == 0:
                st.warning("No frequent itemsets found. Try lowering min support.")
                return
            rules = association_rules(freq_items, metric="lift", min_threshold=min_lift)
            rules = rules[rules["confidence"] >= min_confidence]
            rules = rules.sort_values("lift", ascending=False).head(30)

            if len(rules) == 0:
                st.warning("No rules found with current thresholds.")
                return

            def movie_name(mid):
                r = movies[movies["movie_id"] == mid]["movie_title"]
                return r.values[0] if len(r) > 0 else str(mid)

            rules["antecedents_str"] = rules["antecedents"].apply(lambda x: movie_name(list(x)[0]))
            rules["consequents_str"] = rules["consequents"].apply(lambda x: movie_name(list(x)[0]))
            display_rules = rules[["antecedents_str", "consequents_str", "support", "confidence", "lift"]]
            display_rules.columns = ["If User Liked", "Then Also Liked", "Support", "Confidence", "Lift"]
            st.success(f"Found {len(display_rules)} rules")
            st.dataframe(display_rules.reset_index(drop=True), use_container_width=True)

        except ImportError:
            st.warning("mlxtend not available. Using manual pair-based approach.")
            # Manual pair-based association rules
            from itertools import combinations
            pair_counts = {}
            item_counts = {}
            n_users = len(transactions_raw)

            for items in transactions_raw:
                items = list(set(items))
                for item in items:
                    item_counts[item] = item_counts.get(item, 0) + 1
                for a, b in combinations(items, 2):
                    pair = tuple(sorted([a, b]))
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

            rules_list = []
            for (a, b), count in pair_counts.items():
                sup = count / n_users
                if sup < min_support:
                    continue
                conf_ab = count / item_counts.get(a, 1)
                conf_ba = count / item_counts.get(b, 1)
                lift_ab = conf_ab / (item_counts.get(b, 1) / n_users)
                lift_ba = conf_ba / (item_counts.get(a, 1) / n_users)
                if conf_ab >= min_confidence and lift_ab >= min_lift:
                    rules_list.append((a, b, sup, conf_ab, lift_ab))
                if conf_ba >= min_confidence and lift_ba >= min_lift:
                    rules_list.append((b, a, sup, conf_ba, lift_ba))

            if not rules_list:
                st.warning("No rules found with current thresholds. Try lowering min values.")
                return

            rules_df = pd.DataFrame(rules_list, columns=["antecedent", "consequent", "support", "confidence", "lift"])
            rules_df = rules_df.sort_values("lift", ascending=False).head(30)

            def movie_name(mid):
                r = movies[movies["movie_id"] == mid]["movie_title"]
                return r.values[0] if len(r) > 0 else str(mid)

            rules_df["If Liked"] = rules_df["antecedent"].apply(movie_name)
            rules_df["Then Also Liked"] = rules_df["consequent"].apply(movie_name)
            st.dataframe(rules_df[["If Liked", "Then Also Liked", "support", "confidence", "lift"]], use_container_width=True)


def page_evaluation():
    st.title("Evaluation Metrics")
    st.markdown(
        """
        > **Purpose:** Quantify how well each recommendation technique performs.
        > Metrics are computed on a proper train/test split of the rating data.
        > Expensive metrics are computed on a sample and clearly labeled.
        >
        > | Metric | Formula | Interpretation |
        > |---|---|---|
        > | **RMSE** | √(mean squared error) | Lower is better; penalises large errors |
        > | **MAE** | mean absolute error | Lower is better; average prediction error |
        > | **Precision@K** | relevant in top-K / K | Higher is better; accuracy of top-K list |
        > | **Recall@K** | relevant in top-K / all relevant | Higher is better; coverage of relevant items |
        > | **Silhouette** | mean inter/intra cluster ratio | Closer to 1 = better-separated clusters |
        """
    )

    movies  = load_movies()
    ratings = load_ratings()

    with st.spinner("Running evaluation — this may take 20–40 seconds on first load..."):

        # ── Train/Test split ────────────────────────────────────
        sample_ratings = ratings.sample(min(20000, len(ratings)), random_state=42)
        train, test    = train_test_split(sample_ratings, test_size=0.2, random_state=42)
        matrix         = build_user_movie_matrix(train)

        # Build fast lookup for train ratings
        train_lookup = train.set_index(["user_id", "movie_id"])["rating"]

        all_results = []

        # ─── Mean-Centered SVD Evaluation ─────────────────────
        predicted, u_ids, m_ids = None, [], []
        try:
            predicted, u_ids, m_ids = build_svd_model(matrix)
            u_id_idx = {uid: i for i, uid in enumerate(u_ids)}
            m_id_idx = {mid: i for i, mid in enumerate(m_ids)}

            test_in_matrix = test[
                test["user_id"].isin(u_id_idx) & test["movie_id"].isin(m_id_idx)
            ]
            if len(test_in_matrix) > 0:
                u_indices = test_in_matrix["user_id"].map(u_id_idx).values
                m_indices = test_in_matrix["movie_id"].map(m_id_idx).values
                svd_preds   = predicted[u_indices, m_indices]
                svd_actuals = test_in_matrix["rating"].values
                rmse = float(np.sqrt(mean_squared_error(svd_actuals, svd_preds)))
                mae  = float(mean_absolute_error(svd_actuals, svd_preds))
                all_results.append({
                    "Method": "SVD (mean-centred)",
                    "RMSE": rmse, "MAE": mae,
                    "Precision@10": "—", "Recall@10": "—",
                    "Note": f"{len(test_in_matrix):,} test pairs",
                })
        except Exception as e:
            all_results.append({
                "Method": "SVD (mean-centred)",
                "RMSE": "Error", "MAE": str(e)[:50],
                "Precision@10": "—", "Recall@10": "—", "Note": "",
            })

        # ─── Precision & Recall @10 — SVD ─────────────────────
        try:
            K         = 10
            threshold = 4.0
            eval_users = [uid for uid in test["user_id"].unique() if uid in u_id_idx][:100]
            precisions, recalls = [], []
            skipped_no_relevant = 0
            for uid in eval_users:
                uidx       = u_id_idx[uid]
                pred_row   = predicted[uidx]
                train_seen = set(train[train["user_id"] == uid]["movie_id"].tolist())
                relevant   = set(
                    test.loc[(test["user_id"] == uid) & (test["rating"] >= threshold), "movie_id"]
                )
                if not relevant:
                    skipped_no_relevant += 1
                    continue
                candidates = [
                    (m_ids[i], pred_row[i])
                    for i in range(len(m_ids))
                    if m_ids[i] not in train_seen
                ]
                top_k_ids = {mid for mid, _ in sorted(candidates, key=lambda x: x[1], reverse=True)[:K]}
                hits = len(top_k_ids & relevant)
                precisions.append(hits / K)
                recalls.append(hits / len(relevant))

            if precisions:
                for r in all_results:
                    if r["Method"] == "SVD (mean-centred)":
                        r["Precision@10"] = f"{np.mean(precisions):.4f}"
                        r["Recall@10"]    = f"{np.mean(recalls):.4f}"
                        r["Note"]        += f" | P@10 on {len(precisions)} users"
        except Exception:
            pass

        # ─── User-Based CF Evaluation ──────────────────────────
        try:
            user_sim_mat, user_ids_list = compute_user_similarity(matrix)
            uid_to_simidx = {uid: i for i, uid in enumerate(user_ids_list)}

            cf_preds, cf_actuals = [], []
            eval_cf_users = [uid for uid in test["user_id"].unique() if uid in uid_to_simidx][:40]
            for uid in eval_cf_users:
                u_idx   = uid_to_simidx[uid]
                sim_row = user_sim_mat[u_idx]
                top_sim = np.argsort(sim_row)[::-1][1:21]
                for _, row in test[test["user_id"] == uid].iterrows():
                    mid    = row["movie_id"]
                    actual = row["rating"]
                    num, den = 0.0, 0.0
                    for si in top_sim:
                        sim_uid = user_ids_list[si]
                        w = sim_row[si]
                        if w <= 0:
                            continue
                        try:
                            r_val = train_lookup.loc[(sim_uid, mid)]
                            num += w * r_val
                            den += w
                        except KeyError:
                            pass
                    if den > 0:
                        cf_preds.append(num / den)
                        cf_actuals.append(actual)

            if cf_preds:
                rmse = float(np.sqrt(mean_squared_error(cf_actuals, cf_preds)))
                mae  = float(mean_absolute_error(cf_actuals, cf_preds))
                all_results.append({
                    "Method": "User-Based CF",
                    "RMSE": rmse, "MAE": mae,
                    "Precision@10": "—", "Recall@10": "—",
                    "Note": f"{len(eval_cf_users)}-user sample",
                })
        except Exception as e:
            all_results.append({
                "Method": "User-Based CF",
                "RMSE": "Error", "MAE": str(e)[:50],
                "Precision@10": "—", "Recall@10": "—", "Note": "",
            })

        # ─── Display Results ───────────────────────────────────
        st.subheader("Rating Prediction Metrics")
        st.caption(
            "SVD uses mean-centred matrix factorization (not raw 0-fill). "
            "Expected SVD RMSE ≈ 0.8–1.1 | MAE ≈ 0.6–0.9. "
            "Recommendation datasets are sparse; moderate metrics are expected."
        )
        if all_results:
            df_results = pd.DataFrame(all_results)
            for col in ["RMSE", "MAE"]:
                df_results[col] = df_results[col].apply(
                    lambda x: f"{x:.4f}" if isinstance(x, float) else x
                )
            st.dataframe(df_results, use_container_width=True)

        # ─── Metric interpretation callouts ───────────────────
        st.markdown("**How to read these metrics:**")
        col_i1, col_i2, col_i3 = st.columns(3)
        col_i1.info("**RMSE / MAE** — lower is better. RMSE ≈ 0.85–1.1 is good for movie ratings.")
        col_i2.info("**Precision@10 / Recall@10** — higher is better. Even 0.01–0.05 is expected for sparse data.")
        col_i3.info("**Silhouette** — closer to 1 = well-separated clusters. 0.1–0.4 is typical for genre profiles.")

        st.warning(
            "**Why Precision@10 and Recall@10 are low:** "
            "The rating dataset is extremely sparse — each user rates only ~165 of 9,742 movies (< 2%). "
            "The SVD model recommends from ~3,000 popular movies in the matrix, while a user's relevant test items "
            "may be entirely outside this pool, making hits very rare. "
            "Low P@10 here (~0.01–0.05) is expected and does not indicate a broken model — "
            "RMSE/MAE are more reliable indicators of prediction quality for sparse datasets."
        )

        # ─── K-Means Metrics ──────────────────────────────────
        st.subheader("K-Means Clustering Metrics")
        st.caption("Clustering on user × genre preference matrix (standardised). More interpretable than raw sparse ratings.")
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_score

            gmat = build_user_genre_matrix(ratings, movies, max_users=500)
            X    = StandardScaler().fit_transform(gmat.values)

            inertias, silhouettes, ks = [], [], range(2, 9)
            for ki in ks:
                if ki >= len(gmat):
                    break
                km_  = KMeans(n_clusters=ki, random_state=42, n_init=10)
                lbl_ = km_.fit_predict(X)
                inertias.append(km_.inertia_)
                try:
                    sil_ = silhouette_score(X, lbl_, sample_size=min(300, len(gmat)))
                    silhouettes.append(sil_)
                except Exception:
                    silhouettes.append(None)

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(list(ks)[:len(inertias)], inertias, marker="o", color="#e50914")
                ax.set_xlabel("K")
                ax.set_ylabel("Inertia")
                ax.set_title("Elbow Curve — choose K at the 'elbow'")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            with col2:
                valid_sil = [(ki, s) for ki, s in zip(ks, silhouettes) if s is not None]
                if valid_sil:
                    best_k, best_s = max(valid_sil, key=lambda x: x[1])
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot([ki for ki, _ in valid_sil], [s for _, s in valid_sil], marker="o", color="#564d4d")
                    ax.set_xlabel("K")
                    ax.set_ylabel("Silhouette Score")
                    ax.set_title(f"Silhouette Scores (best K={best_k}, score={best_s:.3f})")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    st.success(f"Best silhouette score: **{best_s:.4f}** at K={best_k}")

        except Exception as e:
            st.warning(f"Clustering metrics error: {e}")

        st.caption(
            "All metrics computed on a random 20% test split. "
            "SVD uses mean-centred factorization for realistic RMSE/MAE. "
            "Sample sizes shown per row."
        )


def page_movie_timeline():
    st.title("Movie Timeline")
    st.markdown(
        """
        > Plot any set of films — a director's filmography, a franchise, or a custom selection —
        > on a chronological timeline annotated with IMDb rating, genre, and community reception.
        > Instantly see how quality, style, and output evolved over the years.
        """
    )

    movies  = load_movies()
    ratings = load_ratings()
    stats   = get_movie_stats(ratings)

    movies_m = movies.merge(stats[["movie_id","avg_rating","rating_count"]], on="movie_id", how="left")
    movies_m = movies_m.dropna(subset=["release_year"])
    movies_m["release_year"] = movies_m["release_year"].astype(int)

    # ── Mode selector ────────────────────────────────────────────
    mode = st.radio(
        "Timeline mode",
        ["Director Filmography", "Franchise / Series", "Custom Selection", "Genre Snapshot"],
        horizontal=True,
        key="tl_mode",
    )
    st.markdown("---")

    selected_movies = pd.DataFrame()

    if mode == "Director Filmography":
        directors = sorted(
            movies_m["director"].dropna()
            .loc[~movies_m["director"].isin(["Unknown", ""])]
            .unique().tolist()
        )
        dir_sel = st.selectbox("Choose a director", directors, key="tl_dir")
        selected_movies = movies_m[movies_m["director"] == dir_sel].copy()
        subtitle = f"Filmography of {dir_sel}"

    elif mode == "Franchise / Series":
        keyword = st.text_input("Enter franchise keyword (e.g. 'Star Wars', 'Batman', 'Fast')", key="tl_fran")
        if keyword.strip():
            mask = movies_m["movie_title"].str.contains(keyword.strip(), case=False, na=False)
            selected_movies = movies_m[mask].copy()
            subtitle = f"Franchise: '{keyword.strip()}'"
        else:
            st.info("Enter a keyword above to build the timeline.")

    elif mode == "Custom Selection":
        titles = sorted(movies_m["movie_title"].dropna().unique().tolist())
        chosen = st.multiselect("Pick movies (up to 30)", titles, max_selections=30, key="tl_custom")
        if chosen:
            selected_movies = movies_m[movies_m["movie_title"].isin(chosen)].copy()
            subtitle = "Custom Selection"
        else:
            st.info("Select at least one movie above.")

    else:  # Genre Snapshot
        all_genres = sorted(set(
            g.strip()
            for g in "|".join(movies_m["genres"].dropna()).split("|")
            if g.strip() and g.strip() != "Unknown"
        ))
        genre_sel = st.selectbox("Choose a genre", all_genres, key="tl_genre")
        yr_min, yr_max = int(movies_m["release_year"].min()), int(movies_m["release_year"].max())
        yr_range = st.slider("Year range", yr_min, yr_max, (1990, yr_max), key="tl_yr")
        top_n = st.slider("Top N films by IMDb rating", 10, 60, 25, key="tl_topn")

        mask = (
            movies_m["genres"].str.contains(genre_sel, na=False) &
            movies_m["release_year"].between(yr_range[0], yr_range[1])
        )
        selected_movies = (
            movies_m[mask]
            .sort_values("imdb_rating", ascending=False)
            .head(top_n)
            .copy()
        )
        subtitle = f"Top {top_n} {genre_sel} films ({yr_range[0]}–{yr_range[1]})"

    if len(selected_movies) == 0:
        st.info("No movies to display yet.")
        return

    selected_movies = selected_movies.sort_values("release_year").reset_index(drop=True)

    # ── KPI strip ────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Films", len(selected_movies))
    k2.metric("Year Span",
              f"{selected_movies['release_year'].min()} – {selected_movies['release_year'].max()}")
    valid_imdb = selected_movies["imdb_rating"].dropna()
    k3.metric("Avg IMDb", f"{valid_imdb.mean():.2f}" if len(valid_imdb) > 0 else "N/A")
    valid_comm = selected_movies["avg_rating"].dropna()
    k4.metric("Avg Community", f"{valid_comm.mean():.2f}" if len(valid_comm) > 0 else "N/A")
    best = selected_movies.loc[selected_movies["imdb_rating"].idxmax()] if len(valid_imdb) > 0 else None
    k5.metric("Highest Rated", best["movie_title"][:22] + "…" if best is not None and len(best["movie_title"]) > 22 else (best["movie_title"] if best is not None else "N/A"))

    tab_tl, tab_qual, tab_genre_evo, tab_table = st.tabs([
        "Timeline Chart", "Quality Over Time", "Genre Evolution", "Data Table"
    ])

    # ── TAB 1: Timeline scatter ──────────────────────────────────
    with tab_tl:
        st.markdown(f"**{subtitle}** — each bubble is one film. Size = community vote count. Colour = IMDb rating.")

        plot_df = selected_movies.copy()
        plot_df["size"] = (
            plot_df["rating_count"].fillna(10).clip(lower=10)
        )
        plot_df["size_scaled"] = np.sqrt(plot_df["size"]) / np.sqrt(plot_df["size"].max()) * 400 + 30

        cmap  = plt.cm.RdYlGn
        imdb_vals = plot_df["imdb_rating"].fillna(5)
        norm  = plt.Normalize(vmin=imdb_vals.min(), vmax=imdb_vals.max())
        colors = cmap(norm(imdb_vals.values))

        fig, ax = plt.subplots(figsize=(13, 5))
        sc = ax.scatter(
            plot_df["release_year"],
            [0] * len(plot_df),
            s=plot_df["size_scaled"],
            c=imdb_vals,
            cmap="RdYlGn",
            vmin=imdb_vals.min(),
            vmax=imdb_vals.max(),
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

        # Horizontal timeline spine
        yr_pad = max(1, (plot_df["release_year"].max() - plot_df["release_year"].min()) * 0.05)
        ax.axhline(0, color="#cccccc", linewidth=1.2, zorder=1)
        ax.set_xlim(
            plot_df["release_year"].min() - yr_pad,
            plot_df["release_year"].max() + yr_pad,
        )
        ax.set_ylim(-1, 1.8)
        ax.set_yticks([])
        ax.set_xlabel("Release Year", fontsize=11)
        ax.set_title(subtitle, fontsize=13, fontweight="bold")

        # Labels — alternate above/below to reduce overlap
        for idx, row in plot_df.iterrows():
            yoff  = 0.35 if idx % 2 == 0 else -0.45
            ha    = "center"
            label = row["movie_title"]
            label = label[:18] + "…" if len(label) > 18 else label
            ax.annotate(
                f"{label}\n({row['imdb_rating']:.1f})" if pd.notna(row.get("imdb_rating")) else label,
                xy=(row["release_year"], 0),
                xytext=(row["release_year"], yoff),
                fontsize=7,
                ha=ha,
                va="bottom" if yoff > 0 else "top",
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.6),
            )

        plt.colorbar(sc, ax=ax, label="IMDb Rating", shrink=0.6)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── TAB 2: Quality over time ─────────────────────────────────
    with tab_qual:
        st.markdown("IMDb rating and community average rating plotted across release years.")

        fig2, ax2 = plt.subplots(figsize=(11, 4.5))
        by_year = selected_movies.groupby("release_year").agg(
            imdb_avg   = ("imdb_rating", "mean"),
            comm_avg   = ("avg_rating",  "mean"),
            film_count = ("movie_id",    "count"),
        ).reset_index()

        ax2_twin = ax2.twinx()
        ax2_twin.bar(by_year["release_year"], by_year["film_count"],
                     color="#dddddd", alpha=0.5, label="Film count", zorder=1, width=0.6)
        ax2_twin.set_ylabel("Films per year", color="#888888")
        ax2_twin.tick_params(axis="y", labelcolor="#888888")

        ax2.plot(by_year["release_year"], by_year["imdb_avg"],
                 marker="o", color="#f5a623", linewidth=2, markersize=5, label="IMDb Avg", zorder=3)
        ax2.plot(by_year["release_year"], by_year["comm_avg"],
                 marker="s", color="#4a90d9", linewidth=2, markersize=5,
                 linestyle="--", label="Community Avg", zorder=3)

        ax2.set_xlabel("Release Year")
        ax2.set_ylabel("Rating")
        ax2.set_title("Quality Over Time")
        ax2.set_ylim(0, 10)
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower left")
        for spine in ["top"]:
            ax2.spines[spine].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # Peak / trough callout
        if len(by_year.dropna(subset=["imdb_avg"])) > 0:
            peak_row  = by_year.loc[by_year["imdb_avg"].idxmax()]
            trough_row = by_year.loc[by_year["imdb_avg"].idxmin()]
            c1, c2 = st.columns(2)
            c1.success(f"Peak year: **{int(peak_row['release_year'])}** — avg IMDb {peak_row['imdb_avg']:.2f}")
            c2.error(f"Weakest year: **{int(trough_row['release_year'])}** — avg IMDb {trough_row['imdb_avg']:.2f}")

    # ── TAB 3: Genre evolution ───────────────────────────────────
    with tab_genre_evo:
        st.markdown("How the genre mix of this selection shifts decade by decade.")

        evo_rows = []
        for _, row in selected_movies.iterrows():
            decade = str(int(row["release_year"] // 10 * 10)) + "s"
            for g in str(row.get("genres","")).split("|"):
                g = g.strip()
                if g and g != "Unknown":
                    evo_rows.append({"decade": decade, "genre": g})

        if not evo_rows:
            st.info("No genre data available.")
        else:
            evo_df = pd.DataFrame(evo_rows)
            pivot  = evo_df.groupby(["decade","genre"]).size().unstack(fill_value=0)
            pivot  = pivot.div(pivot.sum(axis=1), axis=0)  # normalise to proportion

            top_genres = evo_df["genre"].value_counts().head(10).index.tolist()
            pivot = pivot.reindex(columns=[g for g in top_genres if g in pivot.columns])

            fig3, ax3 = plt.subplots(figsize=(10, 4.5))
            pivot.plot(kind="bar", stacked=True, ax=ax3,
                       colormap="tab10", width=0.65, edgecolor="white")
            ax3.set_xlabel("Decade")
            ax3.set_ylabel("Genre proportion")
            ax3.set_title("Genre Mix by Decade")
            ax3.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

    # ── TAB 4: Data table ────────────────────────────────────────
    with tab_table:
        display_cols = ["movie_title","release_year","genres","director",
                        "imdb_rating","avg_rating","rating_count"]
        display_cols = [c for c in display_cols if c in selected_movies.columns]
        tbl = selected_movies[display_cols].rename(columns={
            "movie_title": "Title", "release_year": "Year",
            "genres": "Genres", "director": "Director",
            "imdb_rating": "IMDb", "avg_rating": "Community Avg",
            "rating_count": "Votes",
        })
        st.dataframe(tbl, use_container_width=True)

        csv = tbl.to_csv(index=False).encode("utf-8")
        safe = subtitle.replace("/","_").replace(" ","_")[:40]
        st.download_button(
            "⬇ Download as CSV",
            data=csv,
            file_name=f"timeline_{safe}.csv",
            mime="text/csv",
            key="tl_export",
            use_container_width=True,
        )


def page_serendipity():
    st.title("Serendipity Recommender")
    st.markdown(
        """
        > **Data Mining Concept:** Standard recommenders optimise for predicted accuracy — they
        > keep returning films close to what you already like, creating a *filter bubble*.
        > **Serendipity** deliberately broadens the search: it targets high-quality films that sit
        > *outside* your genre comfort zone, scored by a combined formula:
        >
        > `Serendipity Score = α × Novelty + (1 − α) × Predicted Quality`
        >
        > Novelty = genre distance from the user's taste profile.
        > Predicted Quality = IMDb rating + community rating (normalised).
        > α (the slider) controls how adventurous the recommendations are.
        """
    )

    movies  = load_movies()
    ratings = load_ratings()
    stats   = get_movie_stats(ratings)

    all_users = sorted(ratings["user_id"].dropna().unique().tolist())

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_user = st.selectbox("Select User ID", all_users, key="seren_user")
    with col2:
        alpha = st.slider("α — Novelty weight", 0.0, 1.0, 0.5, 0.05, key="seren_alpha",
                          help="0 = pure quality, 1 = pure novelty, 0.5 = balanced")
    with col3:
        n_recs = st.slider("Recommendations", 5, 30, 12, key="seren_n")

    user_r = ratings[ratings["user_id"] == selected_user]
    if len(user_r) == 0:
        st.warning("No ratings found for this user.")
        return

    already_seen = set(user_r["movie_id"].tolist())

    # ── Build user genre taste profile ───────────────────────────
    user_r_m = user_r.merge(movies[["movie_id","genres","release_year"]], on="movie_id", how="left")
    genre_counts: dict = {}
    for _, row in user_r_m.iterrows():
        w = row["rating"] / 5.0  # weight by rating
        for g in str(row.get("genres","")).split("|"):
            g = g.strip()
            if g and g != "Unknown":
                genre_counts[g] = genre_counts.get(g, 0) + w

    total_w = sum(genre_counts.values()) or 1
    taste_vec = {g: v / total_w for g, v in genre_counts.items()}

    all_genres = sorted(taste_vec.keys() | set(
        g.strip()
        for g in "|".join(movies["genres"].dropna()).split("|")
        if g.strip() and g.strip() != "Unknown"
    ))

    taste_arr = np.array([taste_vec.get(g, 0) for g in all_genres])
    if taste_arr.sum() > 0:
        taste_arr /= taste_arr.sum()

    # ── Score unseen movies ───────────────────────────────────────
    candidates = movies[~movies["movie_id"].isin(already_seen)].copy()
    candidates = candidates.merge(stats[["movie_id","avg_rating","rating_count"]], on="movie_id", how="left")

    # Genre vector for each candidate
    def movie_genre_vec(genres_str):
        tags = [g.strip() for g in str(genres_str).split("|") if g.strip() in all_genres]
        vec  = np.zeros(len(all_genres))
        if tags:
            for t in tags:
                vec[all_genres.index(t)] = 1 / len(tags)
        return vec

    novelty_scores   = []
    quality_scores   = []

    max_comm = candidates["avg_rating"].max() if candidates["avg_rating"].notna().any() else 5
    max_imdb = candidates["imdb_rating"].max() if candidates["imdb_rating"].notna().any() else 10

    for _, row in candidates.iterrows():
        gvec = movie_genre_vec(row.get("genres",""))
        # Novelty = 1 − cosine_sim(user_taste, movie_genre)
        dot = float(np.dot(taste_arr, gvec))
        norm_prod = (np.linalg.norm(taste_arr) * np.linalg.norm(gvec))
        cos_sim = dot / norm_prod if norm_prod > 0 else 0
        novelty = 1.0 - cos_sim  # 0 = identical to taste, 1 = totally foreign

        # Quality = average of normalised IMDb + community avg
        q_imdb = (row["imdb_rating"] / max_imdb) if pd.notna(row.get("imdb_rating")) else 0.5
        q_comm = (row["avg_rating"]  / max_comm)  if pd.notna(row.get("avg_rating"))  else 0.5
        quality = (q_imdb + q_comm) / 2

        novelty_scores.append(novelty)
        quality_scores.append(quality)

    candidates["novelty"]  = novelty_scores
    candidates["quality"]  = quality_scores
    candidates["seren_score"] = alpha * candidates["novelty"] + (1 - alpha) * candidates["quality"]

    # Require at least moderate quality
    min_quality = st.slider("Min quality threshold (0–1)", 0.0, 1.0, 0.35, 0.05, key="seren_minq")
    results = (
        candidates[candidates["quality"] >= min_quality]
        .sort_values("seren_score", ascending=False)
        .head(n_recs)
        .reset_index(drop=True)
    )

    if len(results) == 0:
        st.warning("No results — try lowering the quality threshold.")
        return

    # ── KPI bar ──────────────────────────────────────────────────
    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Movies Already Seen", f"{len(already_seen):,}")
    k2.metric("Candidate Pool", f"{len(candidates):,}")
    k3.metric("After Quality Filter", f"{len(candidates[candidates['quality']>=min_quality]):,}")
    k4.metric("Avg Novelty (top results)", f"{results['novelty'].mean():.2f}")

    dominant_taste = max(taste_vec, key=taste_vec.get) if taste_vec else "N/A"
    st.info(
        f"User {selected_user}'s dominant genre taste: **{dominant_taste}** "
        f"({taste_vec.get(dominant_taste,0)*100:.0f}% of weighted profile). "
        f"Serendipitous picks will lean away from this."
    )

    # ── Taste profile chart ───────────────────────────────────────
    with st.expander("View your Genre Taste Profile"):
        taste_df = (
            pd.Series(taste_vec)
            .sort_values(ascending=False)
            .head(15)
        )
        fig0, ax0 = plt.subplots(figsize=(8, 3.5))
        colors0 = plt.cm.Blues_r(np.linspace(0.2, 0.75, len(taste_df)))
        ax0.barh(taste_df.index[::-1], taste_df.values[::-1], color=colors0[::-1])
        ax0.set_xlabel("Weighted Genre Share")
        ax0.set_title(f"Genre Taste Profile — User {selected_user}")
        for spine in ["top","right"]:
            ax0.spines[spine].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig0)
        plt.close()

    # ── Results tabs ─────────────────────────────────────────────
    tab_cards, tab_scatter, tab_table = st.tabs(["Recommendations", "Score Explorer", "Full Table"])

    with tab_cards:
        st.markdown(
            f"Showing **{len(results)}** serendipitous picks "
            f"(α={alpha:.2f} — {'balanced' if 0.4<=alpha<=0.6 else 'novelty-heavy' if alpha>0.6 else 'quality-heavy'})."
        )
        for _, row in results.iterrows():
            with st.container():
                c1, c2 = st.columns([4, 1])
                with c1:
                    title = row.get("movie_title","Unknown")
                    year  = int(row["release_year"]) if pd.notna(row.get("release_year")) else ""
                    st.markdown(f"**{title}** ({year})")
                    st.caption(f"Genres: {row.get('genres','')}")
                    if pd.notna(row.get("director")) and row["director"] not in ["Unknown",""]:
                        st.caption(f"Director: {row['director']}")
                with c2:
                    st.metric("Serendipity",   f"{row['seren_score']:.3f}")
                    st.metric("Novelty",        f"{row['novelty']:.3f}")
                    st.metric("Quality",        f"{row['quality']:.3f}")
                    if pd.notna(row.get("imdb_rating")):
                        st.metric("IMDb", f"⭐ {row['imdb_rating']:.1f}")

                # Why serendipitous?
                movie_genres = [g.strip() for g in str(row.get("genres","")).split("|")
                                if g.strip() and g.strip() != "Unknown"]
                familiar = [g for g in movie_genres if g in taste_vec and taste_vec[g] > 0.05]
                foreign  = [g for g in movie_genres if g not in taste_vec or taste_vec[g] <= 0.02]
                reason_parts = []
                if foreign:
                    reason_parts.append(f"New territory: {', '.join(foreign[:3])}")
                if familiar:
                    reason_parts.append(f"Familiar anchor: {', '.join(familiar[:2])}")
                if reason_parts:
                    st.success(" · ".join(reason_parts))
                st.divider()

    with tab_scatter:
        st.markdown(
            "Each dot is a candidate film. X = Novelty, Y = Quality. "
            "Top recommendations highlighted in red — they score well on both axes given your α."
        )
        top_ids = set(results["movie_id"].tolist())
        sample  = candidates.sample(min(2000, len(candidates)), random_state=42)

        fig1, ax1 = plt.subplots(figsize=(9, 6))
        mask_top  = sample["movie_id"].isin(top_ids)
        ax1.scatter(
            sample[~mask_top]["novelty"],
            sample[~mask_top]["quality"],
            alpha=0.25, s=12, color="steelblue", label="Other candidates"
        )
        ax1.scatter(
            sample[mask_top]["novelty"],
            sample[mask_top]["quality"],
            alpha=0.85, s=55, color="#e50914", zorder=5, label="Top picks"
        )
        # Decision line: seren_score = 0.5 → α*nov + (1-α)*qual = 0.5
        if 0 < alpha < 1:
            nov_line = np.linspace(0, 1, 100)
            qual_line = (0.5 - alpha * nov_line) / (1 - alpha)
            ax1.plot(nov_line, qual_line, "--", color="orange", linewidth=1.2, label="Score = 0.5 line")
        ax1.set_xlabel("Novelty (genre distance from taste)")
        ax1.set_ylabel("Quality (normalised IMDb + community)")
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_title(f"Serendipity Landscape — User {selected_user} (α={alpha:.2f})")
        ax1.legend(fontsize=9)
        for spine in ["top","right"]:
            ax1.spines[spine].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

    with tab_table:
        display_cols = ["movie_title","release_year","genres","director",
                        "imdb_rating","avg_rating","novelty","quality","seren_score"]
        display_cols = [c for c in display_cols if c in results.columns]
        tbl = results[display_cols].copy()
        tbl.columns = [c.replace("_"," ").title() for c in tbl.columns]
        tbl["Novelty"]     = tbl["Novelty"].round(4)
        tbl["Quality"]     = tbl["Quality"].round(4)
        tbl["Seren Score"] = tbl["Seren Score"].round(4)
        st.dataframe(tbl, use_container_width=True)

        csv = tbl.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download serendipity picks as CSV",
            data=csv,
            file_name=f"serendipity_user{selected_user}_alpha{alpha:.2f}.csv",
            mime="text/csv",
            key="seren_export",
            use_container_width=True,
        )


def page_rating_bias():
    st.title("Rating Bias Analyser")
    st.markdown(
        """
        > **Data Mining Concept:** Every user carries implicit rating bias — they may consistently
        > rate certain genres, directors, or eras higher or lower than the community average.
        > Identifying these biases improves recommendation quality by calibrating predicted scores.
        >
        > Each chart shows the user's average rating **minus** the community average for the same
        > category. Positive = rates higher than community; Negative = rates lower.
        """
    )

    movies  = load_movies()
    ratings = load_ratings()

    all_users = sorted(ratings["user_id"].dropna().unique().tolist())
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_user = st.selectbox("Select User ID", all_users, key="bias_user")
    with col2:
        min_items = st.slider("Min ratings to include a category", 2, 20, 3, key="bias_min")

    user_r = ratings[ratings["user_id"] == selected_user]
    if len(user_r) == 0:
        st.warning("No ratings found for this user.")
        return

    user_r = user_r.merge(
        movies[["movie_id", "genres", "director", "release_year"]],
        on="movie_id", how="left"
    )

    # ── Overall leniency ─────────────────────────────────────────
    st.markdown("---")
    community_avg = ratings["rating"].mean()
    user_avg      = user_r["rating"].mean()
    bias_overall  = user_avg - community_avg

    ka, kb, kc, kd = st.columns(4)
    ka.metric("User Avg Rating",      f"{user_avg:.3f}")
    kb.metric("Community Avg Rating", f"{community_avg:.3f}")
    kc.metric("Overall Bias",         f"{bias_overall:+.3f}",
              delta_color="normal" if bias_overall >= 0 else "inverse")
    if abs(bias_overall) < 0.1:
        label = "Calibrated rater"
    elif bias_overall > 0.4:
        label = "Generous rater"
    elif bias_overall > 0.1:
        label = "Slightly lenient"
    elif bias_overall < -0.4:
        label = "Harsh rater"
    else:
        label = "Slightly strict"
    kd.metric("Profile", label)

    def bias_chart(ax, bias_series, title, color_pos="#2ca02c", color_neg="#e50914"):
        bias_series = bias_series.sort_values()
        colors = [color_pos if v >= 0 else color_neg for v in bias_series.values]
        ax.barh(bias_series.index, bias_series.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Bias (user avg − community avg)")
        ax.set_title(title)
        ax.set_xlim(-3, 3)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    tab_genre, tab_director, tab_era, tab_rating_dist, tab_table = st.tabs([
        "Genre Bias", "Director Bias", "Era Bias", "Rating Distribution", "Full Table"
    ])

    # ── TAB 1: Genre bias ────────────────────────────────────────
    with tab_genre:
        st.markdown(
            "Green bars = user rates this genre **above** community average. "
            "Red bars = rates **below** average."
        )
        genre_rows = []
        for _, row in user_r.iterrows():
            for g in str(row["genres"]).split("|"):
                g = g.strip()
                if g and g != "Unknown":
                    genre_rows.append({"genre": g, "rating": row["rating"]})

        if not genre_rows:
            st.info("No genre data available.")
        else:
            gdf       = pd.DataFrame(genre_rows)
            user_gavg = gdf.groupby("genre")["rating"].agg(["mean", "count"])
            user_gavg = user_gavg[user_gavg["count"] >= min_items]

            # Community genre averages
            comm_rows = []
            for _, row in ratings.merge(movies[["movie_id","genres"]], on="movie_id", how="left").iterrows():
                for g in str(row.get("genres","")).split("|"):
                    g = g.strip()
                    if g and g != "Unknown":
                        comm_rows.append({"genre": g, "rating": row["rating"]})
            comm_gdf  = pd.DataFrame(comm_rows)
            comm_gavg = comm_gdf.groupby("genre")["rating"].mean()

            bias_genre = (user_gavg["mean"] - comm_gavg).dropna().sort_values()
            bias_genre = bias_genre[user_gavg["count"] >= min_items]

            fig, ax = plt.subplots(figsize=(9, max(4, len(bias_genre) * 0.45)))
            bias_chart(ax, bias_genre, f"Genre Rating Bias — User {selected_user}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Callouts
            if len(bias_genre) > 0:
                most_loved  = bias_genre.idxmax()
                most_harsh  = bias_genre.idxmin()
                c1, c2 = st.columns(2)
                c1.success(f"Rates **{most_loved}** most generously vs community (+{bias_genre.max():.2f})")
                c2.error(f"Rates **{most_harsh}** most harshly vs community ({bias_genre.min():.2f})")

    # ── TAB 2: Director bias ─────────────────────────────────────
    with tab_director:
        st.markdown("How does this user rate each director compared to the community?")
        user_d = user_r[user_r["director"].notna() & ~user_r["director"].isin(["Unknown",""])].copy()
        user_davg = user_d.groupby("director")["rating"].agg(["mean","count"])
        user_davg = user_davg[user_davg["count"] >= min_items]

        if len(user_davg) == 0:
            st.info("Not enough director data with current minimum.")
        else:
            comm_dir = (
                ratings
                .merge(movies[["movie_id","director"]], on="movie_id", how="left")
                .groupby("director")["rating"].mean()
            )
            bias_dir = (user_davg["mean"] - comm_dir).dropna()
            bias_dir = bias_dir[user_davg["count"] >= min_items].sort_values()
            bias_dir = bias_dir.iloc[max(0, len(bias_dir)-20):]  # top 20 by count

            fig, ax = plt.subplots(figsize=(9, max(4, len(bias_dir) * 0.45)))
            bias_chart(ax, bias_dir, f"Director Rating Bias — User {selected_user}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            if len(bias_dir) > 0:
                fav_dir  = bias_dir.idxmax()
                cold_dir = bias_dir.idxmin()
                c1, c2 = st.columns(2)
                c1.success(f"Rates **{fav_dir}** most generously (+{bias_dir.max():.2f})")
                c2.error(f"Rates **{cold_dir}** most harshly ({bias_dir.min():.2f})")

    # ── TAB 3: Era bias ──────────────────────────────────────────
    with tab_era:
        st.markdown("Does this user prefer older or newer films compared to the community?")
        user_era = user_r.dropna(subset=["release_year"]).copy()
        user_era["decade"] = (user_era["release_year"] // 10 * 10).astype(int).astype(str) + "s"
        user_eavg = user_era.groupby("decade")["rating"].agg(["mean","count"])
        user_eavg = user_eavg[user_eavg["count"] >= min_items]

        if len(user_eavg) == 0:
            st.info("Not enough data per decade with current minimum.")
        else:
            all_r_era = ratings.merge(movies[["movie_id","release_year"]], on="movie_id", how="left")
            all_r_era = all_r_era.dropna(subset=["release_year"])
            all_r_era["decade"] = (all_r_era["release_year"] // 10 * 10).astype(int).astype(str) + "s"
            comm_eavg = all_r_era.groupby("decade")["rating"].mean()

            bias_era = (user_eavg["mean"] - comm_eavg).dropna().sort_index()
            bias_era = bias_era[user_eavg["count"] >= min_items]

            fig, ax = plt.subplots(figsize=(9, max(3, len(bias_era) * 0.55)))
            bias_chart(ax, bias_era, f"Era Rating Bias — User {selected_user}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            if len(bias_era) > 0:
                fav_era  = bias_era.idxmax()
                cold_era = bias_era.idxmin()
                c1, c2 = st.columns(2)
                c1.success(f"Most generous with **{fav_era}** films (+{bias_era.max():.2f})")
                c2.error(f"Most critical of **{cold_era}** films ({bias_era.min():.2f})")

    # ── TAB 4: Rating distribution ───────────────────────────────
    with tab_rating_dist:
        st.markdown("User's rating distribution vs community — reveals polarising or clustering tendencies.")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # User distribution
        user_counts = user_r["rating"].value_counts().sort_index()
        axes[0].bar(user_counts.index, user_counts.values / user_counts.sum(), color="#e50914", width=0.4)
        axes[0].set_xlabel("Rating")
        axes[0].set_ylabel("Proportion")
        axes[0].set_title(f"User {selected_user} Rating Distribution")
        axes[0].set_xlim(0.25, 5.25)

        # Community distribution
        comm_counts = ratings["rating"].value_counts().sort_index()
        axes[1].bar(comm_counts.index, comm_counts.values / comm_counts.sum(), color="#1f77b4", width=0.4)
        axes[1].set_xlabel("Rating")
        axes[1].set_ylabel("Proportion")
        axes[1].set_title("Community Rating Distribution")
        axes[1].set_xlim(0.25, 5.25)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Stats comparison
        st.markdown("**Statistical Summary**")
        summary = pd.DataFrame({
            "Metric": ["Mean", "Median", "Std Dev", "% 5-star", "% 1-star"],
            f"User {selected_user}": [
                f"{user_r['rating'].mean():.3f}",
                f"{user_r['rating'].median():.1f}",
                f"{user_r['rating'].std():.3f}",
                f"{(user_r['rating']==5).mean()*100:.1f}%",
                f"{(user_r['rating']==1).mean()*100:.1f}%",
            ],
            "Community": [
                f"{ratings['rating'].mean():.3f}",
                f"{ratings['rating'].median():.1f}",
                f"{ratings['rating'].std():.3f}",
                f"{(ratings['rating']==5).mean()*100:.1f}%",
                f"{(ratings['rating']==1).mean()*100:.1f}%",
            ],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    # ── TAB 5: Full bias table ───────────────────────────────────
    with tab_table:
        st.markdown("Complete genre bias breakdown with counts and raw averages.")
        if "bias_genre" in dir() and len(bias_genre) > 0:
            tbl = pd.DataFrame({
                "Genre":           bias_genre.index,
                "User Avg":        user_gavg.loc[bias_genre.index, "mean"].round(3).values,
                "Community Avg":   comm_gavg.reindex(bias_genre.index).round(3).values,
                "Bias":            bias_genre.round(3).values,
                "User Rating Count": user_gavg.loc[bias_genre.index, "count"].values,
            }).sort_values("Bias", ascending=False).reset_index(drop=True)
            st.dataframe(tbl, use_container_width=True)

            csv = tbl.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download bias table as CSV",
                data=csv,
                file_name=f"rating_bias_user{selected_user}.csv",
                mime="text/csv",
                key="bias_export",
                use_container_width=True,
            )
        else:
            st.info("Navigate to the Genre Bias tab first to populate this table.")


@st.cache_data
def build_genre_matrix(movies_df):
    """
    Genre DNA matrix using TF-IDF-style genre weighting.
    Each row = one movie; each column = a genre.
    Weight = IDF(genre) for genres the movie belongs to, 0 elsewhere.
    IDF = log(N / df_genre + 1) so rarer genres score higher than ubiquitous ones.
    Each row is L2-normalised so cosine similarity is meaningful.
    Index is always reset to 0-based so it aligns with movies.reset_index(drop=True).
    """
    movies_clean = movies_df.reset_index(drop=True)
    n_movies = len(movies_clean)

    all_genres = sorted(set(
        g.strip()
        for g in "|".join(movies_clean["genres"].dropna()).split("|")
        if g.strip() and g.strip() not in ["Unknown", "(no genres listed)", ""]
    ))
    genre_to_idx = {g: i for i, g in enumerate(all_genres)}

    # Count how many movies contain each genre (document frequency)
    df_counts = np.zeros(len(all_genres))
    movie_tags = []
    for genres_str in movies_clean["genres"]:
        tags = [
            g.strip() for g in str(genres_str).split("|")
            if g.strip() in genre_to_idx
        ]
        movie_tags.append(tags)
        for t in set(tags):
            df_counts[genre_to_idx[t]] += 1

    # IDF: log((N + 1) / (df + 1)) + 1  (smoothed, always positive)
    idf = np.log((n_movies + 1) / (df_counts + 1)) + 1.0

    # Build weight matrix: for each movie, set active genres to their IDF weight
    mat = np.zeros((n_movies, len(all_genres)), dtype=np.float32)
    for row_i, tags in enumerate(movie_tags):
        for t in tags:
            mat[row_i, genre_to_idx[t]] = idf[genre_to_idx[t]]

    # L2-normalise each row so cosine similarity works correctly
    row_norms = np.linalg.norm(mat, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    mat = mat / row_norms

    gmat = pd.DataFrame(mat, columns=all_genres)
    return gmat, all_genres


def page_genre_dna():
    st.title("Genre DNA")
    st.markdown(
        """
        > Every movie has a unique **genre fingerprint** — not just a list of tags, but a *weighted signature*
        > reflecting how *rare* each of its genres is across the full catalog.
        > This page uses **TF-IDF genre weighting**: common genres like Drama get lower weight,
        > while rare genres like Film-Noir or Documentary are scored higher, making each film's DNA distinct.
        >
        > Similarity search uses cosine distance between L2-normalised genre vectors, so a pure Film-Noir
        > matches very differently from a Drama-Romance-Comedy hybrid, even if both share a single tag.
        """
    )

    movies = load_movies()
    ratings = load_ratings()
    stats   = get_movie_stats(ratings)

    gmat, all_genres = build_genre_matrix(movies)
    movies_idx = movies.reset_index(drop=True)

    # ── Controls ─────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        titles     = sorted(movies_idx["movie_title"].dropna().unique().tolist())
        sel_title  = st.selectbox("Select a movie to analyse", titles, key="dna_movie")
    with col2:
        n_matches  = st.slider("Matches to show", 5, 25, 12, key="dna_n")

    sel_row = movies_idx[movies_idx["movie_title"] == sel_title]
    if len(sel_row) == 0:
        st.warning("Movie not found.")
        return

    sel_idx  = sel_row.index[0]
    sel_vec  = gmat.iloc[sel_idx].values          # genre blend vector
    sel_info = sel_row.iloc[0]

    # ── Debug section ────────────────────────────────────────────
    with st.expander("Debug: Genre Vector Inspection", expanded=False):
        raw_genres_str = sel_info.get("genres", "Unknown")
        st.markdown(f"**Raw genres field:** `{raw_genres_str}`")
        parsed_tags = [g.strip() for g in str(raw_genres_str).split("|") if g.strip() and g.strip() not in ["Unknown", ""]]
        st.markdown(f"**Parsed genre tags:** {parsed_tags}")
        dna_debug = pd.Series(sel_vec, index=all_genres)
        active_entries = dna_debug[dna_debug > 0].sort_values(ascending=False)
        if len(active_entries) > 0:
            st.markdown(f"**Non-zero genre vector entries ({len(active_entries)} genres active):**")
            st.dataframe(active_entries.rename("Weight").to_frame().style.format("{:.4f}"), use_container_width=True)
            zero_count = (dna_debug == 0).sum()
            st.caption(f"{zero_count} of {len(all_genres)} genres are zero (as expected for genres this movie does not have).")
        else:
            st.warning("All entries are zero — the genres field may be 'Unknown' or empty.")

    # ── DNA fingerprint ─────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"Genre DNA — {sel_title}")

    dna_series = pd.Series(sel_vec, index=all_genres)
    dna_active = dna_series[dna_series > 0].sort_values(ascending=False)

    col_chart, col_meta = st.columns([2, 1])

    with col_chart:
        if len(dna_active) == 0:
            st.info("No genre data available for this movie.")
        else:
            # Radar / bar — use bar for reliability
            fig, ax = plt.subplots(figsize=(7, 3.5))
            bar_colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(dna_active)))
            ax.barh(dna_active.index[::-1], dna_active.values[::-1], color=bar_colors[::-1])
            ax.set_xlabel("Genre Weight (TF-IDF, L2-normalised — higher = rarer genre)")
            ax.set_title("Genre DNA Fingerprint")
            ax.set_xlim(0, dna_active.max() * 1.25)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Pie chart for quick visual
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            wedge_colors = plt.cm.Set3(np.linspace(0, 1, len(dna_active)))
            ax2.pie(
                dna_active.values,
                labels=dna_active.index,
                colors=wedge_colors,
                autopct="%1.0f%%",
                startangle=140,
                textprops={"fontsize": 9},
            )
            ax2.set_title("Genre Mix")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

    with col_meta:
        st.markdown("**Movie info**")
        year = int(sel_info["release_year"]) if pd.notna(sel_info.get("release_year")) else "N/A"
        st.markdown(f"**Year:** {year}")
        if pd.notna(sel_info.get("imdb_rating")):
            st.markdown(f"**IMDb:** ⭐ {sel_info['imdb_rating']:.1f}")
        if pd.notna(sel_info.get("director")) and sel_info["director"] not in ["Unknown", ""]:
            st.markdown(f"**Director:** {sel_info['director']}")
        merged_stats = stats[stats["movie_id"] == sel_info["movie_id"]]
        if len(merged_stats) > 0:
            st.markdown(f"**ML Avg Rating:** {merged_stats.iloc[0]['avg_rating']:.2f}")
            st.markdown(f"**ML Votes:** {int(merged_stats.iloc[0]['rating_count']):,}")

        st.markdown("---")
        st.markdown("**Rarest / Highest-weight genre**")
        if len(dna_active) > 0:
            dominant = dna_active.index[0]
            st.markdown(f"`{dominant}`")
            st.caption("This genre scores highest because it is the rarest across the catalog.")
        st.markdown("**Genre count**")
        st.markdown(f"{len(dna_active)} genre tag(s)")
        st.markdown("**Blend type**")
        if len(dna_active) == 1:
            blend_type = "Pure"
        elif len(dna_active) == 2:
            blend_type = "Dual"
        elif dna_active.iloc[0] > 0.6:
            blend_type = "Dominant + Accents"
        else:
            blend_type = "Multi-Genre Blend"
        st.markdown(f"`{blend_type}`")

    # ── Similarity search ────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"Closest Genre DNA Matches — {n_matches} results")
    st.caption(
        "Ranked by cosine similarity between genre blend vectors. "
        "A score of 1.0 = identical composition."
    )

    # Optional filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        min_imdb = st.slider("Min IMDb rating", 0.0, 9.0, 0.0, 0.5, key="dna_imdb")
    with fc2:
        if "release_year" in movies_idx.columns:
            yr_vals  = movies_idx["release_year"].dropna()
            yr_min, yr_max = int(yr_vals.min()), int(yr_vals.max())
            year_range = st.slider("Release year range", yr_min, yr_max, (yr_min, yr_max), key="dna_year")
        else:
            year_range = None
    with fc3:
        same_director = st.checkbox("Same director only", value=False, key="dna_dir")

    # Compute cosine similarity on genre matrix
    sim_vec = cosine_similarity(gmat.iloc[[sel_idx]], gmat).flatten()
    sim_vec[sel_idx] = 0  # exclude self

    # Apply filters to candidate pool
    candidate_mask = pd.Series(True, index=movies_idx.index)
    if min_imdb > 0:
        candidate_mask &= movies_idx["imdb_rating"].fillna(0) >= min_imdb
    if year_range:
        candidate_mask &= (
            movies_idx["release_year"].fillna(0).between(year_range[0], year_range[1])
        )
    if same_director and pd.notna(sel_info.get("director")) and sel_info["director"] not in ["Unknown",""]:
        candidate_mask &= movies_idx["director"] == sel_info["director"]

    filtered_sim = sim_vec.copy()
    filtered_sim[~candidate_mask] = 0

    top_indices = filtered_sim.argsort()[::-1][:n_matches]

    if filtered_sim[top_indices[0]] == 0:
        st.info("No matches found with the current filters.")
    else:
        # Comparison table
        match_rows = []
        for rank, i in enumerate(top_indices, 1):
            if filtered_sim[i] == 0:
                break
            r = movies_idx.iloc[i]
            match_dna = gmat.iloc[i]
            match_active = match_dna[match_dna > 0]
            match_rows.append({
                "Rank":        rank,
                "Title":       r.get("movie_title", ""),
                "Year":        int(r["release_year"]) if pd.notna(r.get("release_year")) else "",
                "Genres":      r.get("genres", ""),
                "IMDb":        f"{r['imdb_rating']:.1f}" if pd.notna(r.get("imdb_rating")) else "",
                "Director":    r.get("director", ""),
                "DNA Score":   round(float(filtered_sim[i]), 4),
                "Genre Count": int(len(match_active)),
            })

        match_df = pd.DataFrame(match_rows)
        st.dataframe(match_df, use_container_width=True)

        # Card view
        st.markdown("#### Top Matches")
        for i in top_indices[:6]:
            if filtered_sim[i] == 0:
                break
            r = movies_idx.iloc[i]
            display_movie_card(r, "DNA Score", float(filtered_sim[i]))

        # Genre overlap heatmap (selected vs top 8 matches)
        st.markdown("---")
        st.subheader("Genre Overlap Heatmap")
        st.caption("Rows = top matches, columns = genres. Colour = blend weight.")

        heat_indices = [i for i in top_indices[:8] if filtered_sim[i] > 0]
        if heat_indices:
            heat_titles = [movies_idx.iloc[i].get("movie_title", str(i))[:28] for i in heat_indices]
            heat_vecs   = gmat.iloc[heat_indices][list(dna_active.index)].values

            fig3, ax3 = plt.subplots(figsize=(max(8, len(dna_active)), max(3, len(heat_indices) * 0.6)))
            im3 = ax3.imshow(heat_vecs, aspect="auto", cmap="YlOrRd")
            ax3.set_xticks(range(len(dna_active)))
            ax3.set_xticklabels(dna_active.index, rotation=45, ha="right", fontsize=9)
            ax3.set_yticks(range(len(heat_titles)))
            ax3.set_yticklabels(heat_titles, fontsize=9)
            ax3.set_title(f"Genre DNA — Top Matches vs '{sel_title[:30]}'")
            plt.colorbar(im3, ax=ax3, label="Genre Weight")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

        # CSV export
        if match_rows:
            csv = match_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download matches as CSV",
                data=csv,
                file_name=f"genre_dna_{sel_title[:30].replace(' ','_')}.csv",
                mime="text/csv",
                key="dna_export",
                use_container_width=True,
            )


def page_popularity_trends():
    st.title("Movie Popularity Trends")
    st.markdown(
        """
        > Explore how movies and genres rise and fall in popularity over time using the
        > **timestamp data** embedded in the ratings dataset. All charts use real rating activity —
        > not release dates — so they reflect when audiences actually engaged with each film.
        """
    )

    movies  = load_movies()
    ratings = load_ratings()

    # Ensure time columns exist
    has_year  = "year"  in ratings.columns
    has_month = "month" in ratings.columns

    if not has_year:
        if "rating_datetime" in ratings.columns:
            ratings["year"]  = pd.to_datetime(ratings["rating_datetime"], errors="coerce").dt.year
            ratings["month"] = pd.to_datetime(ratings["rating_datetime"], errors="coerce").dt.month
            has_year  = True
            has_month = True
        elif "timestamp" in ratings.columns:
            ratings["year"]  = pd.to_datetime(ratings["timestamp"], unit="s", errors="coerce").dt.year
            ratings["month"] = pd.to_datetime(ratings["timestamp"], unit="s", errors="coerce").dt.month
            has_year  = True
            has_month = True

    if not has_year:
        st.error("No time information found in the ratings dataset. Cannot generate trend charts.")
        return

    ratings_m = ratings.merge(movies[["movie_id", "movie_title", "genres", "release_year"]], on="movie_id", how="left")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Movie Over Time",
        "Genre Trends",
        "Rating Heatmap",
        "Breakout Tracker",
    ])

    # ── TAB 1: Single-movie popularity timeline ──────────────────
    with tab1:
        st.markdown("### Rating Activity for Specific Movies")
        st.caption("Select up to 5 movies to compare their rating volume and average score over the years.")

        all_titles = sorted(movies["movie_title"].dropna().unique().tolist())
        selected_titles = st.multiselect(
            "Choose movies (up to 5)",
            all_titles,
            default=all_titles[:3],
            max_selections=5,
            key="trend_movies",
        )
        metric = st.radio("Show", ["Rating Count (volume)", "Average Rating (score)"], horizontal=True, key="trend_metric")

        if not selected_titles:
            st.info("Select at least one movie above.")
        else:
            fig, ax = plt.subplots(figsize=(11, 4))
            colors = ["#e50914", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]

            for idx, title in enumerate(selected_titles):
                mid_row = movies[movies["movie_title"] == title]
                if len(mid_row) == 0:
                    continue
                mid = mid_row.iloc[0]["movie_id"]
                movie_r = ratings_m[ratings_m["movie_id"] == mid]
                if len(movie_r) == 0:
                    continue
                yearly = movie_r.groupby("year")["rating"].agg(
                    count="count", mean="mean"
                ).reset_index()
                yearly = yearly[yearly["year"].between(1990, 2020)]
                if metric.startswith("Rating Count"):
                    ax.plot(yearly["year"], yearly["count"],  marker="o", label=title[:35], color=colors[idx % len(colors)])
                else:
                    ax.plot(yearly["year"], yearly["mean"],   marker="o", label=title[:35], color=colors[idx % len(colors)])

            ax.set_xlabel("Year")
            ax.set_ylabel("Rating Count" if metric.startswith("Rating Count") else "Avg Rating")
            ax.set_title("Movie Popularity Over Time")
            if metric.startswith("Average"):
                ax.set_ylim(0, 5.2)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Summary table
            rows = []
            for title in selected_titles:
                mid_row = movies[movies["movie_title"] == title]
                if len(mid_row) == 0:
                    continue
                mid = mid_row.iloc[0]["movie_id"]
                mr = ratings_m[ratings_m["movie_id"] == mid]
                if len(mr) == 0:
                    continue
                peak_year = mr.groupby("year")["rating"].count().idxmax() if len(mr) > 0 else "N/A"
                rows.append({
                    "Movie": title,
                    "Total Ratings": f"{len(mr):,}",
                    "Avg Rating": f"{mr['rating'].mean():.2f}",
                    "Peak Activity Year": peak_year,
                    "First Rated": int(mr["year"].min()),
                    "Last Rated":  int(mr["year"].max()),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── TAB 2: Genre trends ──────────────────────────────────────
    with tab2:
        st.markdown("### Genre Popularity by Year")
        st.caption("Tracks how many ratings each genre received per year — a proxy for audience interest.")

        genre_list = sorted(set(
            g.strip() for g in "|".join(movies["genres"].dropna()).split("|") if g.strip()
        ))
        selected_genres = st.multiselect(
            "Choose genres (up to 6)",
            genre_list,
            default=["Action", "Drama", "Comedy", "Thriller"] if all(g in genre_list for g in ["Action","Drama","Comedy","Thriller"]) else genre_list[:4],
            max_selections=6,
            key="trend_genres",
        )
        genre_metric = st.radio("Show", ["Volume (rating count)", "Average Score"], horizontal=True, key="genre_metric")

        if not selected_genres:
            st.info("Select at least one genre.")
        else:
            # Explode genres
            ratings_exp = ratings_m.copy()
            ratings_exp["genre_list"] = ratings_exp["genres"].apply(
                lambda g: [x.strip() for x in str(g).split("|")]
            )
            ratings_exp = ratings_exp.explode("genre_list")
            ratings_exp = ratings_exp[ratings_exp["genre_list"].isin(selected_genres)]
            ratings_exp = ratings_exp[ratings_exp["year"].between(1993, 2019)]

            colors = ["#e50914", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
            fig, ax = plt.subplots(figsize=(11, 4))

            for idx, genre in enumerate(selected_genres):
                gdf = ratings_exp[ratings_exp["genre_list"] == genre].groupby("year")["rating"].agg(
                    count="count", mean="mean"
                ).reset_index()
                if genre_metric.startswith("Volume"):
                    ax.plot(gdf["year"], gdf["count"], marker="o", label=genre, color=colors[idx % len(colors)])
                else:
                    ax.plot(gdf["year"], gdf["mean"],  marker="o", label=genre, color=colors[idx % len(colors)])

            ax.set_xlabel("Year")
            ax.set_ylabel("Rating Count" if genre_metric.startswith("Volume") else "Avg Rating")
            ax.set_title("Genre Trends Over Time")
            if genre_metric.startswith("Average"):
                ax.set_ylim(0, 5.2)
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Fastest growing genre (last 5 years vs prior 5)
            try:
                max_yr = int(ratings_exp["year"].max())
                recent = ratings_exp[ratings_exp["year"] >= max_yr - 4]
                prior  = ratings_exp[ratings_exp["year"].between(max_yr - 9, max_yr - 5)]
                rec_vol  = recent.groupby("genre_list")["rating"].count()
                pri_vol  = prior.groupby("genre_list")["rating"].count()
                growth   = ((rec_vol - pri_vol) / pri_vol.replace(0, np.nan) * 100).dropna().sort_values(ascending=False)
                growth_df = growth.reset_index()
                growth_df.columns = ["Genre", "Growth (%)"]
                growth_df["Growth (%)"] = growth_df["Growth (%)"].round(1)
                with st.expander(f"Genre growth: recent 5 years vs prior 5 years (base year {max_yr})"):
                    st.dataframe(growth_df, use_container_width=True)
            except Exception:
                pass

    # ── TAB 3: Rating heatmap ────────────────────────────────────
    with tab3:
        st.markdown("### Rating Activity Heatmap — Month vs Year")
        st.caption("Shows total ratings submitted each month across years. Hot spots = peak engagement periods.")

        if not has_month:
            st.info("Monthly data not available in this dataset.")
        else:
            heatmap_df = ratings[ratings["year"].between(1995, 2019)].groupby(
                ["year", "month"]
            )["rating"].count().reset_index()
            heatmap_pivot = heatmap_df.pivot(index="month", columns="year", values="rating").fillna(0)

            month_names = {
                1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"
            }
            heatmap_pivot.index = [month_names.get(m, str(m)) for m in heatmap_pivot.index]

            fig, ax = plt.subplots(figsize=(14, 5))
            im = ax.imshow(heatmap_pivot.values, aspect="auto", cmap="YlOrRd")
            ax.set_xticks(range(len(heatmap_pivot.columns)))
            ax.set_xticklabels(heatmap_pivot.columns, rotation=45, fontsize=8)
            ax.set_yticks(range(len(heatmap_pivot.index)))
            ax.set_yticklabels(heatmap_pivot.index)
            ax.set_xlabel("Year")
            ax.set_ylabel("Month")
            ax.set_title("Rating Volume Heatmap (Month × Year)")
            plt.colorbar(im, ax=ax, label="Number of Ratings")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Peak month overall
            monthly_totals = ratings[has_month and "month" in ratings.columns and ratings["month"].notna()].groupby("month")["rating"].count()
            if len(monthly_totals) > 0:
                peak_month = month_names.get(int(monthly_totals.idxmax()), "N/A")
                st.info(f"Highest overall rating activity: **{peak_month}**")

    # ── TAB 4: Breakout tracker ──────────────────────────────────
    with tab4:
        st.markdown("### Breakout Movies — Most Rapid Rating Growth")
        st.caption(
            "Finds movies whose ratings spiked in a particular year — potential cult hits, "
            "award-season discoveries, or streaming breakouts."
        )

        min_total = st.slider("Min total ratings to qualify", 50, 500, 100, 50, key="breakout_min")
        compare_year = st.slider(
            "Spike year to inspect",
            1995, 2018, 2000,
            key="breakout_year"
        )

        with st.spinner("Scanning for breakout movies..."):
            yr_counts = ratings_m.groupby(["movie_id", "year"])["rating"].count().reset_index()
            yr_counts.columns = ["movie_id", "year", "count"]

            total_counts = yr_counts.groupby("movie_id")["count"].sum()
            qualified = total_counts[total_counts >= min_total].index

            yr_counts = yr_counts[yr_counts["movie_id"].isin(qualified)]
            prior = yr_counts[yr_counts["year"] < compare_year].groupby("movie_id")["count"].mean().rename("prior_avg")
            spike = yr_counts[yr_counts["year"] == compare_year].set_index("movie_id")["count"].rename("spike_count")

            breakout = pd.concat([prior, spike], axis=1).dropna()
            breakout["growth_ratio"] = breakout["spike_count"] / breakout["prior_avg"].replace(0, np.nan)
            breakout = breakout.dropna().sort_values("growth_ratio", ascending=False).head(15)
            breakout = breakout.reset_index().merge(
                movies[["movie_id", "movie_title", "genres", "release_year", "imdb_rating"]],
                on="movie_id", how="left"
            )

            if len(breakout) == 0:
                st.info("No breakout movies found for the selected year / threshold.")
            else:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(
                    breakout["movie_title"].str[:35][::-1],
                    breakout["growth_ratio"][::-1],
                    color="#e50914"
                )
                ax.set_xlabel("Growth Ratio (spike year vs prior avg)")
                ax.set_title(f"Top Breakout Movies in {compare_year}")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                display_cols = ["movie_title", "release_year", "genres", "imdb_rating", "prior_avg", "spike_count", "growth_ratio"]
                display_cols = [c for c in display_cols if c in breakout.columns]
                out = breakout[display_cols].copy()
                out.columns = [c.replace("_", " ").title() for c in out.columns]
                if "Prior Avg" in out.columns:
                    out["Prior Avg"] = out["Prior Avg"].round(1)
                if "Growth Ratio" in out.columns:
                    out["Growth Ratio"] = out["Growth Ratio"].round(2)
                st.dataframe(out, use_container_width=True)


def page_compare_users():
    st.title("Compare Users")
    st.markdown(
        """
        > Select two users to compare their taste profiles, rating behaviour, genre preferences,
        > and SVD-predicted recommendations side by side. Shared recommendations are highlighted.
        """
    )

    movies  = load_movies()
    ratings = load_ratings()
    all_users = sorted(ratings["user_id"].dropna().unique().tolist())

    col_a, col_b = st.columns(2)
    with col_a:
        user_a = st.selectbox("User A", all_users, index=0,  key="cmp_user_a")
    with col_b:
        user_b = st.selectbox("User B", all_users, index=min(1, len(all_users)-1), key="cmp_user_b")

    if user_a == user_b:
        st.warning("Please select two different users to compare.")
        return

    n_recs = st.slider("Recommendations to show per user", 5, 20, 10, key="cmp_n")

    ra = ratings[ratings["user_id"] == user_a]
    rb = ratings[ratings["user_id"] == user_b]

    # ── helper: build genre preference series ──────────────────
    def genre_prefs(user_ratings):
        merged = user_ratings.merge(movies[["movie_id", "genres"]], on="movie_id", how="left")
        rows = []
        for _, row in merged.iterrows():
            for g in str(row["genres"]).split("|"):
                g = g.strip()
                if g and g != "Unknown":
                    rows.append({"genre": g, "rating": row["rating"]})
        if not rows:
            return pd.Series(dtype=float)
        gdf = pd.DataFrame(rows)
        return gdf.groupby("genre")["rating"].mean().sort_values(ascending=False)

    prefs_a = genre_prefs(ra)
    prefs_b = genre_prefs(rb)

    # ── helper: decade preference ──────────────────────────────
    def decade_prefs(user_ratings):
        merged = user_ratings.merge(movies[["movie_id", "release_year"]], on="movie_id", how="left")
        merged = merged.dropna(subset=["release_year"])
        merged["decade"] = (merged["release_year"] // 10 * 10).astype(int).astype(str) + "s"
        return merged.groupby("decade")["rating"].mean().sort_values(ascending=False)

    dec_a = decade_prefs(ra)
    dec_b = decade_prefs(rb)

    st.markdown("---")
    # ── KPI row ─────────────────────────────────────────────────
    st.subheader("At a Glance")
    k1a, k2a, k3a, k_mid, k1b, k2b, k3b = st.columns([2, 2, 2, 0.5, 2, 2, 2])
    k1a.metric(f"User {user_a} — Ratings", f"{len(ra):,}")
    k2a.metric("Avg Rating", f"{ra['rating'].mean():.2f}")
    k3a.metric("Top Genre", prefs_a.index[0] if len(prefs_a) > 0 else "N/A")
    k_mid.markdown("<div style='text-align:center;font-size:1.5rem;margin-top:1.2rem'>vs</div>", unsafe_allow_html=True)
    k1b.metric(f"User {user_b} — Ratings", f"{len(rb):,}")
    k2b.metric("Avg Rating", f"{rb['rating'].mean():.2f}")
    k3b.metric("Top Genre", prefs_b.index[0] if len(prefs_b) > 0 else "N/A")

    # ── Overlap stats ────────────────────────────────────────────
    ids_a = set(ra["movie_id"].tolist())
    ids_b = set(rb["movie_id"].tolist())
    shared_ids = ids_a & ids_b
    st.markdown("---")
    st.subheader("Rating Overlap")
    ov1, ov2, ov3, ov4 = st.columns(4)
    ov1.metric("Movies Both Rated", len(shared_ids))
    ov2.metric("Only User A rated", len(ids_a - ids_b))
    ov3.metric("Only User B rated", len(ids_b - ids_a))
    if shared_ids:
        shared_a = ra[ra["movie_id"].isin(shared_ids)].set_index("movie_id")["rating"]
        shared_b = rb[rb["movie_id"].isin(shared_ids)].set_index("movie_id")["rating"]
        common = shared_a.index.intersection(shared_b.index)
        if len(common) > 1:
            from scipy.stats import pearsonr
            try:
                corr, _ = pearsonr(shared_a[common], shared_b[common])
                ov4.metric("Rating Correlation", f"{corr:.3f}")
            except Exception:
                ov4.metric("Rating Correlation", "N/A")

    if shared_ids:
        shared_df = (
            ra[ra["movie_id"].isin(shared_ids)][["movie_id", "rating"]]
            .rename(columns={"rating": f"User {user_a}"})
            .merge(
                rb[rb["movie_id"].isin(shared_ids)][["movie_id", "rating"]]
                .rename(columns={"rating": f"User {user_b}"}),
                on="movie_id"
            )
            .merge(movies[["movie_id", "movie_title", "genres"]], on="movie_id", how="left")
        )
        shared_df["Difference"] = (shared_df[f"User {user_a}"] - shared_df[f"User {user_b}"]).abs()
        st.markdown("**Movies both users rated — sorted by disagreement**")
        show_shared = shared_df.sort_values("Difference", ascending=False)[
            ["movie_title", "genres", f"User {user_a}", f"User {user_b}", "Difference"]
        ].rename(columns={"movie_title": "Title", "genres": "Genres"}).head(15)
        st.dataframe(show_shared.reset_index(drop=True), use_container_width=True)

    # ── Genre preference comparison ──────────────────────────────
    st.markdown("---")
    st.subheader("Genre Preferences")
    col1, col2 = st.columns(2)
    all_genres_union = sorted(set(prefs_a.index) | set(prefs_b.index))
    compare_genres = pd.DataFrame({
        f"User {user_a}": prefs_a.reindex(all_genres_union),
        f"User {user_b}": prefs_b.reindex(all_genres_union),
    }).dropna(how="all").sort_values(f"User {user_a}", ascending=False).head(15)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        top_a = prefs_a.head(10)
        ax.barh(top_a.index[::-1], top_a.values[::-1], color="#e50914")
        ax.set_xlabel("Avg Rating")
        ax.set_title(f"User {user_a} — Top Genres")
        ax.set_xlim(0, 5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        top_b = prefs_b.head(10)
        ax.barh(top_b.index[::-1], top_b.values[::-1], color="#1f77b4")
        ax.set_xlabel("Avg Rating")
        ax.set_title(f"User {user_b} — Top Genres")
        ax.set_xlim(0, 5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Side-by-side genre table
    if len(compare_genres) > 0:
        with st.expander("Full genre comparison table"):
            st.dataframe(compare_genres.round(2), use_container_width=True)

    # ── Decade preferences ────────────────────────────────────────
    if len(dec_a) > 0 or len(dec_b) > 0:
        st.markdown("---")
        st.subheader("Decade Preferences")
        all_decades = sorted(set(dec_a.index) | set(dec_b.index))
        dec_df = pd.DataFrame({
            f"User {user_a}": dec_a.reindex(all_decades),
            f"User {user_b}": dec_b.reindex(all_decades),
        }).dropna(how="all").sort_index()
        fig, ax = plt.subplots(figsize=(9, 3))
        x = np.arange(len(dec_df))
        width = 0.35
        ax.bar(x - width/2, dec_df[f"User {user_a}"].fillna(0), width, label=f"User {user_a}", color="#e50914")
        ax.bar(x + width/2, dec_df[f"User {user_b}"].fillna(0), width, label=f"User {user_b}", color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(dec_df.index, rotation=45)
        ax.set_ylabel("Avg Rating")
        ax.set_ylim(0, 5)
        ax.legend()
        ax.set_title("Average Rating by Decade")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Top rated movies per user ────────────────────────────────
    st.markdown("---")
    st.subheader("Highest-Rated Movies")
    col1, col2 = st.columns(2)
    def top_rated_table(user_ratings, label):
        top = (
            user_ratings.nlargest(10, "rating")
            .merge(movies[["movie_id", "movie_title", "genres", "release_year"]], on="movie_id", how="left")
        )[["movie_title", "release_year", "genres", "rating"]]
        top = top.rename(columns={"movie_title": "Title", "release_year": "Year", "rating": "Rating"})
        if "Year" in top.columns:
            top["Year"] = top["Year"].apply(lambda y: int(y) if pd.notna(y) else "")
        st.markdown(f"**User {label}**")
        st.dataframe(top.reset_index(drop=True), use_container_width=True)

    with col1:
        top_rated_table(ra, user_a)
    with col2:
        top_rated_table(rb, user_b)

    # ── SVD recommendations comparison ───────────────────────────
    st.markdown("---")
    st.subheader("SVD Recommendations — Side by Side")
    st.caption("Movies appearing in both lists are marked as shared recommendations.")

    matrix = build_user_movie_matrix(ratings)
    u_ids_list = list(matrix.index)

    def svd_recs_for_user(uid, n):
        if uid not in u_ids_list:
            return []
        predicted, u_ids, m_ids = build_svd_model(matrix)
        uidx = u_ids.index(uid)
        pred_row = predicted[uidx]
        rated = set(ratings[ratings["user_id"] == uid]["movie_id"].tolist())
        scores = [(m_ids[i], pred_row[i]) for i in range(len(m_ids)) if m_ids[i] not in rated]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    recs_a = svd_recs_for_user(user_a, n_recs)
    recs_b = svd_recs_for_user(user_b, n_recs)

    ids_recs_a = {mid for mid, _ in recs_a}
    ids_recs_b = {mid for mid, _ in recs_b}
    shared_recs = ids_recs_a & ids_recs_b

    def format_rec_row(mid, score, is_shared):
        row = movies[movies["movie_id"] == mid]
        if len(row) == 0:
            return None
        r = row.iloc[0]
        title = r.get("movie_title", "Unknown")
        year  = int(r["release_year"]) if pd.notna(r.get("release_year")) else ""
        tag   = " ★ Shared" if is_shared else ""
        return f"**{title}** ({year}){tag} — score: {score:.3f}"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### User {user_a}")
        for mid, score in recs_a:
            line = format_rec_row(mid, score, mid in shared_recs)
            if line:
                st.markdown(line)
    with col2:
        st.markdown(f"#### User {user_b}")
        for mid, score in recs_b:
            line = format_rec_row(mid, score, mid in shared_recs)
            if line:
                st.markdown(line)

    if shared_recs:
        st.success(f"Both users share **{len(shared_recs)}** recommendation(s) marked with ★ above.")
    else:
        st.info("No overlapping recommendations — these users have very different tastes.")


def page_cold_start():
    st.title("New User / Cold Start")
    st.markdown(
        """
        > **Cold Start Problem:** New users have no rating history, so collaborative filtering cannot work.
        > This page collects initial preferences (genre selection + a few ratings) and uses **content-based filtering**
        > to generate a starter set of recommendations.
        >
        > **Technique:** Genre preference vector + TF-IDF similarity on seeded movies
        """
    )
    st.info(
        "Collaborative filtering (User-Based CF, SVD) requires rating history. "
        "For new users, we bootstrap with genre preferences and initial ratings."
    )

    movies = load_movies()
    genre_list = sorted(set(
        g.strip() for g in "|".join(movies["genres"].dropna()).split("|") if g.strip()
    ))

    st.subheader("Step 1 — Select Your Favorite Genres")
    selected_genres = st.multiselect("Choose genres you enjoy", genre_list, default=["Action", "Drama"] if "Action" in genre_list else genre_list[:2])

    st.subheader("Step 2 — Rate 5–10 Popular Movies")
    popular = movies.merge(
        load_ratings().groupby("movie_id")["rating"].count().reset_index().rename(columns={"rating": "cnt"}),
        on="movie_id"
    ).nlargest(50, "cnt")

    user_seed_ratings = {}
    cols = st.columns(2)
    for i, (_, row) in enumerate(popular.head(10).iterrows()):
        with cols[i % 2]:
            rating_val = st.select_slider(
                f"{row['movie_title']} ({int(row['release_year']) if pd.notna(row.get('release_year')) else 'N/A'})",
                options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                value=3.0,
                key=f"cs_{row['movie_id']}"
            )
            user_seed_ratings[row["movie_id"]] = rating_val

    if st.button("Generate Recommendations"):
        st.subheader("Your Personalized Recommendations")
        # Filter movies by genre preference
        genre_filtered = movies[
            movies["genres"].apply(
                lambda g: any(sg in str(g) for sg in selected_genres)
            )
        ]

        # Content similarity from seed movies rated >= 3.5
        liked_seeds = [mid for mid, r in user_seed_ratings.items() if r >= 3.5]
        mat, indexed_movies = build_content_similarity(movies)
        cb_scores = {}
        for seed_mid in liked_seeds:
            seed_row = indexed_movies[indexed_movies["movie_id"] == seed_mid]
            if len(seed_row) == 0:
                continue
            sidx = seed_row.index[0]
            sims = cosine_similarity(mat[sidx], mat).flatten()
            for j, sv in enumerate(sims):
                cand_mid = int(indexed_movies.iloc[j]["movie_id"])
                if cand_mid in user_seed_ratings:
                    continue
                cb_scores[cand_mid] = cb_scores.get(cand_mid, 0) + sv

        # Apply genre filter
        genre_mid_set = set(genre_filtered["movie_id"].tolist())
        filtered_scores = {mid: s for mid, s in cb_scores.items() if mid in genre_mid_set}
        if not filtered_scores and cb_scores:
            filtered_scores = cb_scores

        top_recs = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)[:12]
        for mid, score in top_recs:
            row = movies[movies["movie_id"] == mid]
            if len(row) == 0:
                continue
            display_movie_card(row.iloc[0], "Relevance", min(score / max(filtered_scores.values()), 1.0))


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Hybrid Movie Recommender",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("🎬 Movie Recommender")
    st.sidebar.markdown("*Hybrid Recommendation System*")
    st.sidebar.markdown("*Data Mining & Warehousing Project*")
    st.sidebar.divider()

    pages = {
        "Dashboard": page_dashboard,
        "Movie Explorer": page_movie_explorer,
        "Content-Based Recommender": page_content_based,
        "User-Based Collaborative Filtering": page_user_based_cf,
        "Item-Based Collaborative Filtering": page_item_based_cf,
        "Matrix Factorization / SVD": page_svd,
        "Hybrid Recommendation": page_hybrid,
        "User Clustering": page_user_clustering,
        "Association Rules": page_association_rules,
        "Evaluation Metrics": page_evaluation,
        "New User / Cold Start": page_cold_start,
        "Compare Users": page_compare_users,
        "Popularity Trends": page_popularity_trends,
        "Genre DNA": page_genre_dna,
        "Rating Bias Analyser": page_rating_bias,
        "Serendipity Recommender": page_serendipity,
        "Movie Timeline": page_movie_timeline,
    }

    selection = st.sidebar.radio("Navigate to", list(pages.keys()))

    st.sidebar.divider()
    st.sidebar.markdown(
        """
        **Project Pipeline**  
        `Data Sources` → `ETL` → `Data Warehouse`  
        → `Data Mining` → `Recommendation` → `Evaluation`

        ---
        **Datasets**  
        MovieLens ratings + IMDb metadata  
        ETL-processed, warehouse-ready
        """
    )

    pages[selection]()


if __name__ == "__main__":
    main()
