# 🎬 Hybrid Movie Recommendation System

### Data Warehousing + Data Mining + Machine Learning

A comprehensive **Hybrid Movie Recommendation System** built with Python and Streamlit that combines **Data Warehousing, ETL, Data Mining, Machine Learning, Content-Based Filtering, Collaborative Filtering, and Matrix Factorization** to generate personalized movie recommendations.

The system integrates movie metadata with user-rating data and provides multiple recommendation approaches through an interactive Streamlit application.

---

## 📌 Project Overview

Choosing a movie from thousands of available titles can be difficult. This project addresses that problem by building a recommendation platform that analyzes:

- Movie metadata
- Genres
- Directors
- Cast
- IMDb ratings
- IMDb votes
- MovieLens user ratings
- User–movie interactions

Instead of depending on a single recommendation algorithm, the system provides multiple recommendation techniques and combines them into a **Hybrid Recommendation Engine**.

### Overall Pipeline

```text
MovieLens Ratings
        +
IMDb Movie Metadata
        ↓
      ETL
        ↓
Data Warehouse
        ↓
Data Preprocessing
        ↓
Data Mining / ML Models
        ↓
Recommendation Algorithms
        ↓
Hybrid Recommendation Engine
        ↓
Interactive Streamlit Application
```

---

# 🚀 Key Features

## 1. 📊 Data Warehouse Dashboard

The application provides an OLAP-style analytics dashboard for exploring the movie data.

The dashboard includes:

- Total number of movies
- Total users
- Total ratings
- Average rating
- Number of genres
- IMDb metadata coverage
- Rating distribution
- Most-rated movies
- Highest-rated movies
- Movie rating statistics

The dashboard demonstrates how ETL-processed data can be used for dimensional/analytical reporting.

---

## 2. 🎥 Movie Explorer

The Movie Explorer allows users to search and explore movies using their metadata.

Users can explore:

- Movie title
- Release year
- Genres
- IMDb rating
- IMDb votes
- Director
- Cast
- MovieLens rating statistics

The application also provides similar movies using content-based similarity.

---

## 3. 👤 Actor / Director Search

Users can search for an actor or director and explore their filmography.

The system provides:

- Filmography
- Number of films in the dataset
- Average MovieLens rating
- Average IMDb rating
- Active years
- Genre distribution
- Movie list
- Recommendations based on the person's body of work

Recommendations are generated using the content similarity of the person's highly rated films.

---

# 🤖 Recommendation Techniques

The project implements multiple recommendation algorithms.

## 4. 🧠 Content-Based Filtering

Content-based filtering recommends movies that are similar to a selected movie based on its metadata.

### Features Used

- Movie title
- Genres
- Director
- Cast
- Other content features

### Technique

```text
Movie Metadata
      ↓
Content Feature Construction
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Similar Movies
```

The system uses **TF-IDF Vectorization** and **Cosine Similarity** to calculate movie similarity.

### Formula

```text
Cosine Similarity(A,B)
        =
      A · B
     -------
     ||A|| ||B||
```

This approach does not require user rating history.

---

# 👥 5. User-Based Collaborative Filtering

User-Based Collaborative Filtering recommends movies by finding users with similar rating behavior.

### Process

```text
User-Movie Rating Matrix
          ↓
User Similarity
          ↓
Similar Users
          ↓
Analyze Movies Rated by Similar Users
          ↓
Remove Movies Already Rated
          ↓
Generate Recommendations
```

The system uses **Cosine Similarity between users**.

The prediction mechanism also uses:

- Mean-centered ratings
- Similarity weights
- Neighbor filtering
- Minimum neighbor requirements
- Shrinkage for items with limited supporting ratings

---

# 🎞️ 6. Item-Based Collaborative Filtering

Item-Based Collaborative Filtering finds movies that have similar rating patterns.

### Process

```text
User-Movie Rating Matrix
          ↓
Movie Similarity Matrix
          ↓
Find Similar Movies
          ↓
Rank Similarity Scores
          ↓
Generate Recommendations
```

This approach identifies movies that users tend to rate similarly.

---

# 📐 7. Matrix Factorization / SVD

The project also implements **Truncated Singular Value Decomposition (SVD)**.

SVD decomposes the user-movie rating matrix into latent factors representing hidden characteristics of users and movies.

```text
R ≈ U × Σ × Vᵀ
```

Where:

- `R` = User-Movie Rating Matrix
- `U` = User latent factors
- `Σ` = Singular values
- `Vᵀ` = Movie latent factors

The resulting predicted ratings are used to recommend movies that a selected user has not already rated.

---

# 🔥 8. Hybrid Recommendation System

The Hybrid Recommendation System is the main recommendation engine of the project.

It combines three recommendation signals:

1. **SVD**
2. **User-Based Collaborative Filtering**
3. **Content-Based Filtering**

### Hybrid Formula

```text
Hybrid Score =
    w₁ × SVD Score
  + w₂ × Collaborative Score
  + w₃ × Content Score
```

The application allows the user to adjust the recommendation weights.

For example:

```text
SVD Weight           → 0.50
Collaborative Weight → 0.30
Content Weight       → 0.20
```

The weights can be adjusted interactively to change the recommendation behavior.

---

# 🔍 Recommendation Filters

The application provides interactive controls including:

- Number of recommendations
- Genre filtering
- Year range
- User selection
- Movie selection
- Recommendation weights

This allows users to customize the recommendation results.

---

# 📥 Recommendation Export

Recommendation results can be exported as CSV files.

Exported information can include:

- Rank
- Movie title
- Release year
- Genres
- Director
- IMDb rating
- IMDb votes
- Similarity score
- Predicted score
- SVD score

This makes the system useful not only for visualization but also for further analysis.

---

# 🏗️ Project Structure

```text
movierecommendationsystem/
│
├── artifacts/
│   └── Project artifacts
│
├── attached_assets/
│   └── Supporting/static assets
│
├── lib/
│   └── Supporting project libraries
│
├── output/
│   ├── final_movies_dataset.csv
│   ├── final_ratings_dataset.csv
│   └── movie_warehouse.db
│
├── scripts/
│   └── Data processing / project scripts
│
├── PROJECT_REPORT_EXTRACT.txt
│
├── app.py
│
├── requirements.txt
│
├── package.json
├── pnpm-lock.yaml
├── pnpm-workspace.yaml
│
├── tsconfig.base.json
├── tsconfig.json
│
└── replit.md
```

The main application reads the processed movie dataset, ratings dataset, and SQLite warehouse from the `output/` directory.

---

# 🛠️ Technologies Used

## Programming Language

- Python

## Data Processing

- Pandas
- NumPy

## Data Visualization

- Matplotlib

## Machine Learning

- Scikit-learn
- TF-IDF
- Cosine Similarity
- Truncated SVD
- K-Means
- Machine Learning evaluation metrics

## Data Warehousing

- SQLite
- ETL
- OLAP-style analytics
- Structured movie/rating datasets

## Application

- Streamlit

## Additional Libraries

- SciPy
- mlxtend

The repository's `requirements.txt` specifies Streamlit 1.41.0 along with Pandas, NumPy, scikit-learn, SciPy, Matplotlib, and mlxtend dependencies.

---

# 📊 Data Sources

The project combines movie and rating information from multiple sources.

### Movie Metadata

Movie metadata contains information such as:

- Movie ID
- Movie title
- Release year
- Genres
- Director
- Cast
- IMDb rating
- IMDb votes

### User Ratings

The rating dataset contains:

- User ID
- Movie ID
- Rating

These datasets are processed through the ETL pipeline before being used by the recommendation models.

---

# 🔄 ETL Pipeline

The project follows an ETL-based data warehousing workflow.

### Extract

Movie and rating data are collected from the source datasets.

### Transform

Data is:

- Cleaned
- Standardized
- Merged
- Structured
- Prepared for machine learning

### Load

The processed datasets are stored for application and analytical use.

```text
Extract
   ↓
Transform
   ↓
Load
   ↓
Data Warehouse
   ↓
Analytics + Machine Learning
```

---

# 🧮 Data Mining Workflow

```text
Raw Data
   ↓
Data Cleaning
   ↓
Data Integration
   ↓
Feature Engineering
   ↓
Exploratory Data Analysis
   ↓
Similarity / Rating Matrix
   ↓
Recommendation Models
   ↓
Model Combination
   ↓
Hybrid Recommendations
```

---

# 📈 Model Evaluation

The project imports standard evaluation techniques including:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Train/Test splitting

These metrics can be used to evaluate prediction quality for recommendation models.

---

# 💻 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Raj1648/movierecommendationsystem.git
```

## 2. Navigate to the Project

```bash
cd movierecommendationsystem
```

## 3. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Run the Application

```bash
streamlit run app.py
```

The Streamlit application will start locally.

---

# 🖥️ Application Modules

The application is organized around the following major modules:

| Module | Purpose |
|---|---|
| 📊 Dashboard | Data warehouse and movie analytics |
| 🎬 Movie Explorer | Search and explore movies |
| 👤 Person Search | Explore actor/director filmographies |
| 🧠 Content-Based | Metadata-based recommendations |
| 👥 User-Based CF | Similar-user recommendations |
| 🎞️ Item-Based CF | Similar-movie rating recommendations |
| 📐 SVD | Matrix factorization recommendations |
| 🔥 Hybrid | Combined recommendation engine |

These modules are implemented in the main Streamlit application.

---

# 🎯 Project Objectives

The main objectives of this project are:

- Build an end-to-end movie recommendation system.
- Demonstrate ETL and data warehousing concepts.
- Apply data mining techniques to movie data.
- Implement content-based recommendation.
- Implement collaborative filtering.
- Implement matrix factorization.
- Build a hybrid recommendation algorithm.
- Provide interactive data analytics.
- Allow users to customize recommendations.
- Export recommendation results for further analysis.

---

# 🌟 Why a Hybrid Recommendation System?

Individual recommendation algorithms have different strengths.

### Content-Based Filtering

Advantages:

- Works without user history.
- Provides explainable similarity.
- Uses movie metadata.

Limitation:

- Can recommend movies too similar to what the user already likes.

### Collaborative Filtering

Advantages:

- Uses collective user behavior.
- Can discover unexpected movies.
- Captures user preferences.

Limitation:

- Can suffer from sparse rating data and cold-start problems.

### SVD

Advantages:

- Learns hidden user/movie patterns.
- Compresses the rating matrix into latent factors.

Limitation:

- Requires sufficient rating data.

### Hybrid Recommendation

Combining multiple signals allows the system to use:

```text
Movie Content
      +
User Behavior
      +
Latent Preferences
      ↓
Hybrid Recommendation
```

---

# 📚 Academic Concepts Demonstrated

This project demonstrates concepts from:

### Data Warehousing

- ETL
- Data integration
- Data warehouse
- OLAP-style analytics
- Aggregation

### Data Mining

- Similarity analysis
- Pattern discovery
- Collaborative filtering
- Clustering-related techniques

### Machine Learning

- Feature engineering
- TF-IDF
- Cosine similarity
- Matrix factorization
- SVD
- Model evaluation

### Data Analytics

- KPI dashboards
- Rating distributions
- Movie statistics
- Genre analysis
- User behavior analysis

---

# 🔮 Future Enhancements

Potential improvements include:

- [ ] Deep Learning recommendation models
- [ ] Neural Collaborative Filtering
- [ ] Transformer-based movie embeddings
- [ ] LLM-powered movie recommendations
- [ ] RAG-based movie assistant
- [ ] Real-time recommendation updates
- [ ] User authentication
- [ ] Personal watch history
- [ ] Like/dislike feedback
- [ ] Movie search API integration
- [ ] TMDB API integration
- [ ] Movie trailers
- [ ] Personalized user profiles
- [ ] Cloud deployment
- [ ] Recommendation explanation using LLMs
- [ ] Advanced recommendation evaluation
- [ ] Cold-start recommendation strategy

---

# 🚀 Future AI/GenAI Architecture

A future version of this project can extend the current recommendation engine with Generative AI:

```text
User
 ↓
Natural Language Query
 ↓
LLM
 ↓
User Preference Extraction
 ↓
Hybrid Recommendation Engine
 ↓
Vector Database
 ↓
Movie Metadata + Ratings
 ↓
Personalized Recommendations
 ↓
LLM Explanation
```

Example:

> "I want a science-fiction movie like Interstellar, but something less than 2 hours and with a high IMDb rating."

The system could understand the natural-language request, convert it into filters and semantic preferences, retrieve suitable movies, and explain why each movie was recommended.

---

# 👨‍💻 Author

**Raj1648**

GitHub:  
https://github.com/Raj1648

---

# 📄 Project Type

**College Project — Data Mining and Data Warehousing**

### Project Domain

```text
Machine Learning
Data Mining
Data Warehousing
Recommendation Systems
Data Analytics
```

---

# ⭐ Acknowledgements

This project uses concepts and techniques from:

- MovieLens
- IMDb movie metadata
- Python Data Science ecosystem
- Scikit-learn
- Streamlit
- SQLite

---

# 📜 License

This project is intended for educational and academic purposes.

---

## 🎬 Final Summary

The **Hybrid Movie Recommendation System** combines traditional data analytics with machine learning recommendation techniques.

```text
                 ┌─────────────────────┐
                 │    Movie Metadata    │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │   MovieLens Ratings │
                 └──────────┬──────────┘
                            │
                       ┌────▼────┐
                       │   ETL   │
                       └────┬────┘
                            │
                  ┌─────────▼─────────┐
                  │  Data Warehouse   │
                  └─────────┬─────────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
       Content-Based    User-Based       SVD
          TF-IDF            CF        Matrix Factorization
             │              │              │
             └──────────────┼──────────────┘
                            │
                     ┌──────▼──────┐
                     │   HYBRID    │
                     │ RECOMMENDER │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │  Streamlit  │
                     │     App     │
                     └─────────────┘
```

**A complete end-to-end recommendation system combining Data Warehousing, Data Mining, Machine Learning, and an interactive Streamlit application.**
