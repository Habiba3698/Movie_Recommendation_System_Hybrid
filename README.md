# Movie_Recommendation_System_Hybrid
# Short description
A simple project that uses Content-based and Collaborative Filtering to find which movies a user likes and recommend new movies to them based on both movie attributes likes tags and genres (Cosine Similarity) and rating patterns  (SVD)

# Movie Recommendation System

A small movie recommendation project combining data exploration, content-based features and collaborative filtering experiments.

# **Note:** The original dataset was very large especially when uncompressed so in order to have the original data placed in the data/raw folder you must download from this link: [The original Dataset](https://grouplens.org/datasets/movielens/32m/), then uncompress it and put in the data/raw folder. This is how I ran it on my PC.


## Repo layout
- Movie_Recs_App.py
- requirements.txt
- data/
  - data/clean/
  - data/raw/
  - data/raw_parquet/
- Models/
- Notebook/
  - Notebook/Movie_Rec_Analysis.ipynb
  - Notebook/Movie_Rec_ML.ipynb

## Quick start
1. Create and activate a virtual environment.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Open notebooks to explore analysis and models:
   - Notebook/Movie_Rec_Analysis.ipynb
   - Notebook/Movie_Rec_ML.ipynb

## Run the app
```sh
python Movie_Recs_App.py
```

## Data
- Put raw CSVs into data/raw/ or parquet in data/raw_parquet/.
- Cleaned artifacts are written to data/clean/.

## Models & outputs
- Trained models, pickles and artifacts stored in Models/.

## Notes
- See requirements.txt for exact package versions.
- Use the notebooks for reproducible analysis and to retrain models.
