# Song2Vec: Music Recommendation with Word2Vec Embeddings

A music recommendation system that applies Word2Vec — the NLP embedding technique — to playlist data, treating songs as "words" and playlists as "sentences." The model learns latent semantic relationships between tracks and uses them to generate personalized recommendations.

---

## Overview

The core insight behind Song2Vec is that the distributional hypothesis from NLP transfers to music: *songs that appear in similar playlist contexts have similar meanings (vibes, genres, moods)*. By training a Word2Vec model on playlist sequences, each song gets a dense vector representation that encodes its musical context.

This project implements:
- Full data pipeline (loading, cleaning, tokenization, train/test split)
- Baseline and hyperparameter-tuned Song2Vec (Word2Vec) models
- Systematic one-factor-at-a-time hyperparameter sweep
- Recommendation engine with fallback and diversity-aware (MMR) strategies
- Evaluation with HR@K and NDCG@10 metrics
- Visualizations: t-SNE embeddings, artist similarity heatmaps, hyperparameter ranking charts

---

## Results Snapshot

| Model Variant | HR@10 |
|---|---|
| Baseline (default Word2Vec) | 3.56% |
| Best single param: `min_count=10` | 6.53% |
| `epochs=20` | 5.41% |
| `window=40` | 5.30% |
| CBOW (`sg=0`) | 0.77% |

**Key findings:**
- Skip-gram dramatically outperforms CBOW for music recommendation
- Larger windows and more epochs consistently improve hit rate
- Higher `min_count` (filtering rare tracks) substantially boosts performance
- NS exponent of 0.75 (standard NLP default) works better than music-specific negative values

---

## Dataset
Official Source: https://remaplab.deib.polimi.it/resources/

Playlist data in IDOAPP format (~47K playlists after cleaning):

| Split | Playlists | Track Events |
|---|---|---|
| Train (80%) | 37,592 | 1,266,295 |
| Test (20%) | 9,399 | 329,195 |

- **Vocabulary:** 53,727 unique tracks
- Tracks tokenized as `artist__song_title`
- Playlists with fewer than 2 songs filtered out
- Random seed `455` for reproducibility

---

## Methodology

### Song2Vec Training

Songs are treated as tokens; playlists are treated as documents. A Word2Vec Skip-gram model is trained to predict surrounding songs given a target song, learning a 100-dimensional (configurable) embedding space.

```
playlist: [song_A, song_B, song_C, song_D, ...]
                      ↑
              predict neighbors within window
```

### Hyperparameter Sweep

One-factor-at-a-time sweep over:

| Parameter | Values Tested |
|---|---|
| `sg` (architecture) | 0 (CBOW), 1 (Skip-gram) |
| `window` | 5, 20, 40 |
| `negative` (neg. samples) | 5, 10, 15 |
| `ns_exponent` | -1.0, -0.5, 0.0, 0.5, 0.75, 1.0 |
| `min_count` | 3, 5, 10 |
| `vector_size` | 50, 100, 300 |
| `epochs` | 5, 20, 50 |
| `K_playlist` (context window) | 25, 50, 100, 150 |

### Recommendation Strategies

1. **Mean Embedding Lookup** — Average embeddings of seed songs, retrieve nearest neighbors by cosine similarity
2. **Artist-Level Fallback** — When seed songs are out-of-vocabulary (OOV), fall back to artist-level embeddings
3. **Maximal Marginal Relevance (MMR)** — Balances relevance vs. diversity (λ=0.5) to avoid redundant recommendations

### Evaluation

- **HR@K** (Hit Rate at K=5, 10, 20, 50) — Does the true next song appear in the top-K recommendations?
- **NDCG@10** — Normalized Discounted Cumulative Gain, measuring ranking quality
- Two evaluation modes: A1 (full playlist context), A2 (query-only)

---

## Notebook Structure

```
Part A — Data Preparation & Exploration
  ├── Load & parse playlist ZIP archive
  ├── Tokenize track identifiers
  ├── Train/test split (playlist-level)
  └── Corpus statistics

Part B — Song2Vec Model Training
  ├── Baseline model
  ├── Hyperparameter sweep (one-factor-at-a-time)
  └── Best combined model

Part C — Evaluation
  ├── HR@K computation (A1 & A2)
  └── NDCG@10

Part D — Recommender System
  ├── Mean-embedding recommender
  ├── Artist fallback strategy
  └── MMR diversity-aware recommender

Part E — Analysis & Visualization
  ├── t-SNE embedding plots (baseline vs. best)
  ├── Artist similarity heatmap
  ├── Hyperparameter ranking charts
  └── Qualitative examples (song algebra, similar-song lookups)
```

---

## Qualitative Examples

The model learns musically meaningful relationships. Example similar-song lookups from the trained embedding space:

- **"Bohemian Rhapsody" (Queen)** → other classic rock anthems
- **"Creep" (Radiohead)** → alternative 90s tracks
- **Song algebra**: `Creep − Karma Police + Oasis - Wonderwall ≈ ?` (analogical reasoning in embedding space)

---

## References

- Mikolov et al. (2013) — *Efficient Estimation of Word Representations in Vector Space*
- Caselles-Dupré et al. (2018) — *Word2Vec Applied to Recommendation: Hyperparameters Matter*
- Gensim Word2Vec documentation

---

## Author

**Ali Kumral**
