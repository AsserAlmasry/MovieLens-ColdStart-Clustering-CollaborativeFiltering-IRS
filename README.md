# Cold-Start Recommendation & Clustering Analysis on MovieLens 20M

An end-to-end recommender systems study analyzing matrix sparsity, co-rating overlap dynamics, and unsupervised clustering mitigation for items under severe cold-start constraints ($\le 1$ interaction) on the **MovieLens 20M** dataset.

---

## Technical Overview

* **Dataset Scale:** Evaluated across $20,000,263$ ratings from $138,493$ users across $26,744$ movies.
* **Co-Rating Sparsity Dynamics:** Analyzed common-rater thresholds ($threshold \ge 5$) on edge cases, characterizing failure modes of classical memory-based Collaborative Filtering when isolated items exhibit extreme interaction sparsity.
* **Vectorized Pairwise Computations:** Replaced nested iteration routines with high-performance sparse CSR representation (`scipy.sparse`), accelerating user-user and item-item interaction sweeps.
* **Clustering-Based Cold-Start Partitioning:** Implemented K-Means clustering on item interaction representations, measuring distortion (WCSS) and silhouette coefficients across hyperparameter boundaries ($K \in [5, 50]$) to cluster cold-start candidates against dense neighborhood anchors.

---

## Quickstart

```bash
git clone [https://github.com/](https://github.com/)<your-username>/MovieLens-ColdStart-Clustering-CollaborativeFiltering.git
cd MovieLens-ColdStart-Clustering-CollaborativeFiltering
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/MovieLens_ColdStart_Analysis.ipynb
