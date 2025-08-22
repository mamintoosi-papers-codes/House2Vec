# House2Vec

House2Vec is a Python repository for **house representation learning** using **graph-based embedding methods** (DeepWalk and Node2Vec).  
The goal is to enhance **house price prediction** by capturing **spatial relationships** among properties through network representation learning.

The repository implements and extends the work from the paper:  
_"Beyond Coordinates: Integrating Graph Embeddings in House Price Prediction Models"_.

![](images/graph.png)

---

## Abstract

Traditional house price prediction models often rely on tabular features (area, rooms, age, etc.) and use latitude/longitude only as raw spatial features. Such approaches fail to capture the **structural relationships** between neighboring houses.

In this work, we build a **spatial graph of houses**, connecting properties that lie within a predefined geographical threshold. Using this graph, we train **graph embeddings** with both **DeepWalk** and **Node2Vec**. These embeddings encode higher-order spatial proximity into dense vectors, which are then combined with conventional house features.

We evaluate three regression models (**Random Forest, Linear Regression, Gradient Boosting**) across two real-world datasets (California Housing, Mashhad Housing). Results show that:

- Graph embeddings **significantly improve predictive performance** compared to raw features.
- For **California**, Node2Vec embeddings outperform both DeepWalk and raw features.
- For **Mashhad**, DeepWalk embeddings show the strongest improvements.
- Gradient Boosting and Random Forest benefit the most from the enriched feature space, while Linear Regression sees only marginal gains.

This demonstrates the flexibility of graph-based embeddings for **geospatial feature engineering**, offering more robust house price prediction models.

![](images/proposedModel.png)

---

## Results

![](images/results.png)

> Grouped bar charts showing the performance of regression models (R² and RMSE) with Raw features, DeepWalk embeddings, and Node2Vec embeddings.

Our experiments highlight that:

- **Node2Vec** yields the best results in the California dataset.
- **DeepWalk** yields the best results in the Mashhad dataset.
- In both datasets, graph embeddings **outperform raw baseline features** in terms of higher R² and lower prediction errors.
- The benefit is especially clear for **tree-based models** (Random Forest, Gradient Boosting).

---

## Requirements

- Python 3.10
- NetworkX 3.3
- NumPy 1.24+
- Scikit-learn 1.5+
- gensim 4.3.2
- scipy 1.12
- pandas 2.2+
- matplotlib 3.8+

Install requirements via:

```bash
pip install -r requirements.txt
```

---

## Getting Started

### 1. Run experiments from Jupyter Notebook

Open `main.ipynb` and execute the cells to:

- Run experiments for both datasets
- Save Excel result tables
- Generate comparison plots (R² and MSE/Log-MSE)

### 2. Run experiments from command line

Example usage:

```bash
python main_hpp.py --dataset CA --embedding_sizes 2 5 10 20
python main_hpp.py --dataset MHD --embedding_sizes 2 5 10 20
```

This performs grid search over embedding sizes, selects the best size, and evaluates models with Raw, DeepWalk, and Node2Vec embeddings.
Results (Excel + plots) are saved in `results/<dataset_name>/`.

---

## Data

The `data` folder contains two datasets:

1. **California-housing.csv**
   Classic California housing dataset.
2. **MHD-housing.xlsx**
   A dataset of Mashhad housing records.

Both datasets include geographic coordinates for spatial graph construction.

---

## Citation

If you use this repository in your research, please cite:

```
@article{House2Vec2025,
  title={Beyond Coordinates: Integrating Graph Embeddings in House Price Prediction Models},
  author={Amintoosi, Mahmood and Ashkezari-Toussi, Soheila},
  year={2025},
  note={pre-print}
}
```
