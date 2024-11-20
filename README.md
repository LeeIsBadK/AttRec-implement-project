# AttRec: A Self-Attention Recommender System

This repository implements a self-attention-based recommender system, **AttRec**, for next-item recommendation. The project includes model implementation, evaluation, and tuning scripts, alongside Jupyter notebooks for detailed experimentation.

Link: [https://arxiv.org/pdf/1808.06414](https://arxiv.org/pdf/1808.06414)

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Project Structure](#project-structure)  
3. [Setup and Installation](#setup-and-installation)
4. [Experimentation](#Experimentation) 
5. [References](#references)  
6. [Acknowledgments](#acknowledgments)  

---

## Project Overview
**AttRec** uses a self-attention mechanism to model sequential user behavior for accurate next-item predictions. It is designed for flexibility, scalability, and ease of experimentation.

Key features:
- **End-to-end pipeline**: Data preprocessing, model training, and evaluation scripts.
- **Pretrained models**: Included checkpoint files for quick testing.
- **Jupyter notebooks**: For visual exploration and detailed analyses.

---

## Project Structure
```
.
├── attrec.ipynb              # Main notebook demonstrating AttRec implementation
├── ncf_deep_dive.ipynb       # Notebook exploring Neural Collaborative Filtering
├── processed_data/           # Preprocessed training and test datasets
│   ├── info.pkl
│   ├── test.csv
│   └── train.csv
├── pyproject.toml            # Python project configuration
├── recommenders/             # Core implementation directory (From https://github.com/recommenders-team/recommenders)
│   ├── README.md             # Detailed documentation for this module
│   ├── datasets/             # Data loading and processing scripts
│   ├── evaluation/           # Evaluation metrics and scripts
│   ├── models/               # Model architecture definitions
│   ├── tuning/               # Hyperparameter tuning scripts
│   └── utils/                # Utility functions
├── requirements.txt          # Dependencies for the project
├── save_path/                # Directory for saving model checkpoints
└── setup.py                  # Setup script for the project
```

---

## Setup and Installation
### Prerequisites
- Python 3.8 or higher
- GPU-enabled machine (recommended for training)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/attrec.git
   cd attrec
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the project package:
   ```bash
   pip install -e .
   ```

---

## Experimentation
Use the included notebooks for interactive exploration:
- **`attrec.ipynb`**: Full implementation walkthrough for AttRec.
- **`ncf_deep_dive.ipynb`**: Insights into Neural Collaborative Filtering. (From https://github.com/recommenders-team/recommenders)

---

## References
- Shuai Zhang et al., *"Next Item Recommendation with Self-Attention."*
- S. Ge, "AttRec: A Recommender System with Self-Attention Mechanism," [slientGe/AttRec](https://github.com/slientGe/AttRec)

---

## Acknowledgments
- This project is inspired by and adapted from the **AttRec** implementation by S. Ge.  
- Special thanks to the authors of *Next Item Recommendation with Self-Attention* for the foundational research.
- Special thanks to the recommenders-team for the recommender template [recommenders-team/recommenders](https://github.com/recommenders-team/recommenders)
- This project is part of the **2301491 SPECIAL TOPICS IN COMPUTER SCIENCE (Popular Techniques in Recommender Systems)** course at Chulalongkorn University in 2024.
--- 
