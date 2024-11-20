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
### Getting Started

For environment management, we recommend using [conda](https://docs.conda.io/projects/conda/en/latest/glossary.html?highlight=environment#conda-environment), and for development, [VS Code](https://code.visualstudio.com/) is suggested. Follow these steps to install the recommenders package and run a sample notebook on Linux/WSL:

1. **Install GCC**  
   If GCC is not already installed, you can install it on Ubuntu with:  
   ```bash
   sudo apt install gcc
   ```

2. **Set Up a Conda Environment**  
   Create and activate a new Conda environment:  
   ```bash
   conda create -n <environment_name> python=3.9
   conda activate <environment_name>
   ```

3. **Install the Recommenders Package**  
   Install the core `recommenders` package to run all CPU-compatible notebooks:  
   ```bash
   pip install recommenders
   ```

4. **Create a Jupyter Kernel**  
   Set up a Jupyter kernel for the environment:  
   ```bash
   python -m ipykernel install --user --name <environment_name> --display-name <kernel_name>
   ```

5. **Clone the Repository**  
   Clone the repository using VS Code or the command line:  
   ```bash
   git clone https://github.com/recommenders-team/recommenders.git
   ```

6. **Run an Example Notebook in VS Code**  
   - Open a notebook, such as `examples/00_quick_start/sar_movielens.ipynb`.  
   - Select the Jupyter kernel `<kernel_name>`.  
   - Run the notebook.  

This setup ensures you have everything needed to start working with the `recommenders` package efficiently.

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
