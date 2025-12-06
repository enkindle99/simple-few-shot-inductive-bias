# Few-Shot Inductive Bias Comparison

## Project Overview

This project explores how different inductive biases affect learning performance in few-shot or task adaptation settings. We compare three models on synthetic linear regression tasks:

- **MLP (Baseline)**: A standard multilayer perceptron with minimal inductive bias.
- **L2-Regularized MLP**: Adds L2 regularization to encourage smaller weights.
- **Sparse/Modular MLP**: Only a subset of weights are trainable, simulating sparse/modular inductive biases.

The goal is to understand how these biases impact generalization when training data is limited.

## Setup Instructions

### Clone the repository

```bash
git clone https://github.com/enkindle99/few-shot-inductive-bias.git
cd few-shot-inductive-bias
```

### Create and activate a virtual environment (optional but recommended)

- On macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

- On Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Running Experiments

Launch Jupyter Notebook or Jupyter Lab:

```bash
jupyter notebook
# or
jupyter lab
```

Open `notebooks/few_shot_bias_analysis.ipynb` and run cells sequentially to reproduce experiments and visualize results.

## Project Structure

```
few-shot-inductive-bias/
├── notebooks/
│   └── main.ipynb
├── results/
├── README.md
├── requirements.txt
├── .gitignore
└── run_experiment.py
```

## Further Exploration

- Vary L2 regularization strength and observe effects.
- Adjust sparsity level in Sparse MLP.
- Compare to analytical methods like ordinary least squares.
- Test performance with fewer training samples per task.

## License and Contribution

This project is for educational and research purposes. Contributions are welcome. Please cite the repository if used in your work.
