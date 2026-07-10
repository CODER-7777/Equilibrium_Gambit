# Equilibrium Gambit

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Domain](https://img.shields.io/badge/Domain-Quantitative_Finance-green.svg)]()

> A mathematical modeling and algorithmic simulation repository focused on probability theory, stochastic processes, game theory, and quantitative finance.

## Overview

**Equilibrium Gambit** is a structured, week-by-week exploration into quantitative modeling, probability, and Python-based algorithmic simulations. The project covers the mechanics of stochastic calculus, Markov chains, Monte Carlo methods, betting theory, and strategic decision-making under uncertainty — with practical implementations grounded in financial and game-theoretic applications.

This repository serves as both a theoretical study framework and a practical implementation sandbox for advanced mathematical modeling.

## Repository Structure

```
Equilibrium_Gambit/
├── Week_2/              Markov Chains and Market Regime Simulation
├── Week_3/              Monte Carlo Methods and Geometric Brownian Motion
├── Week_4/              Dynamic Casino Strategies (Kelly, Martingale, Random)
├── Week_5/              Game Theory — Bayesian Espionage Simulation
├── Resources/           Curated study materials and roadmap
├── notes/               Detailed solution explanations and mathematical notes
├── plots/               Visualizations generated from simulations
└── README.md
```

### Weekly Breakdown

| Week | Topic | Key Concepts |
|------|-------|-------------|
| 2 | Market Regime Simulator | Markov Chains, transition matrices, steady-state distributions, eigenvectors, portfolio simulation |
| 3 | Monte Carlo and GBM | Monte Carlo Pi estimation, Geometric Brownian Motion, stock price path simulation |
| 4 | The Good Casino | Kelly Criterion, Martingale strategy, dynamic bet sizing, card counting analysis |
| 5 | I Spy with my Little Eye | Bayesian updating, mixed strategies, imperfect information, Nash Equilibria |

## Core Concepts

- **Markov Chains and Random Walks** — State space modeling, transition matrices, steady-state and stationary distributions, memoryless processes.
- **Monte Carlo Methods** — Randomized simulations for estimating probabilities, expected returns, and path dependencies in stochastic environments.
- **Quantitative Finance** — Geometric Brownian Motion, market regime simulation, dynamic portfolio valuation, Kelly Criterion for optimal bet sizing.
- **Game Theory** — Bayesian reasoning under imperfect information, mixed-strategy equilibria, strategic deception and signaling.
- **Linear Algebra in Practice** — Power iteration, eigenvector computations for equilibrium states and steady-state distributions.

## Getting Started

### Prerequisites

Python 3.x with the core scientific computing stack:

```bash
pip install numpy pandas matplotlib jupyter
```

### Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Equilibrium_Gambit.git
   cd Equilibrium_Gambit
   ```

2. **Run Jupyter Notebooks:**
   Each week's assignment is a self-contained `.ipynb` notebook. Open them locally or upload to Google Colab:
   ```bash
   jupyter notebook
   ```

3. **Run standalone scripts:**
   Some weeks also include `.py` scripts for direct execution:
   ```bash
   python Week_2/week2_solution.py
   ```

## Study Notes and Resources

- [Solution Explanations and Mathematical Notes](./notes/solutions_explanation.md) — Detailed step-by-step breakdown of each week's approach, formulas, and key findings.
- [Curated Resources and Roadmap](./Resources/resources.md) — Organized reading list covering probability, Markov chains, Monte Carlo, stochastic calculus, betting theory, and game theory.


