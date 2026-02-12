# Hybrid QAOA–Guided Travelling Salesman Problem Solver

This repository contains the reproducible implementation for the hybrid quantum–classical framework proposed in:

"Hybrid QAOA–Classical Framework for the Travelling Salesman Problem Using Real-World Geographic Data"

## Overview

This work presents a hybrid optimization architecture that integrates:

- Quantum Approximate Optimization Algorithm (QAOA)
- Greedy route construction
- Classical 2-opt local search refinement

The framework evaluates quantum-guided decision layers during sequential TSP tour construction and compares performance against classical baselines.

The implementation includes:

- Hybrid QAOA-guided TSP solver
- Statistical evaluation across 20 random seeds
- Ablation study (fallback vs. no fallback)
- Probability concentration visualization across decision stages
- Performance comparison against greedy and pure 2-opt baselines

---

## Repository Structure

hybrid_qaoa_tsp.py # Main reproducible implementation
requirements.txt # Python dependencies
README.md # Documentation


---

## Features Implemented

1. Hybrid QAOA-guided tour construction
2. Classical fallback mechanism
3. Pure greedy baseline
4. Pure 2-opt refinement baseline
5. Multi-seed statistical evaluation
6. Ablation study
7. Probability distribution visualization across decision stages

---

## Reproducibility

All tables and figures reported in the manuscript can be reproduced by running:


The script will:

- Run 20-seed evaluation
- Generate performance summary table
- Produce probability concentration plots
- Save output figures locally

---

## Dependencies

Python 3.10+ recommended.

Install required packages using:


---

## Experimental Configuration

- QAOA depth: p = 4
- Backend: PennyLane Lightning Simulator
- Classical refinement: 2-opt
- Evaluation metric: Tour length ratio (method / optimal or best-known)
- Statistical evaluation: 20 independent seeds

---

## Notes

This repository provides a clean and consolidated implementation of the final experimental framework. Development prototypes and intermediate experiments are not included to ensure clarity and reproducibility.

---

## License

MIT License
