# Fraud-GNN-Normalisation

This repository contains the code and experimental data for evaluating the interaction between graph density, node representation collapse, normalisation strategies, and initialisation schemes in Graph Neural Networks (GNNs) for financial and review fraud detection.

## 📄 Paper

*Placeholder: A link to the arXiv preprint or the final published journal version will be added here once available.* 

For a quick overview of the methodology and results, please see [PAPER_SUMMARY.md](PAPER_SUMMARY.md).

## 🚀 Experiments & Codebase

The `experiments/` directory contains all Python scripts required to run the grid search, evaluate models, and generate the figures and tables used in the paper.

### Prerequisites

You will need a Python environment with PyTorch and PyTorch Geometric installed.

### Reproducing Results

1. **Running the Models:** Use `run_grid.py` to execute the experiment grid across the datasets. The script supports filtering by model architecture, initialisation scheme, and normalisation method.
   ```bash
   python experiments/run_grid.py --dataset elliptic --out experiments-output/
   python experiments/run_grid.py --dataset yelp --out experiments-output/
   python experiments/run_grid.py --dataset amazon --out experiments-output/
   ```
2. **Additional Analysis:** You can run specific sweep scripts for deeper analysis:
   - `run_density_sweep.py`
   - `run_depth_sweep.py`
   - `run_edge_thinning.py`
   - `run_leakage_check.py`

3. **Generating Tables and Figures:**
   Once the models have been run (or using the provided CSV and NPZ files in the `experiments-output/` folder), you can automatically generate the paper's graphics and statistical tables:
   ```bash
   python experiments/make_figures.py
   python experiments/make_anova_table.py
   python experiments/analyze_results.py
   ```

## 📊 Pre-computed Outputs

To facilitate rapid reproducibility and allow for immediate generation of figures without requiring hours of GPU compute time, we have provided the aggregated output logs (`.csv`) and model prediction arrays (`.npz`) in the `experiments-output/` directory. 

## 📝 License

This project is licensed under the [MIT License](LICENSE).
