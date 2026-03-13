# Interpreting Double Descent
This project studies the double descent phenomenon in neural networks by varying model depth and tracking train loss, test loss, and effective dimensionality.

Using the codebase structure from Maddox et al. (2020), we adapted their setup from a two-spirals setting to the Moons dataset and observed clear double descent behavior. In our experiments, this appears on a comparatively more non-linear dataset (Moons) than the spirals configuration used in their original demonstration.

## How to Run

1. Create and activate a Python environment (Python 3.10+ recommended).
2. Install core dependencies:

```bash
pip install -r requirements.txt
pip install -e ./hessian-eff-dim
```

3. Launch and run the main notebook:

```bash
jupyter notebook moons_depth.ipynb
```

This notebook is the main entrypoint for reproducing the Moons double-descent experiments.

### Optional: CUDA Worker (Parallelized Training)

If you have a strong CUDA-capable GPU and want faster parallel runs, you can use the helper in `cuda_worker.py` (function `run_single_experiment`) from the notebook parallel-execution cells.

Note: `cuda_worker.py` currently contains a hardcoded path for one environment (`/root/double-descent/hessian-eff-dim`). Update that path to your local workspace path if needed.

## Acknowledgment and Citation
This project uses and adapts code from [1], and applies it to sklearn's Moons dataset [4].

## References:
1. Maddox, W. J., Benton, G., & Wilson, A. G. (2020). Rethinking parameter counting in deep models: Effective dimensionality revisited. arXiv preprint arXiv:2003.02139.
2. Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). Reconciling modern machine-learning practice and the classical bias–variance trade-off. Proceedings of the National Academy of Sciences, 116(32), 15849-15854.
3. Nakkiran, P., Kaplun, G., Bansal, Y., Yang, T., Barak, B., & Sutskever, I. (2021). Deep double descent: Where bigger models and more data hurt. Journal of Statistical Mechanics: Theory and Experiment, 2021(12), 124003.
4. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
