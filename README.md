# GPU SHAP

> **Research status:** experimental GPU-batched implementation of a sampled, SHAP-like feature-attribution estimator. It is approximate and is not a drop-in, correctness-equivalent replacement for official SHAP implementations.

`GPUExplainer.shap_values` samples random feature coalitions and batches model evaluations on the selected device. Runtime depends on `n_samples`, feature count, instance count, and model inference cost; it is **not O(1)**.

```python
from gpu_shap import GPUExplainer

explainer = GPUExplainer(model_fn, background_data)
values = explainer.shap_values(X_test, n_samples=300)

explainer.feature_importance(feature_names)
explainer.plot(feature_names)
```

## Method and validation

The implementation estimates marginal feature contributions from sampled coalitions. The repository contains a lightweight smoke test and an optional exploratory comparison path against `shap.KernelExplainer`, but it does not contain a committed benchmark/result artifact that supports a universal accuracy or speedup claim.

Important limitations:

- attribution values are approximate and depend on the sampling budget;
- runtime scales with the sampling budget, feature count, instance count, and model cost;
- the estimator is not mathematically equivalent to every SHAP explainer;
- a comparison against another explainer is meaningful only when dataset, model, background set, sample budgets, hardware/software versions, and raw outputs are recorded;
- the optional sklearn-to-GPU demo requires a compatible converter that is not bundled into this repository.

## Continuous integration

The GitHub Actions workflow performs CPU-safe source compilation checks on Python 3.10, 3.11, and 3.12. It intentionally does **not** claim to validate CUDA performance or the scientific accuracy of the attribution estimator.

`test_quick.py` can exercise the core estimator on CPU when CUDA is unavailable. GPU-specific converter/demo behavior should still be validated on appropriate NVIDIA hardware.

## Requirements

Core estimator:

- Python 3.10+
- PyTorch
- NumPy

Optional GPU/demo paths:

- CUDA-capable PyTorch and an NVIDIA GPU for GPU execution;
- scikit-learn for the demo model;
- a compatible `sklearn_to_gpu` / py2tensor converter for the optional sklearn-to-GPU demo.

## License

MIT License. See [`LICENSE`](LICENSE).
