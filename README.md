# GPU SHAP

> **Research status:** experimental GPU implementation of a sampled, SHAP-like feature-attribution estimator. It is approximate and is not a drop-in correctness-equivalent replacement for official SHAP implementations.
**GPU-batched Monte Carlo-style feature attribution experiments.**

`GPUExplainer.shap_values` samples random feature coalitions and batches model evaluations
on the selected device. Runtime therefore depends on `n_samples`, feature count, instance
count, and model inference cost; it is not O(1).

```python
from gpu_shap import GPUExplainer

explainer = GPUExplainer(model_fn, background_data)
shap_values = explainer.shap_values(X_test, n_samples=300)

explainer.feature_importance(feature_names)
explainer.plot(feature_names)
```

## Method and validation

The implementation estimates marginal contributions from sampled coalitions. The current
repository includes a CUDA smoke test and an optional comparison path against
`shap.KernelExplainer`, but it does not contain a committed benchmark/result artifact that
supports the historical scaling table or a universal speedup claim.

Important limitations:

- attribution values are approximate and depend on the sampling budget;
- runtime scales with the sampling budget, feature count, instance count, and model cost;
- the estimator is not mathematically equivalent to every SHAP explainer;
- any accuracy or speed comparison should record the dataset, model, background set,
  sample budget, hardware, software versions, and raw outputs.
## Requirements

- PyTorch 2.0+ with CUDA
- sklearn (for model training)
- py2tensor (for sklearn → GPU conversion)

## License

MIT License. See [`LICENSE`](LICENSE).