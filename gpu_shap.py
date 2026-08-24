"""
Experimental GPU-batched, sampled feature-attribution estimator.

GPUExplainer samples random feature coalitions and estimates marginal feature
contributions by batching model evaluations on the selected device. The output is
approximate and depends on the sampling budget. This module is not a drop-in,
correctness-equivalent replacement for official SHAP explainers, and its runtime is
not O(1): it depends on n_samples, feature count, instance count, and model cost.

Usage:
    from gpu_shap import GPUExplainer

    explainer = GPUExplainer(model_fn, background_data)
    shap_values = explainer.shap_values(X_test)

    explainer.feature_importance()
    explainer.plot()  # text-based plot
"""
import torch
import numpy as np
import time


class GPUExplainer:
    """Sampled, GPU-batched SHAP-like feature-attribution estimator."""

    def __init__(self, model_fn, background, device=None):
        """
        Args:
            model_fn: callable that takes (batch, features) tensor -> (batch,) predictions
            background: numpy array or tensor, reference data for the attribution baseline
            device: torch device (default: cuda if available, otherwise cpu)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store model
        self.model_fn = model_fn

        # Process background
        if isinstance(background, np.ndarray):
            background = torch.tensor(background, dtype=torch.float32, device=self.device)
        elif isinstance(background, torch.Tensor):
            background = background.float().to(self.device)

        self.background_mean = background.mean(dim=0, keepdim=True)  # (1, features)
        self.n_features = background.shape[1]
        self._last_shap = None
        self._last_X = None

    def shap_values(self, X, n_samples=200):
        """Estimate feature-attribution values for X from sampled coalitions.

        Args:
            X: input data — numpy array, torch tensor, or pandas DataFrame
            n_samples: number of random coalitions (larger budgets reduce sampling noise)

        Returns:
            numpy array of shape (n_instances, n_features)
        """
        # Convert input
        if isinstance(X, torch.Tensor):
            X = X.float().to(self.device)
        elif isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        elif hasattr(X, 'values'):  # pandas DataFrame
            X = torch.tensor(X.values, dtype=torch.float32, device=self.device)

        n_instances = X.shape[0]
        bg = self.background_mean
        nf = self.n_features

        shap_vals = torch.zeros(n_instances, nf, device=self.device)
        count = torch.zeros(nf, device=self.device)

        for s in range(n_samples):
            # Random coalition
            mask = (torch.rand(nf, device=self.device) > 0.5).float()
            if mask.sum() < 1:
                mask[0] = 1
            if mask.sum() >= nf:
                mask[0] = 0

            # Prediction with coalition
            X_with = X * mask.unsqueeze(0) + bg * (1 - mask.unsqueeze(0))
            pred_with = self.model_fn(X_with)

            # Marginal contributions for all included features (batched)
            included = torch.where(mask > 0.5)[0]
            n_inc = len(included)
            if n_inc == 0:
                continue

            # Build all "without" versions at once
            masks_wo = mask.unsqueeze(0).repeat(n_inc, 1)
            for j, f in enumerate(included):
                masks_wo[j, f] = 0

            # Batch predict: (n_inc * n_instances, features)
            X_rep = X.unsqueeze(0).expand(n_inc, -1, -1).reshape(-1, nf)
            bg_rep = bg.expand(n_inc * n_instances, -1)
            mask_rep = masks_wo.unsqueeze(1).expand(-1, n_instances, -1).reshape(-1, nf)

            X_wo_batch = X_rep * mask_rep + bg_rep * (1 - mask_rep)
            pred_wo_batch = self.model_fn(X_wo_batch).reshape(n_inc, n_instances)

            # Contributions
            contributions = pred_with.unsqueeze(0).expand(n_inc, -1) - pred_wo_batch

            for j, f in enumerate(included):
                shap_vals[:, f] += contributions[j]
                count[f] += 1

        count = count.clamp(min=1)
        shap_vals /= count.unsqueeze(0)

        self._last_shap = shap_vals
        self._last_X = X
        return shap_vals.cpu().numpy()

    def feature_importance(self, feature_names=None):
        """Print a mean-absolute-attribution feature ranking."""
        if self._last_shap is None:
            print("Run shap_values() first.")
            return

        importance = self._last_shap.abs().mean(dim=0).cpu().numpy()
        ranking = importance.argsort()[::-1]

        print("\nFeature Importance (mean |attribution|):")
        print(f"{'Rank':<6} {'Feature':<20} {'Importance':<12}")
        print("-" * 38)
        max_importance = importance.max()
        for i, f in enumerate(ranking):
            name = feature_names[f] if feature_names else f"Feature {f}"
            bar = "#" * int(importance[f] / max_importance * 20) if max_importance > 0 else ""
            print(f"{i+1:<6} {name:<20} {importance[f]:<12.4f} {bar}")

        return importance

    def plot(self, feature_names=None, top_k=10):
        """Print a text-based attribution summary."""
        if self._last_shap is None:
            print("Run shap_values() first.")
            return

        importance = self._last_shap.abs().mean(dim=0).cpu().numpy()
        ranking = importance.argsort()[::-1][:top_k]

        print(f"\nAttribution Summary (top {top_k} features):")
        max_imp = importance[ranking[0]] if len(ranking) else 0
        for f in ranking:
            name = feature_names[f] if feature_names else f"F{f}"
            bar_len = int(importance[f] / max_imp * 40) if max_imp > 0 else 0
            bar = "+" * bar_len
            print(f"  {name:<12} |{bar:<40}| {importance[f]:.4f}")


# ================================================================
if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    try:
        from py2tensor.sklearn_to_gpu import convert_rf
    except ImportError:
        try:
            from sklearn_to_gpu import convert_rf
        except ImportError as exc:
            raise SystemExit(
                "Demo requires a compatible sklearn_to_gpu/py2tensor converter on PYTHONPATH."
            ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("The sklearn-to-GPU demo requires CUDA. The GPUExplainer class itself can run on CPU.")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)
    print("GPU-batched sampled attribution demo")
    print("=" * 60)

    # Train
    X, y = make_classification(
        n_samples=5000,
        n_features=10,
        n_informative=4,
        n_redundant=2,
        random_state=42,
    )
    feature_names = [f"feat_{i}" for i in range(10)]
    rf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
    rf.fit(X, y)

    # Convert to GPU
    gpu_rf = convert_rf(rf).to(device)

    def predict(X_t):
        return gpu_rf(X_t).float()

    # Explain
    print("\n--- Explaining 500 predictions ---")
    explainer = GPUExplainer(predict, X[:100])

    t0 = time.time()
    shap_vals = explainer.shap_values(X[:500], n_samples=300)
    t = time.time() - t0
    print(f"  500 instances in {t:.1f}s")

    explainer.feature_importance(feature_names)
    explainer.plot(feature_names)

    # Exploratory comparison with an official SHAP implementation when installed.
    try:
        import shap

        print("\n--- Exploratory SHAP comparison ---")
        t0 = time.time()
        exp_off = shap.KernelExplainer(lambda x: rf.predict_proba(x)[:, 1], X[:50])
        sv_off = exp_off.shap_values(X[:500], nsamples=100, silent=True)
        t_off = time.time() - t0

        corr = np.corrcoef(shap_vals.flatten(), np.asarray(sv_off).flatten())[0, 1]
        print(f"  Official: {t_off:.1f}s | sampled GPU path: {t:.1f}s")
        print(f"  Attribution correlation: {corr:.4f}")
        print("  Different algorithms/budgets: do not interpret this timing as a universal speedup.")
    except ImportError:
        print("  (shap not installed)")

    print("\n  Usage:")
    print("    explainer = GPUExplainer(model_fn, background_data)")
    print("    values = explainer.shap_values(X_test)")
    print("    explainer.plot(feature_names)")
