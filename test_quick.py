"""Quick smoke test for the sampled GPU SHAP-like estimator."""

# Test 1: Import
try:
    from gpu_shap import GPUExplainer
    print("[OK] GPUExplainer import")
except Exception as e:
    print(f"[FAIL] import: {e}")

# Test 2: Basic usage with a simple model. This path can run on CPU when CUDA is absent.
try:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def model_fn(X):
        """X: [batch, features] -> [batch]."""
        weights = torch.tensor([1.0, 2.0, -1.0, 0.5], device=device)
        return (X * weights).sum(dim=1)

    bg = torch.randn(50, 4, device=device)
    explainer = GPUExplainer(model_fn, bg, device=device)
    print(f"[OK] GPUExplainer created on {device}")

    test_data = torch.randn(10, 4, device=device)
    values = explainer.shap_values(test_data, n_samples=32)
    print(f"[OK] attribution values computed: shape={values.shape}")

    imp = explainer.feature_importance()
    print(f"[OK] feature importance: {imp}")

except Exception as e:
    print(f"[FAIL] usage: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Optional sklearn-to-GPU converter discovery. No machine-local path injection.
try:
    from sklearn.ensemble import RandomForestClassifier  # noqa: F401
    from sklearn.datasets import make_classification  # noqa: F401
    try:
        from py2tensor.sklearn_to_gpu import convert_rf  # noqa: F401
    except ImportError:
        from sklearn_to_gpu import convert_rf  # noqa: F401
    print("[OK] optional sklearn-to-GPU converter available")
except Exception as e:
    print(f"[SKIP] optional sklearn demo dependency unavailable: {e}")
