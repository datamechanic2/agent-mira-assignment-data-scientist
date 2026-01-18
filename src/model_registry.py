"""
Simple model versioning and registry system
Tracks model versions, metadata, and performance metrics
"""
import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from io import BytesIO
import joblib


class ModelRegistry:
    """
    Local model registry for versioning and tracking ML models.
    Stores models with metadata, metrics, and allows rollback.
    """

    def __init__(self, registry_path="models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.registry_path / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self):
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {"models": {}, "current_version": None, "production_version": None}

    def _save_manifest(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2, default=str)

    def _generate_version(self, model_name):
        """Generate version string: v{major}.{minor}.{patch}"""
        versions = self.manifest["models"].get(model_name, {}).get("versions", [])
        if not versions:
            return "v1.0.0"

        # get latest and bump patch
        latest = versions[-1]["version"]
        parts = latest[1:].split(".")
        parts[2] = str(int(parts[2]) + 1)
        return "v" + ".".join(parts)

    def _compute_hash(self, model):
        """Create hash of model for deduplication"""
        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        return hashlib.md5(buffer.read()).hexdigest()[:12]

    def register(self, model, model_name, metrics, params=None, tags=None, preprocessor=None):
        """
        Register a new model version.

        Args:
            model: trained model object
            model_name: name for the model (e.g., "house_price_lgbm")
            metrics: dict of performance metrics
            params: dict of hyperparameters
            tags: list of tags for the model
            preprocessor: optional preprocessor to save with model

        Returns:
            version string
        """
        version = self._generate_version(model_name)
        model_hash = self._compute_hash(model)
        timestamp = datetime.now().isoformat()

        # create version directory
        version_dir = self.registry_path / model_name / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # save model artifacts
        joblib.dump(model, version_dir / "model.joblib")
        if preprocessor:
            joblib.dump(preprocessor, version_dir / "preprocessor.joblib")

        # version metadata
        version_info = {
            "version": version,
            "created_at": timestamp,
            "model_hash": model_hash,
            "metrics": metrics,
            "params": params or {},
            "tags": tags or [],
            "stage": "development",
            "model_type": type(model).__name__
        }

        # save version metadata
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(version_info, f, indent=2, default=str)

        # update manifest
        if model_name not in self.manifest["models"]:
            self.manifest["models"][model_name] = {"versions": [], "latest": None}

        self.manifest["models"][model_name]["versions"].append(version_info)
        self.manifest["models"][model_name]["latest"] = version
        self.manifest["current_version"] = f"{model_name}/{version}"
        self._save_manifest()

        print(f"Registered: {model_name}/{version}")
        print(f"  Metrics: {metrics}")
        print(f"  Hash: {model_hash}")

        return version

    def promote(self, model_name, version, stage="production"):
        """Promote a model version to production/staging"""
        versions = self.manifest["models"].get(model_name, {}).get("versions", [])

        for v in versions:
            if v["version"] == version:
                v["stage"] = stage
                if stage == "production":
                    self.manifest["production_version"] = f"{model_name}/{version}"
                break

        self._save_manifest()
        print(f"Promoted {model_name}/{version} to {stage}")

    def load(self, model_name, version=None, stage=None):
        """
        Load a model by name and version or stage.

        Args:
            model_name: name of the model
            version: specific version (e.g., "v1.0.0")
            stage: load by stage ("production", "staging", "development")

        Returns:
            tuple of (model, preprocessor, metadata)
        """
        if stage:
            # find version with matching stage
            versions = self.manifest["models"].get(model_name, {}).get("versions", [])
            for v in reversed(versions):  # latest first
                if v["stage"] == stage:
                    version = v["version"]
                    break
            if not version:
                raise ValueError(f"No {stage} version found for {model_name}")

        if not version:
            version = self.manifest["models"].get(model_name, {}).get("latest")

        if not version:
            raise ValueError(f"Model {model_name} not found")

        version_dir = self.registry_path / model_name / version

        model = joblib.load(version_dir / "model.joblib")
        preprocessor = None
        if (version_dir / "preprocessor.joblib").exists():
            preprocessor = joblib.load(version_dir / "preprocessor.joblib")

        with open(version_dir / "metadata.json") as f:
            metadata = json.load(f)

        print(f"Loaded: {model_name}/{version} ({metadata['stage']})")
        return model, preprocessor, metadata

    def list_versions(self, model_name=None):
        """List all versions, optionally filtered by model name"""
        if model_name:
            versions = self.manifest["models"].get(model_name, {}).get("versions", [])
            return versions

        all_versions = []
        for name, data in self.manifest["models"].items():
            for v in data["versions"]:
                all_versions.append({"model": name, **v})
        return all_versions

    def compare(self, model_name, version1, version2):
        """Compare metrics between two versions"""
        v1_dir = self.registry_path / model_name / version1
        v2_dir = self.registry_path / model_name / version2

        with open(v1_dir / "metadata.json") as f:
            m1 = json.load(f)
        with open(v2_dir / "metadata.json") as f:
            m2 = json.load(f)

        print(f"\nComparing {version1} vs {version2}:")
        print("-" * 40)

        all_metrics = set(m1["metrics"].keys()) | set(m2["metrics"].keys())
        for metric in sorted(all_metrics):
            v1_val = m1["metrics"].get(metric, "N/A")
            v2_val = m2["metrics"].get(metric, "N/A")
            if isinstance(v1_val, float) and isinstance(v2_val, float):
                diff = v2_val - v1_val
                sign = "+" if diff > 0 else ""
                print(f"  {metric}: {v1_val:.4f} -> {v2_val:.4f} ({sign}{diff:.4f})")
            else:
                print(f"  {metric}: {v1_val} -> {v2_val}")

        return m1, m2

    def rollback(self, model_name, version):
        """Rollback to a previous version by promoting it to production"""
        self.promote(model_name, version, stage="production")
        print(f"Rolled back to {model_name}/{version}")

    def delete(self, model_name, version):
        """Delete a specific version (cannot delete production)"""
        versions = self.manifest["models"].get(model_name, {}).get("versions", [])

        for v in versions:
            if v["version"] == version:
                if v["stage"] == "production":
                    raise ValueError("Cannot delete production version")
                break

        version_dir = self.registry_path / model_name / version
        if version_dir.exists():
            shutil.rmtree(version_dir)

        self.manifest["models"][model_name]["versions"] = [
            v for v in versions if v["version"] != version
        ]
        self._save_manifest()
        print(f"Deleted {model_name}/{version}")

    def get_production_model(self):
        """Get the current production model"""
        prod = self.manifest.get("production_version")
        if not prod:
            raise ValueError("No production model set")

        model_name, version = prod.split("/")
        return self.load(model_name, version)

    def summary(self):
        """Print registry summary"""
        print("\n" + "=" * 50)
        print("MODEL REGISTRY SUMMARY")
        print("=" * 50)
        print(f"Production: {self.manifest.get('production_version', 'None')}")
        print(f"Current: {self.manifest.get('current_version', 'None')}")
        print()

        for name, data in self.manifest["models"].items():
            print(f"{name}:")
            for v in data["versions"]:
                stage_tag = f"[{v['stage'].upper()}]" if v['stage'] != 'development' else ""
                metrics_str = ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in list(v['metrics'].items())[:3])
                print(f"  {v['version']} {stage_tag} - {metrics_str}")
        print()


if __name__ == "__main__":
    # demo usage
    registry = ModelRegistry("models/registry")
    registry.summary()
