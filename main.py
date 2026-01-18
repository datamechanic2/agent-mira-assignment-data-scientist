"""
Main entry point - run training or API server
EDA is now in notebooks/eda_and_feature_engineering.ipynb
"""
import argparse
import subprocess
import sys
from pathlib import Path
import os


def run_training():
    print("\n=== Running Model Training ===")
    subprocess.run([sys.executable, "src/model_training.py"], check=True)


def run_api(host="0.0.0.0", port=8000):
    print(f"\n=== Starting API at http://{host}:{port} ===")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api.app:app",
        "--host", host,
        "--port", str(port),
        "--reload"
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="House Price Prediction Pipeline")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--api", action="store_true", help="Start API server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")

    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    if args.train:
        run_training()
    elif args.api:
        run_api(args.host, args.port)
    else:
        # default: run training
        run_training()
        print("\n=== Training Complete ===")
        print("Run 'python main.py --api' to start the API")
        print("See notebooks/eda_and_feature_engineering.ipynb for EDA")
