import logging

import matplotlib.pyplot as plt

from ds_code_challenge.config import Config
from ds_code_challenge.data.image_loader import load_swimming_pool_dataset
from ds_code_challenge.modeling.pool_detector import PoolDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics.

    Parameters
    ----------
    history : dict
        Training history with keys 'train_loss', 'train_acc', 'val_loss', 'val_acc'
    save_path : str or Path, optional
        Path to save plot
    """
    logger = logging.getLogger(__name__)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Accuracy
    ax1.plot(epochs, history["train_acc"], label="Train")
    if history["val_acc"]:
        ax1.plot(epochs, history["val_acc"], label="Validation")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    # Loss
    ax2.plot(epochs, history["train_loss"], label="Train")
    if history["val_loss"]:
        ax2.plot(epochs, history["val_loss"], label="Validation")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Training plot saved to {save_path}")

    plt.show()


def train_pool_detector():
    """
    Train swimming pool detection model.

    Returns
    -------
    PoolDetector
        Trained model
    dict
        Evaluation results
    """
    logger.info("=" * 60)
    logger.info("SWIMMING POOL DETECTION - TRAINING (PyTorch)")
    logger.info("=" * 60)

    # 1. Load data
    data_dir = Config.DATA_DIR / "raw" / "images" / "swimming-pool"

    train_loader, test_loader, class_names = load_swimming_pool_dataset(
        data_dir=data_dir, image_size=224, batch_size=32, test_size=0.2, random_state=42
    )

    # 2. Create model
    detector = PoolDetector()

    # Print model architecture
    logger.info("\nModel Architecture:")
    logger.info(detector.model)

    # 3. Train model
    history = detector.train(
        train_loader=train_loader, val_loader=test_loader, epochs=20, lr=0.001
    )

    # 4. Plot training history
    plot_path = Config.PROJECT_ROOT / "outputs" / "figures" / "pool_detector_training.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_training_history(history, save_path=plot_path)

    # 5. Evaluate
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 60)

    results = detector.evaluate(test_loader)

    # 6. Save model
    model_dir = Config.MODEL_DIR / "object_detection"
    detector.save(model_dir)

    # 7. Save results
    import json

    results_path = Config.PROJECT_ROOT / "outputs" / "pool_detector_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    results_json = {
        "accuracy": float(results["accuracy"]),
        "precision": float(results["precision"]),
        "recall": float(results["recall"]),
        "f1": float(results["f1"]),
        "confusion_matrix": results["confusion_matrix"].tolist(),
    }

    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)

    logger.info(f"\n✓ Results saved to {results_path}")
    logger.info("✓ Training complete")

    return detector, results


if __name__ == "__main__":
    detector, results = train_pool_detector()
