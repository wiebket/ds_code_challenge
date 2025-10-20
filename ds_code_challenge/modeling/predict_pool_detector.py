import logging
import sys

from ds_code_challenge.config import Config
from ds_code_challenge.modeling.pool_detector import PoolDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_image(image_path):
    """
    Predict if an image contains a swimming pool.

    Parameters
    ----------
    image_path : str or Path
        Path to image file

    Returns
    -------
    int
        Prediction (0=no pool, 1=has pool)
    float
        Confidence probability
    """
    # Load model
    model_dir = Config.MODEL_DIR / "object_detection"
    detector = PoolDetector.load(model_dir)

    # Predict
    prediction, probability = detector.predict(image_path)

    # Display results
    logger.info(f"\nImage: {image_path}")
    logger.info(f"Prediction: {'Has Pool' if prediction == 1 else 'No Pool'}")
    logger.info(f"Confidence: {probability:.2%}")

    return prediction, probability


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict_pool.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    predict_image(image_path)
