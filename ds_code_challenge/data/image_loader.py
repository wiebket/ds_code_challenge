import logging
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class PoolDataset(Dataset):
    """
    PyTorch Dataset for swimming pool images.

    Parameters
    ----------
    image_paths : list of str
        List of paths to images
    labels : list of int
        List of labels (0 or 1)
    transform : torchvision.transforms, optional
        Optional transform to apply to images
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label


def get_transforms(image_size=224, augment=True):
    """
    Get image transforms for training and validation.

    Parameters
    ----------
    image_size : int, default=224
        Target image size
    augment : bool, default=True
        Whether to apply data augmentation (for training)

    Returns
    -------
    torchvision.transforms.Compose
        Composed transforms
    """
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        # Validation/test transforms (no augmentation)
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return transform


def load_swimming_pool_dataset(
    data_dir, image_size=224, batch_size=32, test_size=0.2, random_state=42
):
    """
    Load swimming pool classification dataset and create DataLoaders.

    Parameters
    ----------
    data_dir : str or Path
        Path to data/raw/images/swimming-pool directory
    image_size : int, default=224
        Target image size
    batch_size : int, default=32
        Batch size for DataLoaders
    test_size : float, default=0.2
        Proportion for test set
    random_state : int, default=42
        Random seed

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        Training DataLoader
    test_loader : torch.utils.data.DataLoader
        Test DataLoader
    class_names : list of str
        List of class names
    """
    logger = logging.getLogger(__name__)

    data_dir = Path(data_dir)

    # Paths to yes/no directories
    yes_dir = data_dir / "yes"
    no_dir = data_dir / "no"

    logger.info(f"Loading dataset from {data_dir}")

    image_paths = []
    labels = []

    # Load "yes" images (has pool)
    yes_manifest = yes_dir / "manifest.txt"
    if yes_manifest.exists():
        with open(yes_manifest, "r") as f:
            yes_files = [line.strip() for line in f if line.strip()]
        logger.info(f"Found {len(yes_files)} images with pools (from manifest)")
    else:
        yes_files = [f.name for f in yes_dir.glob("*.tif")]
        logger.info(f"Found {len(yes_files)} images with pools (from directory)")

    for filename in yes_files:
        img_path = yes_dir / filename
        if img_path.exists():
            image_paths.append(str(img_path))
            labels.append(1)  # 1 = has pool

    # Load "no" images (no pool)
    no_manifest = no_dir / "manifest.txt"
    if no_manifest.exists():
        with open(no_manifest, "r") as f:
            no_files = [line.strip() for line in f if line.strip()]
        logger.info(f"Found {len(no_files)} images without pools (from manifest)")
    else:
        no_files = [f.name for f in no_dir.glob("*.tif")]
        logger.info(f"Found {len(no_files)} images without pools (from directory)")

    for filename in no_files:
        img_path = no_dir / filename
        if img_path.exists():
            image_paths.append(str(img_path))
            labels.append(0)  # 0 = no pool

    logger.info(f"Loaded {len(image_paths)} images total")
    logger.info(f"  With pool: {sum(labels)}")
    logger.info(f"  Without pool: {len(labels) - sum(labels)}")

    # Split into train/test
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    logger.info(f"Train set: {len(train_paths)} images")
    logger.info(f"Test set: {len(test_paths)} images")

    # Create datasets
    train_dataset = PoolDataset(
        train_paths, train_labels, transform=get_transforms(image_size, augment=True)
    )

    test_dataset = PoolDataset(
        test_paths, test_labels, transform=get_transforms(image_size, augment=False)
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    class_names = ["no_pool", "has_pool"]

    return train_loader, test_loader, class_names
