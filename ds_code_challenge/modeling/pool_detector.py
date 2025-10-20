# src/models/pool_detector.py
import logging
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """
    Simple CNN for binary classification.

    Architecture:
    - 3 convolutional blocks with max pooling
    - 2 fully connected layers
    - Dropout for regularization
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        # After 3 pooling layers: 224 -> 112 -> 56 -> 28
        self.fc1 = nn.Linear(64 * 28 * 28, 64)
        self.fc2 = nn.Linear(64, 1)

        # Regularization
        self.dropout = nn.Dropout(0.5)

        # Activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 3, 224, 224)

        Returns
        -------
        torch.Tensor
            Output probabilities of shape (batch_size, 1)
        """
        # Conv block 1
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # Conv block 2
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Conv block 3
        x = self.relu(self.conv3(x))
        x = self.pool(x)

        # Flatten
        x = x.view(-1, 64 * 28 * 28)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))

        return x


class PoolDetector:
    """
    Swimming pool detector using CNN.

    Parameters
    ----------
    model : nn.Module, optional
        PyTorch model (if None, creates SimpleCNN)
    device : torch.device, optional
        torch device (if None, uses CUDA if available)

    Attributes
    ----------
    model : nn.Module
        The neural network model
    device : torch.device
        Device where model is running
    history : dict
        Training history containing losses and accuracies
    """

    def __init__(self, model=None, device=None):
        logger = logging.getLogger(__name__)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.model = model or SimpleCNN()
        self.model = self.model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model created with {total_params:,} parameters")

        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def train(self, train_loader, val_loader=None, epochs=20, lr=0.001):
        """
        Train the model.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            Training DataLoader
        val_loader : torch.utils.data.DataLoader, optional
            Validation DataLoader
        epochs : int, default=20
            Number of training epochs
        lr : float, default=0.001
            Learning rate

        Returns
        -------
        dict
            Training history with keys 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        """
        logger = logging.getLogger(__name__)

        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Learning rate: {lr}")

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Statistics
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # Validation phase
            if val_loader:
                val_loss, val_acc = self._validate(val_loader, criterion)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                logger.info(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
            else:
                logger.info(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                )

        logger.info("✓ Training complete")

        return self.history

    def _validate(self, val_loader, criterion):
        """
        Validate the model.

        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader
            Validation DataLoader
        criterion : nn.Module
            Loss function

        Returns
        -------
        float
            Validation loss
        float
            Validation accuracy
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        return val_loss, val_acc

    def evaluate(self, test_loader):
        """
        Evaluate model on test set.

        Parameters
        ----------
        test_loader : torch.utils.data.DataLoader
            Test DataLoader

        Returns
        -------
        dict
            Dictionary containing:
            - accuracy : float
            - precision : float
            - recall : float
            - f1 : float
            - confusion_matrix : ndarray
            - predictions : ndarray
            - probabilities : ndarray
        """
        logger = logging.getLogger(__name__)

        logger.info("Evaluating model")

        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)

                outputs = self.model(images)
                predicted = (outputs > 0.5).float()

                all_probs.extend(outputs.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        # Convert to numpy arrays
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs).flatten()

        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        logger.info(f"Accuracy:  {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall:    {recall:.4f}")
        logger.info(f"F1 Score:  {f1:.4f}")
        logger.info(f"\nConfusion Matrix:\n{conf_matrix}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": conf_matrix,
            "predictions": all_preds,
            "probabilities": all_probs,
        }

    def predict(self, image_path, image_size=224):
        """
        Predict on a single image.

        Parameters
        ----------
        image_path : str or Path
            Path to image
        image_size : int, default=224
            Target image size

        Returns
        -------
        int
            Prediction (0 or 1)
        float
            Probability
        """

        self.model.eval()

        # Load and transform image
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_tensor = transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            probability = output.item()
            prediction = 1 if probability > 0.5 else 0

        return prediction, probability

    def save(self, model_dir):
        """
        Save model to directory.

        Parameters
        ----------
        model_dir : str or Path
            Directory to save model
        """
        logger = logging.getLogger(__name__)

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "pool_detector.pth"
        torch.save(
            {"model_state_dict": self.model.state_dict(), "history": self.history}, model_path
        )

        logger.info(f"Model saved to {model_path}")

    @classmethod
    def load(cls, model_dir, device=None):
        """
        Load model from directory.

        Parameters
        ----------
        model_dir : str or Path
            Directory containing saved model
        device : torch.device, optional
            Device to load model to

        Returns
        -------
        PoolDetector
            Loaded model instance
        """
        logger = logging.getLogger(__name__)

        model_path = Path(model_dir) / "pool_detector.pth"

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device or "cpu")

        # Create model and load weights
        detector = cls(device=device)
        detector.model.load_state_dict(checkpoint["model_state_dict"])
        detector.history = checkpoint.get("history", {})

        logger.info(f"Model loaded from {model_path}")

        return detector
