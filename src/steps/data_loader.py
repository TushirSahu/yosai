import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


def load_data(
    data_path: str = "data/raw",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads images, normalizes them, and splits into train/test."""
    images, labels = [], []

    # 0 = Normal, 1 = Anomaly
    classes = {"normal": 0, "anomaly": 1}
    for cls, label in classes.items():
        cls_dir = os.path.join(data_path, cls)
        if not os.path.exists(cls_dir):
            continue

        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            img = cv2.imread(img_path)  # pylint: disable=no-member
            if img is not None:
                img = img.astype(np.float32) / 255.0
                images.append(np.transpose(img, (2, 0, 1)))  # NCHW format
                labels.append(label)

    X = np.array(images)
    y = np.array(labels).reshape(-1, 1).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✅ Data Loaded: {len(X_train)} train, {len(X_test)} test.")
    return X_train, y_train, X_test, y_test
