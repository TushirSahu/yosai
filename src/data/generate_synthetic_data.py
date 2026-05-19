import numpy as np
import os
import cv2


def generate_dataset(num_samples=100, output_dir="data/raw"):
    os.makedirs(f"{output_dir}/normal", exist_ok=True)
    os.makedirs(f"{output_dir}/anomaly", exist_ok=True)

    print(f"Generating {num_samples} synthetic images...")

    for i in range(num_samples):
        # Base image: gray metallic surface (simulated)
        base = np.ones((224, 224, 3), dtype=np.uint8) * 150
        noise = np.random.normal(0, 10, (224, 224, 3)).astype(np.uint8)
        img = cv2.add(base, noise)  # pylint: disable=no-member
        # 80% Normal, 20% Anomaly
        if np.random.rand() > 0.2:
            cv2.imwrite(
                f"{output_dir}/normal/img_{i}.jpg", img
            )  # pylint: disable=no-member
        else:
            # Add synthetic "scratch" or "dent" anomaly
            h, w, _ = img.shape
            start = (np.random.randint(0, w // 2), np.random.randint(0, h // 2))
            end = (np.random.randint(w // 2, w), np.random.randint(h // 2, h))
            cv2.line(
                img, start, end, (50, 50, 50), thickness=np.random.randint(2, 6)
            )  # pylint: disable=no-member
            cv2.imwrite(
                f"{output_dir}/anomaly/img_{i}.jpg", img
            )  # pylint: disable=no-member

    print(f"Data saved to {output_dir}/")


if __name__ == "__main__":
    generate_dataset()
