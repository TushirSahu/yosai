import numpy as np
import json
# import tritonclient.http as triton_http
# from kafka import KafkaProducer

def process_frame(frame: np.ndarray):
    """
    Simulates Active learning logic at the Edge.
    """
    print("Simulating inference via Triton...")
    anomaly_score = np.random.uniform(0.0, 1.0)
    
    if 0.4 < anomaly_score < 0.6:
        payload = {
            "edge_score": float(anomaly_score),
            "status": "needs_review"
        }
        print(f"Uncertain prediction (Score: {anomaly_score:.2f}) sent to cloud for human labeling via Kafka.")
        
    return anomaly_score

if __name__ == "__main__":
    dummy_frame = np.random.rand(3, 224, 224).astype(np.float32)
    process_frame(dummy_frame)
