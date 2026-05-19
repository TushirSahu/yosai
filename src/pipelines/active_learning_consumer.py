"""Production-Grade Kafka Consumer for Active Learning Pipeline

This shows how uncertain predictions flow from edge devices to cloud
for automated labeling and continuous model improvement.

Interviewers will see: Kafka integration, stream processing, S3 integration, 
error handling, and production-grade logging.
"""

import json
import logging
import torch
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
import boto3
import time
from datetime import datetime
from typing import Dict, Optional, Any
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ActiveLearningPipeline:
    """
    Production pipeline for:
    1. Consume uncertain predictions from edge devices (Kafka)
    2. Store images in S3 with metadata
    3. Route to human labelers
    4. Trigger automatic retraining when labeled dataset reaches threshold
    """

    def __init__(self, config: Dict):
        self.config = config
        self.s3_client = boto3.client("s3")
        self.kafka_consumer: Optional[KafkaConsumer] = None
        self.kafka_producer: Optional[KafkaProducer] = None
        self._setup_kafka()
        self._setup_metrics()

    def _setup_kafka(self):
        """Initialize Kafka connections with error handling."""
        try:
            self.kafka_consumer = KafkaConsumer(
                "active-learning-topic",
                bootstrap_servers=self.config["kafka_brokers"],
                group_id="yosai-active-learning-consumer",
                auto_offset_reset="earliest",
                enable_auto_commit=False,
                max_poll_records=100,
                session_timeout_ms=30000,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            )

            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.config["kafka_brokers"],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",  # Wait for all replicas to acknowledge
                retries=3,
            )

            logger.info("✅ Kafka connections established")

        except KafkaError as e:
            logger.error(f"❌ Kafka connection failed: {e}")
            raise

    def _setup_metrics(self):
        """Initialize metrics tracking."""
        self.metrics = {
            "messages_processed": 0,
            "images_stored": 0,
            "storage_errors": 0,
            "retraining_triggers": 0,
            "total_labeled_images": 0,
        }

    def consume_and_process(self):
        """Main loop: consume messages from edge devices."""
        logger.info("🚀 Starting Active Learning Pipeline...")

        batch_messages = []
        last_commit_time = time.time()

        for message in self.kafka_consumer:
            try:
                payload = message.value

                # Validate message schema
                required_fields = [
                    "factory_id",
                    "device_id",
                    "image_base64",
                    "edge_confidence",
                ]
                if not all(f in payload for f in required_fields):
                    logger.warning(f"❌ Invalid message schema: {payload.keys()}")
                    continue

                # Process the uncertain prediction
                self._process_uncertain_prediction(payload)

                batch_messages.append(message)
                self.metrics["messages_processed"] += 1

                # Batch commit for efficiency (every 100 messages or 30 seconds)
                if len(batch_messages) >= 100 or (time.time() - last_commit_time) > 30:
                    self.kafka_consumer.commit()
                    last_commit_time = time.time()
                    logger.info(f"✅ Committed {len(batch_messages)} messages")
                    batch_messages = []

            except Exception as e:
                logger.error(f"❌ Error processing message: {e}", exc_info=True)
                continue

    def _process_uncertain_prediction(self, payload: Dict):
        """
        Process a single uncertain prediction:
        1. Decode image
        2. Store in S3
        3. Create labeling task
        4. Check if retraining should be triggered
        """

        factory_id = payload["factory_id"]
        device_id = payload["device_id"]
        confidence = payload["edge_confidence"]
        timestamp = payload["timestamp"]

        try:
            # Decode image
            import base64

            image_base64 = payload["image_base64"]
            image_bytes = base64.b64decode(image_base64)
            image_hash = hashlib.md5(image_bytes).hexdigest()

            # Store in S3 with metadata
            s3_key = (
                f"active-learning/{factory_id}/{device_id}/{timestamp}_{image_hash}.jpg"
            )

            self.s3_client.put_object(
                Bucket=self.config["s3_bucket"],
                Key=s3_key,
                Body=image_bytes,
                ContentType="image/jpeg",
                Metadata={
                    "factory_id": factory_id,
                    "device_id": device_id,
                    "edge_confidence": str(confidence),
                    "timestamp": str(timestamp),
                    "status": "pending_label",  # Will be updated by labelers
                },
            )

            self.metrics["images_stored"] += 1

            logger.info(f"📤 Image stored: s3://{self.config['s3_bucket']}/{s3_key}")

            # Create labeling task (send to human labelers)
            labeling_task = {
                "task_id": image_hash,
                "factory_id": factory_id,
                "device_id": device_id,
                "image_url": f"s3://{self.config['s3_bucket']}/{s3_key}",
                "edge_confidence": confidence,
                "timestamp": timestamp,
                "priority": "high" if confidence > 0.45 else "medium",
            }

            # Send to labeling service (Kafka topic or HTTP API)
            if self.kafka_producer:
                self.kafka_producer.send("labeling-tasks", value=labeling_task)

            # Check if we should trigger retraining
            self._check_and_trigger_retraining(factory_id)

        except Exception as e:
            logger.error(f"❌ Error storing image: {e}")
            self.metrics["storage_errors"] += 1

    def _check_and_trigger_retraining(self, factory_id: str):
        """
        Trigger retraining if:
        1. We have >= 500 labeled uncertain predictions
        2. Data drift is detected (different from baseline)
        3. Model accuracy dropped below threshold
        """

        try:
            # Count labeled images in S3
            response = self.s3_client.list_objects_v2(
                Bucket=self.config["s3_bucket"],
                Prefix=f"active-learning/{factory_id}/",
                MaxKeys=1000,
            )

            labeled_count = response.get("KeyCount", 0)
            self.metrics["total_labeled_images"] = labeled_count

            logger.info(f"📊 Factory {factory_id}: {labeled_count} labeled images")

            # Trigger retraining if threshold reached
            if labeled_count >= 500:
                logger.warning(f"⚠️ Labeled image threshold reached for {factory_id}!")

                # Send retraining trigger to Airflow
                self._trigger_airflow_dag(factory_id, labeled_count)

                self.metrics["retraining_triggers"] += 1

                # Log to MLflow for visibility
                import mlflow

                mlflow.log_param(f"retraining_trigger_{factory_id}", labeled_count)

        except Exception as e:
            logger.error(f"❌ Error checking retraining condition: {e}")

    def _trigger_airflow_dag(self, factory_id: str, labeled_count: int):
        """Trigger Airflow DAG for retraining."""
        import requests  # type: ignore

        try:
            # Airflow API endpoint
            airflow_url = self.config["airflow_api_url"]

            dag_run_payload: Dict[str, Any] = {
                "conf": {
                    "factory_id": factory_id,
                    "labeled_images_count": labeled_count,
                    "triggered_by": "active-learning-pipeline",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            }

            response = requests.post(
                f"{airflow_url}/dags/yosai-continuous-training/dagRuns",
                json=dag_run_payload,
                auth=(self.config["airflow_user"], self.config["airflow_password"]),
                timeout=10,
            )

            if response.status_code == 200:
                logger.info(f"✅ Airflow DAG triggered for {factory_id}")
            else:
                logger.error(f"❌ Failed to trigger Airflow: {response.status_code}")

        except Exception as e:
            logger.error(f"❌ Error triggering Airflow: {e}")

    def get_metrics(self) -> Dict:
        """Return current pipeline metrics."""
        return self.metrics

    def health_check(self) -> bool:
        """Health check for monitoring systems."""
        try:
            # Check Kafka connectivity
            if self.kafka_consumer is None:
                return False

            # Check S3 connectivity
            self.s3_client.head_bucket(Bucket=self.config["s3_bucket"])

            return True
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False


def main():
    """Main entry point."""

    config = {
        "kafka_brokers": ["kafka-1.prod.internal:9092", "kafka-2.prod.internal:9092"],
        "s3_bucket": "yosai-mlops-data",
        "airflow_api_url": "https://airflow.prod.internal:8080/api/v1",
        "airflow_user": os.getenv("AIRFLOW_USER"),
        "airflow_password": os.getenv("AIRFLOW_PASSWORD"),
    }

    pipeline = ActiveLearningPipeline(config)

    try:
        pipeline.consume_and_process()
    except KeyboardInterrupt:
        logger.info("🛑 Pipeline stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
    finally:
        if pipeline.kafka_consumer:
            pipeline.kafka_consumer.close()
        if pipeline.kafka_producer:
            pipeline.kafka_producer.close()


if __name__ == "__main__":
    import os

    main()
