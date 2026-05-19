# Yōsai (要塞) - Edge-to-Cloud MLOps Pipeline for Manufacturing

> A production-grade MLOps system demonstrating continuous training, data drift detection, and automated model deployment—tailored for Japan's precision manufacturing industry.

## 🎯 Why This Project Stands Out

### 1. **Complete MLOps Lifecycle**
Not just a Jupyter notebook. This shows the **entire ML engineering pipeline**:
- Data ingestion from edge devices (Kafka streaming)
- Automated drift detection (Evidently AI patterns)
- Triggered retraining pipelines (ZenML orchestration)
- Model versioning & registry (MLflow)
- CI/CD automation (GitHub Actions)
- Infrastructure as Code (Terraform/Kubernetes)

### 2. **Manufacturing-First Design**
Built for **Japan's industrial automation leaders** (Fanuc, Keyence, Sony):
- Edge inference on factory floors (NVIDIA Triton)
- Active learning loop (uncertain predictions → human review → retraining)
- Strict quality gates (99.5% recall requirement)
- Multi-tenancy support (multiple factory locations)

### 3. **Real-World Problem Solving**
Addresses the **#1 challenge in production ML**:
- ❌ Problem: Train model once, deploy, and pray
- ✅ Solution: Continuous monitoring, automatic retraining, quality gates

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      FACTORY FLOOR (Edge)                        │
├─────────────────────────────────────────────────────────────────┤
│  Camera Stream → NVIDIA Triton Inference → Confidence Filter   │
│                                              ↓                   │
│                                    High Confidence: Local inference
│                                    Low Confidence: Stream to Cloud
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    Apache Kafka Topic
                           │
        ┌──────────────────┴──────────────────┐
        ↓                                      ↓
    ┌─────────────────┐              ┌──────────────────┐
    │  MinIO/S3       │              │  Human Labeling  │
    │  (Data Lake)    │              │  (Active Learning)
    └────────┬────────┘              └────────┬─────────┘
             │                                │
             └──────────────┬─────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │   ZenML Pipeline Orchestration        │
        ├───────────────────────────────────────┤
        │  1. Load Data                         │
        │  2. Evidently AI Drift Detection      │
        │  3. PyTorch Model Training            │
        │  4. Strict Evaluation Gates           │
        │  5. MLflow Model Registry             │
        └──────┬────────────────┬───────────────┘
               │                │
          Promote ✅        Reject ❌
               │                │
               ↓                ↓
        ┌─────────────┐   Keep v(n-1)
        │  GitHub     │
        │  Actions    │
        └─────┬───────┘
              ↓
       Docker Build & Push
              ↓
       Deploy to Edge Devices
              ↓
       Model Serving (ONNX/Triton)
```

---

## 🚀 Quick Start

### Prerequisites
```bash
# Install Python 3.10+
python --version

# Install dependencies in virtual environment
cd yosai_mlops
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Basic Pipeline
```bash
# Generate synthetic data (100 images)
python src/data/generate_synthetic_data.py

# Run the complete MLOps pipeline
python run_pipeline.py

# View results in MLflow UI
mlflow ui
# Open http://127.0.0.1:5000
```

### Run the Continuous Training Demo (⭐ IMPRESSIVE FOR INTERVIEWS)
```bash
# This demonstrates the full continuous training loop
python demonstrate_continuous_training.py
```

**What this shows:**
1. Initial model trained (v1: 78% accuracy)
2. Data drift detected (LED lighting upgrade)
3. Automatic retraining triggered (v2)
4. Performance comparison (keep v1 or promote v2?)
5. Week 2 monitoring (continuous detection)
6. New model v3 automatically pushed to production if better

---

## 📁 Project Structure

```
yosai_mlops/
├── README.md                              # This file
├── CONTINUOUS_TRAINING_DEMO.md            # ⭐ Show this to interviewers
├── demonstrate_continuous_training.py     # ⭐ Run this demo
├── run_pipeline.py                        # Simple pipeline execution
├── requirements.txt                       # Dependencies
│
├── src/
│   ├── models/
│   │   └── anomaly_net.py                # PyTorch CNN (edge-optimized)
│   ├── pipelines/
│   │   └── continuous_training.py        # Main ZenML pipeline
│   ├── steps/
│   │   ├── data_loader.py                # Data ingestion
│   │   ├── drift_detection.py            # Evidently AI patterns
│   │   ├── trainer.py                    # PyTorch training
│   │   ├── evaluator.py                  # Model evaluation
│   │   └── model_register.py             # MLflow registry
│   └── data/
│       └── generate_synthetic_data.py    # Synthetic data gen
│
├── edge_inferencing/
│   ├── edge_client.py                    # Runs on factory floor
│   └── model_repository/
│       └── anomaly_detector/
│           └── config.pbtxt              # Triton config
│
├── infrastructure/
│   ├── main.tf                           # Terraform (Kubernetes)
│   ├── deployment.yaml                   # K8s deployment
│   └── hpa.yaml                          # Auto-scaling
│
├── .github/workflows/
│   └── mlops_pipeline.yml                # CI/CD automation
│
└── mlruns/                               # MLflow tracking data
```

---

## 🎓 Key MLOps Concepts Demonstrated

### 1. Data Drift Detection
```python
# In src/steps/drift_detection.py
def detect_drift(X_train):
    mean_brightness = np.mean(X_train)
    drift_detected = not (0.2 < mean_brightness < 0.8)
    
    if drift_detected:
        print("⚠️ DRIFT DETECTED! Triggering retraining...")
    return drift_detected
```

**In production:** Use Evidently AI's statistical profiles
- Population Stability Index (PSI)
- Kolmogorov-Smirnov test
- Embedding distributions

### 2. Automated Retraining
```python
# Triggered automatically when drift_detected == True
if drift_detected:
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    
    if accuracy > threshold:
        register_model_in_mlflow()  # New production version
    else:
        keep_previous_version()     # Rollback
```

### 3. Model Registry & Versioning
```
mlruns/
└── models/
    └── yosai-anomaly-detector/
        ├── version-1/   (Baseline - Production)
        ├── version-2/   (After drift - Rejected)
        └── version-3/   (After drift - Staging)
```

Each version has:
- Model weights (PyTorch)
- Metadata (hyperparameters, metrics)
- Provenance (which data, which commit)
- Status (Production, Staging, Archived)

---
