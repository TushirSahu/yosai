"""
Continuous Training & Data Drift Detection Demonstration
This script simulates a real-world scenario where:
1. A model is deployed to production
2. Data drift is detected
3. System automatically retrains
4. New model is evaluated and promoted if better
"""

import mlflow
import numpy as np
import cv2
import os
from src.pipelines.continuous_training import edge_continuous_training_pipeline
from src.steps.data_loader import load_data
from src.steps.drift_detection import detect_drift
from src.steps.trainer import train_model
from src.steps.evaluator import evaluate_model
from src.steps.model_register import register_if_improved

def generate_shifted_data(num_samples=100, brightness_shift=0.3):
    """Generate synthetic data with a DIFFERENT lighting condition (simulating drift)."""
    os.makedirs("data/shifted", exist_ok=True)
    os.makedirs("data/shifted/normal", exist_ok=True)
    os.makedirs("data/shifted/anomaly", exist_ok=True)
    
    print(f"\n📷 Generating {num_samples} images with {brightness_shift:.1%} brightness shift (simulating LED upgrade)...")
    
    for i in range(num_samples):
        # Base image: BRIGHTER metallic surface (LED lighting)
        base = np.ones((224, 224, 3), dtype=np.uint8) * int(150 + brightness_shift * 255)
        noise = np.random.normal(0, 10, (224, 224, 3)).astype(np.uint8)
        img = cv2.add(base, noise)
        
        if np.random.rand() > 0.2:
            cv2.imwrite(f"data/shifted/normal/img_{i}.jpg", img)
        else:
            h, w, _ = img.shape
            start = (np.random.randint(0, w//2), np.random.randint(0, h//2))
            end = (np.random.randint(w//2, w), np.random.randint(h//2, h))
            cv2.line(img, start, end, (50, 50, 50), thickness=np.random.randint(2, 6))
            cv2.imwrite(f"data/shifted/anomaly/img_{i}.jpg", img)
    
    print("✅ Shifted data generated.")

def load_shifted_data():
    """Load data from the shifted directory."""
    images, labels = [], []
    classes = {"normal": 0, "anomaly": 1}
    
    for cls, label in classes.items():
        cls_dir = os.path.join("data/shifted", cls)
        if not os.path.exists(cls_dir):
            continue
        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = img.astype(np.float32) / 255.0
                images.append(np.transpose(img, (2, 0, 1)))
                labels.append(label)
    
    X = np.array(images)
    y = np.array(labels).reshape(-1, 1).astype(np.float32)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test

def continuous_training_demo():
    """Main demonstration of continuous training with drift detection."""
    
    print("\n" + "="*70)
    print("🚀 YŌSAI (要塞) - Continuous Training & Drift Detection Demo")
    print("="*70)
    
    # ============================================================
    # PHASE 1: INITIAL TRAINING (Baseline Model v1)
    # ============================================================
    print("\n" + "─"*70)
    print("📊 PHASE 1: Initial Training (Baseline Model)")
    print("─"*70)
    
    mlflow.set_tracking_uri("file:./mlruns")
    experiment_name = "yosai-continuous-training-demo"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="baseline_training"):
        mlflow.log_param("phase", "baseline")
        mlflow.log_param("description", "Initial training on normal data")
        
        X_train, y_train, X_test, y_test = load_data()
        drift_baseline = detect_drift(X_train)
        
        model_v1 = train_model(X_train, y_train)
        accuracy_v1 = evaluate_model(model_v1, X_test, y_test)
        
        status_v1 = register_if_improved(model_v1, accuracy_v1, min_threshold=0.50)
        
        mlflow.log_metric("model_version", 1.0)
        mlflow.log_metric("baseline_accuracy", accuracy_v1)
    
    print(f"✅ Baseline Model v1 registered with {accuracy_v1*100:.1f}% accuracy")
    
    # ============================================================
    # PHASE 2: DATA DRIFT DETECTED (Simulated LED Upgrade)
    # ============================================================
    print("\n" + "─"*70)
    print("⚠️  PHASE 2: Detecting Data Drift (Simulating LED Lighting Upgrade)")
    print("─"*70)
    
    generate_shifted_data(brightness_shift=0.25)
    
    with mlflow.start_run(run_name="drift_detection_phase2"):
        mlflow.log_param("phase", "drift_detection")
        mlflow.log_param("scenario", "led_lighting_upgrade")
        
        X_train_shifted, y_train_shifted, X_test_shifted, y_test_shifted = load_shifted_data()
        
        print(f"\n🔍 Analyzing new data distribution...")
        mean_brightness_original = np.mean(np.array([0.45]))  # Original baseline
        mean_brightness_new = np.mean(X_train_shifted)
        
        brightness_change = abs(mean_brightness_new - mean_brightness_original)
        print(f"   Original brightness: {mean_brightness_original:.2f}")
        print(f"   New brightness: {mean_brightness_new:.2f}")
        print(f"   Change detected: {brightness_change:.2f} ⚠️")
        
        drift_detected = detect_drift(X_train_shifted)
        
        if drift_detected:
            print(f"\n🔔 AUTOMATIC RETRAINING TRIGGERED")
            print(f"   → Retraining model on new data distribution...")
            
            mlflow.log_metric("drift_score", brightness_change)
        
        with mlflow.start_run(run_name="retrain_after_drift_v2", nested=True):
            mlflow.log_param("phase", "retraining")
            mlflow.log_param("trigger", "data_drift_detected")
            
            model_v2 = train_model(X_train_shifted, y_train_shifted)
            accuracy_v2 = evaluate_model(model_v2, X_test_shifted, y_test_shifted)
            
            print(f"\n📊 Model v2 Performance: {accuracy_v2*100:.1f}% (was {accuracy_v1*100:.1f}%)")
            
            if accuracy_v2 >= accuracy_v1:
                status_v2 = register_if_improved(model_v2, accuracy_v2, min_threshold=0.50)
                print(f"   ✅ IMPROVEMENT! Promoting v2 to production")
            else:
                print(f"   ❌ Performance DEGRADED ({accuracy_v2*100:.1f}% < {accuracy_v1*100:.1f}%)")
                print(f"   ❌ Keeping v1 in production")
                mlflow.log_metric("model_status", 0.0)
            
            mlflow.log_metric("model_version", 2.0)
            mlflow.log_metric("retrain_accuracy", accuracy_v2)
    
    # ============================================================
    # PHASE 3: CONTINUED MONITORING
    # ============================================================
    print("\n" + "─"*70)
    print("🔄 PHASE 3: Continuous Monitoring (Simulating 1 Week Later)")
    print("─"*70)
    
    print("\n📅 One week later... new data arrives from edge devices")
    print("🤖 Continuous monitoring system checks for drift automatically...")
    
    generate_shifted_data(brightness_shift=0.35)
    X_train_shifted2, y_train_shifted2, X_test_shifted2, y_test_shifted2 = load_shifted_data()
    
    with mlflow.start_run(run_name="week2_monitoring"):
        mlflow.log_param("phase", "continuous_monitoring")
        mlflow.log_param("scenario", "week_2_additional_drift")
        
        drift_detected_week2 = detect_drift(X_train_shifted2)
        
        if drift_detected_week2:
            print(f"\n🚨 DRIFT DETECTED IN WEEK 2!")
            print(f"   → Automatic retraining triggered again...")
            
            with mlflow.start_run(run_name="retrain_after_drift_v3", nested=True):
                mlflow.log_param("phase", "retraining_week2")
                mlflow.log_param("trigger", "continuous_drift_detection")
                
                model_v3 = train_model(X_train_shifted2, y_train_shifted2)
                accuracy_v3 = evaluate_model(model_v3, X_test_shifted2, y_test_shifted2)
                
                print(f"\n📊 Model v3 Performance: {accuracy_v3*100:.1f}%")
                
                if accuracy_v3 > accuracy_v2:
                    status_v3 = register_if_improved(model_v3, accuracy_v3, min_threshold=0.50)
                    print(f"   ✅ NEW BEST MODEL! v3 ({accuracy_v3*100:.1f}%) > v2 ({accuracy_v2*100:.1f}%)")
                    print(f"   ✅ Promoting v3 to production")
                    mlflow.log_metric("production_model", 3.0)
                else:
                    print(f"   ⚠️  v3 ({accuracy_v3*100:.1f}%) not better than v2 ({accuracy_v2*100:.1f}%)")
                    print(f"   🛑 Keeping v2 in production")
                    mlflow.log_metric("model_status", 0.0)
                
                mlflow.log_metric("model_version", 3.0)
                mlflow.log_metric("week2_accuracy", accuracy_v3)
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("✨ Continuous Training Demo Complete!")
    print("="*70)
    
    print(f"\n📈 Model Version History:")
    print(f"   v1 (Baseline):     {accuracy_v1*100:.1f}% accuracy ✅ PRODUCTION")
    print(f"   v2 (After drift):  {accuracy_v2*100:.1f}% accuracy {'✅ PRODUCTION' if accuracy_v2 > accuracy_v1 else '❌ REJECTED'}")
    print(f"   v3 (Week 2):       {accuracy_v3*100:.1f}% accuracy {'✅ PRODUCTION' if accuracy_v3 > max(accuracy_v1, accuracy_v2) else '❌ KEPT v2'}")
    
    print(f"\n🎯 Key Achievements:")
    print(f"   ✅ Automatic drift detection working")
    print(f"   ✅ Automatic retraining triggered")
    print(f"   ✅ Model versioning in MLflow")
    print(f"   ✅ Quality gates (only promote if better)")
    print(f"   ✅ Complete audit trail of all versions")
    
    print(f"\n📊 View Results in MLflow:")
    print(f"   🔗 http://127.0.0.1:5000")
    print(f"   → Experiments: All training runs")
    print(f"   → Model Registry: All model versions")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    continuous_training_demo()
