import mlflow
from src.steps.data_loader import load_data
from src.steps.drift_detection import detect_drift
from src.steps.trainer import train_model
from src.steps.evaluator import evaluate_model
from src.steps.model_register import register_if_improved

def edge_continuous_training_pipeline():
    """MLOps Pipeline defining the continuous training loop."""
    print("\n" + "="*60)
    print("🚀 YŌSAI (要塞) - Edge MLOps Pipeline Started")
    print("="*60 + "\n")
    
    # 1. Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # 2. Detect data drift
    drift = detect_drift(X_train)
    
    # 3. Train the edge model
    model = train_model(X_train, y_train)
    
    # 4. Evaluate its performance
    accuracy = evaluate_model(model, X_test, y_test)
    
    # 5. Register conditional logic (55% threshold for synthetic data demo)
    status = register_if_improved(model, accuracy, min_threshold=0.55)
    
    print("\n" + "="*60)
    print(f"✨ Pipeline Complete - Final Status: {status}")
    print("="*60 + "\n")

if __name__ == "__main__":
    edge_continuous_training_pipeline()
