import sys
import os
import argparse
from pathlib import Path

# Add the parent directory to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.modles.train import train
from src.pipeline.models.evaluate import evaluate_model
from config.settings import settings

def main():
    parser = argparse.ArgumentParser(description="Train sentiment analysis model")
    parser.add_argument('--data_path', type=str, default=str(settings.RAW_DATA_DIR / settings.RAW_DATA_FILE), help='Path to data file')
    parser.add_argument('--model_path', type=str, default=str(settings.MODELS_DIR / 'sentiment_model.pkl'), help='Path to save model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model after training')
    
    args = parser.parse_args()
    
    print("Starting training...")
    try:
        train(data_path=Path(args.data_path), model_path=Path(args.model_path))
        print("Training completed.")
        
        if args.evaluate:
            print("Evaluating model...")
            evaluate_model(model_path=Path(args.model_path), data_path=Path(args.data_path))
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
