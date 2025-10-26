# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "mlflow",
#    "tqdm",
#   ]
# ///
import mlflow
import os
import re
from pathlib import Path
from tqdm import tqdm
import pandas as pd

mlflow.set_tracking_uri("http://mlflow.junebug.lan:80")
mlflow.set_experiment("admete-archive")

def parse_config(config_path):
    """Parse config.txt into parameters dict"""
    params = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if ':' in line and "<" not in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                if key == "model":
                    break  # Stop parsing at 'Model' section, because it's not really parsable
                value = value.strip()
                
                # Try to parse as number
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except:
                    pass
                
                params[key] = value
    return params

def parse_metrics(metrics_path):
    """Parse metrics.csv into timestamped metrics"""
    df = pd.read_csv(metrics_path)
    return df

def upload_experiment(exp_dir):
    """Upload a single experiment folder to MLflow"""
    exp_name = exp_dir.name
    
    print(f"\nUploading {exp_name}...")
    
    # Parse config
    config_path = exp_dir / "config.txt"
    if not config_path.exists():
        print(f"  Skipping {exp_name} - no config.txt")
        return
    
    params = parse_config(config_path)
    
    # Create run with timestamp from folder name
    run_name = exp_name
    
    with mlflow.start_run(run_name=run_name):
        # Set timestamp tag
        mlflow.set_tag("folder_name", exp_name)
        mlflow.set_tag("migrated_from", "folder_archive")
        mlflow.set_tag("original_path", str(exp_dir))
        
        # Log all parameters
        mlflow.log_params(params)
        
        # Log metrics from CSV if it exists
        metrics_path = exp_dir / "metrics.csv"
        if metrics_path.exists():
            df = parse_metrics(metrics_path)
            for _, row in df.iterrows():
                step = row.get('step', 0)
                for col in df.columns:
                    if col not in ['step']:
                      mlflow.log_metric(col, float(row[col]), step=int(step))
        
        # Log artifacts
        artifacts_to_log = [
            ("config.txt", "config"),
            ("training.log", "logs"),
            ("training.py", "code"),
            ("metrics.csv", "metrics"),
        ]
        
        for filename, artifact_path in artifacts_to_log:
            file_path = exp_dir / filename
            if file_path.exists():
              mlflow.log_artifact(str(file_path), artifact_path=artifact_path)
        
        # Log checkpoints (can be selective here)
        checkpoint_files = list(exp_dir.glob("state_*"))
        if checkpoint_files:
            # Log only final checkpoint by default (to save space)
            # Or log all if you want
            final_checkpoint = max(checkpoint_files, key=lambda p: int(p.name.split('_')[1]))
            mlflow.log_artifact(str(final_checkpoint), artifact_path="checkpoints")
            
            # Also log the 'state' symlink/final state if it exists
            state_path = exp_dir / "state"
            if state_path.exists() and not state_path.is_symlink():
                mlflow.log_artifact(str(state_path), artifact_path="checkpoints")
        
def main():
    archive_path = Path.home() / "share/junebug/chess/training/checkpoints"
    
    # Get all experiment directories
    exp_dirs = sorted([d for d in archive_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(exp_dirs)} experiments to upload")
    print(f"Uploading to: {mlflow.get_tracking_uri()}")
    
    # Upload with progress bar
    for exp_dir in exp_dirs:
      upload_experiment(exp_dir)

if __name__ == "__main__":
    main()