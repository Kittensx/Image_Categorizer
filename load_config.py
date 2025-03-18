import yaml
import os

def load_config(config_file):
        """Load configuration settings."""
        if not os.path.exists(config_file):
            return {"output_format": "json", "execution_mode": "concurrent", "model_paths": {}}
        with open(config_file, "r") as f:
            return yaml.safe_load(f)