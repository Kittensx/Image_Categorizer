import os
import subprocess
import requests
from config_manager import ConfigManager

class SpacyInstall(ConfigManager):
    """Class to handle downloading and installing spaCy models."""

    def __init__(self):
        super().__init__()
        """Initialize repository directory and model URLs."""       
        os.makedirs(self.spacy_model_folder, exist_ok=True)  # Ensure directory exists

        # SpaCy model download links
        self.SPACY_MODELS = {
            "en_core_web_sm": "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
            "en_core_web_md": "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl",
            "en_core_web_lg": "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl",
            "en_core_web_trf": "https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl"
        }

    def download_file(self, url, save_path):
        """Downloads a file from a URL and saves it locally."""
        if os.path.exists(save_path):
            print(f"✅ File already exists: {save_path}")
            return
        
        print(f"⬇️ Downloading {url}...")
        response = requests.get(url, stream=True)
        
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"✅ Downloaded: {save_path}")
        else:
            print(f"❌ Failed to download: {url} (Status Code: {response.status_code})")

    def install_spacy_model(self, model_name):
        """Finds, downloads (if needed), and installs the spaCy model."""
        if model_name not in self.SPACY_MODELS:
            print(f"❌ Model {model_name} is not in the known spaCy models list.")
            return

        # Get the model URL
        url = self.SPACY_MODELS[model_name]
        whl_filename = url.split("/")[-1]
        whl_path = os.path.join(self.spacy_model_folder, whl_filename)

        # Download if not already present
        self.download_file(url, whl_path)

        # Install the model
        try:
            print(f"📦 Installing {model_name} from {whl_path}...")
            subprocess.run(["pip", "install", whl_path], check=True)
            print(f"✅ {model_name} installed successfully!")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {model_name}.")

    def setup_spacy_models(self):
        """Downloads and installs all required spaCy models."""
        for model_name in self.SPACY_MODELS.keys():
            self.install_spacy_model(model_name)

if __name__ == "__main__":
    spacy_installer = SpacyInstall()
    spacy_installer.setup_spacy_models()
    print("\n✅ All spaCy models are set up!")
