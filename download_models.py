import os
import yaml
import subprocess
import shutil
import time
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel, AutoModel, AutoProcessor
from clip_interrogator import Config, Interrogator
from ultralytics import YOLO
from nudenet import NudeDetector
from deepface import DeepFace
import logging
import spacy
from download_spacy import SpacyInstall
import requests
import zipfile
import timm
from torchvision import models
import torch
import cv2
import imagehash
from PIL import Image
import tensorflow as tf
from keras.applications.xception import Xception
from tqdm import tqdm
from config_manager import ConfigManager



class DownloadModels(ConfigManager):
    def __init__(self):
        super().__init__()
        self.CONFIG_FILE = "config.yaml"
        self.Config = self.load_config()
        self.model_paths = self.Config.get("model_paths", {})
        
        
        LOG_FILE = "download_errors.log"
        logging.basicConfig(filename=LOG_FILE, level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
    
   
    def load_config(self):
        """Loads model paths from config.yaml"""
        if not os.path.exists(self.CONFIG_FILE):
            print(f"⚠️ Config file '{self.CONFIG_FILE}' not found! Using default paths.")
            return {
                "model_paths": {
                    "blip": "repositories/blip/",
                    "clip": "repositories/clip/",
                    "yolov8": "repositories/yolo/yolov8n.pt",
                    "nudenet": "repositories/nudenet/",
                }
            }
        with open(self.CONFIG_FILE, "r") as f:
            return yaml.safe_load(f)

   
    

    # ---------------- HELPER FUNCTIONS ----------------
    @staticmethod
    def ensure_directory(path):
        """Creates the directory if it doesn't exist."""
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def run_command(command):
        """Runs a shell command and handles errors."""
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running command: {command}")
            print(e)

    # ---------------- MODEL DOWNLOAD FUNCTIONS ----------------
    def download_blip(self):
        """Downloads BLIP-2 model."""
        blip_dir = Path(self.model_paths.get("blip", "repositories/blip/"))
        self.ensure_directory(blip_dir)

        if not (blip_dir / "pytorch_model.bin").exists():
            print("⬇️ Downloading BLIP-2 model...")
            BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir=str(blip_dir))
            BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir=str(blip_dir))
        else:
            print("✅ BLIP-2 model already exists.")



    def download_clip(self):
        """Downloads CLIP models from Hugging Face and saves them into repositories/clip/."""
        clip_dir = Path(self.model_paths.get("clip", "repositories/clip/")).resolve()
        self.ensure_directory(clip_dir)

        # Expected files needed for a complete CLIP model
        required_files = {"config.json", "preprocessor_config.json", "tokenizer.json", "special_tokens_map.json"}

        # Check if any required file is missing
        missing_files = [file for file in required_files if not (clip_dir / file).exists()]
        
        if missing_files:
            print(f"⚠️ Missing CLIP files: {missing_files}. Downloading from Hugging Face...")

            try:
                model_name = "openai/clip-vit-large-patch14"  # Adjust this if using another CLIP model
                processor = CLIPProcessor.from_pretrained(model_name)
                model = CLIPModel.from_pretrained(model_name)

                # Save the model in the correct directory
                processor.save_pretrained(str(clip_dir))
                model.save_pretrained(str(clip_dir))

                print(f"✅ CLIP model downloaded and saved to {clip_dir}")

            except Exception as e:
                print(f"❌ Failed to download CLIP model: {e}")
                return

        else:
            print(f"✅ CLIP model already exists in {clip_dir}")


    def download_yolo(self):
        """Downloads YOLOv8 model and ensures it is stored in the correct directory."""
        yolo_dir = Path(self.model_paths.get("yolov8", "repositories/yolo/"))  # ✅ Corrected directory
        yolo_path = yolo_dir / "yolov8n.pt"  # ✅ Ensure correct filename        
        self.ensure_directory(yolo_dir)  # ✅ Create directory if missing

        if not yolo_path.exists():
            print(f"⬇️ Downloading YOLOv8 model to temporary location...")

            # ✅ This will trigger the YOLO model download
            model = YOLO("yolov8n.pt")

            # ✅ Check possible download locations
            possible_locations = [
                Path.home() / ".torch/models/yolov8n.pt",  # Default YOLO location
                Path.home() / "yolov8n.pt",  # Sometimes saves in home directory
                Path("yolov8n.pt"),  # Current directory (root)
            ]

            # ✅ Move the model if found
            for default_path in possible_locations:
                if default_path.exists():
                    shutil.move(str(default_path), str(yolo_path))
                    print(f"✅ YOLOv8 model moved to {yolo_path}")
                    break  # ✅ Stop checking once the file is moved
            else:
                print("⚠️ YOLO model was not found in expected locations!")

        else:
            print(f"✅ YOLOv8 model already exists at {yolo_path}.")





    def download_nudenet(self):
        """Downloads NudeNet model and stores it in repositories/nudenet/."""
        nudenet_dir = Path(self.model_paths.get("nudenet", "repositories/nudenet/"))
        self.ensure_directory(nudenet_dir)

        model_path = nudenet_dir / "classifier.onnx"  # ✅ Corrected filename

        if not model_path.exists():
            print("⬇️ Downloading NudeNet model...")

            # ✅ Initialize NudeNet, which downloads the model if missing
            detector = NudeDetector()

            # ✅ NudeNet's default model location (Windows/Linux/macOS)
            default_model_path = os.path.expanduser("~/.NudeNet/classifier.onnx")

            # ✅ Move the downloaded model to the correct directory
            if os.path.exists(default_model_path):
                shutil.move(default_model_path, model_path)
                print(f"✅ NudeNet model saved to {model_path}")
            else:
                print("⚠️ NudeNet model was not found in the expected location!")

        else:
            print("✅ NudeNet model already exists.")





    def download_deepface(self):
        """Moves DeepFace models if found, otherwise downloads them."""
        deepface_dir = Path(self.model_paths.get("deepface", "repositories/deepface/"))
        self.ensure_directory(deepface_dir)

        deepface_cache = Path.home() / ".deepface" / "weights"

        # Full list of DeepFace models, including Age, Gender, Emotion
        models = {
            "VGG-Face": "vgg_face_weights.h5",
            "Facenet": "facenet_weights.h5",
            "Facenet512": "facenet512_weights.h5",
            "ArcFace": "arcface_weights.h5",
            "Dlib": "dlib_face_recognition_resnet_model_v1.dat",
            "SFace": "face_recognition_sface_2021dec.onnx",
            "Age": "age_model_weights.h5",
            "Gender": "gender_model_weights.h5",
            "Emotion": "facial_expression_model_weights.h5",
            "DeepFace": "VGGFace2_DeepFace_weights_val-0.9034.h5"
        }

        for model_name, weight_file in models.items():
            model_path = deepface_dir / weight_file

            # Skip if model already exists
            if model_path.exists():
                print(f"✅ DeepFace model {model_name} already exists.")
                continue

            # Check if model is already in DeepFace cache
            downloaded_model_path = deepface_cache / weight_file

            if downloaded_model_path.exists():
                print(f"⬇️ Moving DeepFace model: {model_name} from cache...")
                shutil.move(str(downloaded_model_path), str(model_path))
                print(f"✅ {model_name} model moved to {model_path}")
                continue

            # If the model isn't found, trigger DeepFace download
            print(f"⬇️ Downloading DeepFace model: {model_name}...")

            try:
                DeepFace.build_model(model_name)

                # Verify if it downloaded to cache
                if downloaded_model_path.exists():
                    shutil.move(str(downloaded_model_path), str(model_path))
                    print(f"✅ {model_name} model saved to {model_path}")
                else:
                    print(f"⚠️ Model {model_name} did not download correctly.")
            except Exception as e:
                print(f"❌ Error downloading {model_name}: {e}")

    def download_spacy_model(self):
        """Downloads and verifies the `en_core_web_md` spaCy model."""
        install_spacy=SpacyInstall()
        try:
            spacy.load("en_core_web_md")  # Check if it's already installed
            print("✅ `en_core_web_md` model is already installed.")
        except OSError:
            print("⬇️ Downloading `en_core_web_md` spaCy model...")
            install_spacy.setup_spacy_models()       
            print("\n✅ All spaCy models are set up!")




    def download_deepdanbooru(self, MODEL_URL=None):
        """Downloads and extracts the DeepDanbooru model to the repositories folder."""
        
        MODEL_FOLDER = "repositories/deepdanbooru"
        MODEL_ZIP = os.path.join(MODEL_FOLDER, "deepdanbooru.zip")
        MODEL_URL=MODEL_URL if MODEL_URL is not None else "https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/deepdanbooru-v3-20211112-sgd-e28.zip"
        
        REQUIRED_FILES = [
            "model-resnet_custom_v3.h5",
            "project.json",
            "tags.txt",
            "tags-character.txt",
            "tags-general.txt",
            "categories.json",
            "projects.json",
            "tags-log.json"
            
        ]
        os.makedirs(MODEL_FOLDER, exist_ok=True)

        # Check if all required files exist
        missing_files = [file for file in REQUIRED_FILES if not os.path.exists(os.path.join(MODEL_FOLDER, file))]
        
        if missing_files:
            print("✅ DeepDanbooru model already exists. Skipping download.")
            return  # Exit the function early if files exist
        else:
            print(f"📥 Downloading DeepDanbooru model from {MODEL_URL}...")
            
            # Download the model
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open(MODEL_ZIP, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        file.write(chunk)
                print("✅ Download complete!")

                # Extract the zip file
                print("📂 Extracting model...")
                with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
                    zip_ref.extractall(MODEL_FOLDER)

                # Cleanup: Remove ZIP after extraction
                os.remove(MODEL_ZIP)
                print("✅ Model extracted successfully!")
            else:
                print(f"❌ Failed to download model. HTTP Status: {response.status_code}")



    # ---------------- MODEL DOWNLOAD FUNCTIONS ----------------

    def download_dino(self):
        """Downloads the DINO model (Self-Supervised Vision Transformer)."""
        dino_dir = Path("repositories/dino/")
        dino_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = dino_dir / "dino_vit_base.pth"
        if not model_path.exists():
            print("⬇️ Downloading DINO model...")
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
            torch.save(model.state_dict(), model_path)
            print("✅ DINO model saved to repositories/dino/")
        else:
            print("✅ DINO model already exists.")


    def download_efficientnet(self):
        """Downloads the EfficientNet model."""
        efficientnet_dir = Path("repositories/efficientnet/")
        efficientnet_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = efficientnet_dir / "efficientnet_b0.pth"
        if not model_path.exists():
            print("⬇️ Downloading EfficientNet model...")
            model = models.efficientnet_b0(pretrained=True)
            torch.save(model.state_dict(), model_path)
            print("✅ EfficientNet model saved to repositories/efficientnet/")
        else:
            print("✅ EfficientNet model already exists.")


    def download_mobilenet(self):
        """Downloads the MobileNet model."""
        mobilenet_dir = Path("repositories/mobilenet/")
        mobilenet_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = mobilenet_dir / "mobilenet_v3.pth"
        if not model_path.exists():
            print("⬇️ Downloading MobileNet model...")
            model = models.mobilenet_v3_large(pretrained=True)
            torch.save(model.state_dict(), model_path)
            print("✅ MobileNet model saved to repositories/mobilenet/")
        else:
            print("✅ MobileNet model already exists.")


    def download_swin_transformer(self):
        """Downloads the Swin Transformer model."""
        swin_dir = Path("repositories/swin_transformer/")
        swin_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = swin_dir / "swin_base_patch4_window7_224.pth"
        if not model_path.exists():
            print("⬇️ Downloading Swin Transformer model...")
            model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
            torch.save(model.state_dict(), model_path)
            print("✅ Swin Transformer model saved to repositories/swin_transformer/")
        else:
            print("✅ Swin Transformer model already exists.")


    def download_ai_or_not(self):
        """Downloads AI-Or-Not model (Nahrawy/AIorNot) for AI-generated image detection."""
        model_dir = Path("repositories/ai_or_not/")
        model_dir.mkdir(parents=True, exist_ok=True)

        model_url = "https://huggingface.co/Nahrawy/AIorNot/resolve/main/pytorch_model.bin"
        config_url = "https://huggingface.co/Nahrawy/AIorNot/resolve/main/config.json"
        preprocessor_url = "https://huggingface.co/Nahrawy/AIorNot/resolve/main/preprocessor_config.json"

        model_path = model_dir / "pytorch_model.bin"
        config_path = model_dir / "config.json"
        preprocessor_path = model_dir / "preprocessor_config.json"

        def download_file(url, path):
            """Helper function to download files only if missing."""
            if not path.exists():
                print(f"⬇️ Downloading {path.name}...")
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(path, "wb") as file:
                        for chunk in response.iter_content(chunk_size=1024):
                            file.write(chunk)
                    print(f"✅ {path.name} downloaded successfully!")
                else:
                    print(f"❌ Failed to download {path.name}. HTTP Status: {response.status_code}")
            else:
                print(f"✅ {path.name} already exists. Skipping download.")

        # Download model files
        download_file(model_url, model_path)
        download_file(config_url, config_path)
        download_file(preprocessor_url, preprocessor_path)

        print("✅ AI-Or-Not model setup complete!")



    def download_deepfake_detector(self):
        """Downloads XceptionNet model for DeepFake detection."""
        model_dir = Path("repositories/deepfake_detector/")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "xception_model.h5"
        if not model_path.exists():
            print("⬇️ Downloading DeepFake Detection model...")
            model = Xception(weights='imagenet', include_top=True)
            model.save(model_path)
            print("✅ DeepFake Detector model saved to repositories/deepfake_detector/")
        else:
            print("✅ DeepFake Detector model already exists.")


    '''
    on Hold
    def download_texture_analysis(self): #
        """Downloads the TID2013 dataset for texture consistency analysis."""
        model_dir = Path("repositories/texture_analysis/")
        model_dir.mkdir(parents=True, exist_ok=True)

        dataset_url = "https://www.ponomarenko.info/tid2013/tid2013.rar"
        zip_path = model_dir / "tid2013.rar"
        extract_path = model_dir / "TID2013"

        if extract_path.exists():
            print("✅ TID2013 dataset already exists. Skipping download.")
            return
        else:
            print("⬇️ Downloading TID2013 dataset for texture analysis...")

            response = requests.get(dataset_url, stream=True)

            # Check if the download was successful
            if response.status_code != 200:
                print(f"❌ Failed to download dataset. HTTP Status: {response.status_code}")
                return

            # Save the file
            with open(zip_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)

            print("✅ Download complete! Extracting dataset...")

            # Check if it's a valid ZIP file
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(model_dir)
                os.remove(zip_path)  # Remove ZIP after extraction
                print("✅ TID2013 dataset extracted successfully!")
            except zipfile.BadZipFile:
                print("❌ Error: Downloaded file is not a valid ZIP. Possible corrupted download.")
                os.remove(zip_path)  # Delete invalid file

    '''

    def download_image_forgery_detector(self):
        """Downloads the Image Forgery Detection model from GitHub."""
        model_dir = Path("repositories/image_forgery/")
        model_dir.mkdir(parents=True, exist_ok=True)

        repo_url = "https://github.com/ShivamGupta92/ImageForgeryDetection.git"
        repo_path = model_dir / "ImageForgeryDetection"

        if repo_path.exists():
            print("✅ Image Forgery Detection repository already exists. Skipping download.")
            return

        print("⬇️ Cloning Image Forgery Detection repository...")

        try:
            subprocess.run(["git", "clone", repo_url, str(repo_path)], check=True)
            print("✅ Image Forgery Detection repository cloned successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to clone repository: {e}")



# ---------------- MAIN EXECUTION ----------------
if __name__ == "__main__":
    print("\n🔄 Checking and downloading missing models...\n")
    download_models = DownloadModels()

    # Define each step with descriptions
    steps = [
        ("Downloading BLIP model (for vision-language understanding)", download_models.download_blip),
        ("Downloading CLIP model (for image-text similarity analysis)", download_models.download_clip),
        ("Downloading YOLO model (for object detection)", download_models.download_yolo),
        ("Downloading NudeNet model (for NSFW image classification)", download_models.download_nudenet),
        ("Downloading DeepFace model (for facial recognition & analysis)", download_models.download_deepface),
        ("Downloading spaCy model (for NLP-based image descriptions)", download_models.download_spacy_model),
        ("Downloading DeepDanbooru model (for anime-style image tagging)", download_models.download_deepdanbooru),
        ("Downloading DINO model (for self-supervised vision tasks)", download_models.download_dino),
        ("Downloading EfficientNet model (for general image classification)", download_models.download_efficientnet),
        ("Downloading MobileNet model (for lightweight image classification)", download_models.download_mobilenet),
        ("Downloading Swin Transformer model (for hierarchical vision analysis)", download_models.download_swin_transformer),
        ("Downloading AI-Or-Not model (for detecting AI-generated images)", download_models.download_ai_or_not),
        ("Downloading DeepFake Detector model (for identifying manipulated faces)", download_models.download_deepfake_detector),
        #("Downloading Texture Analysis dataset (for detecting image inconsistencies)", download_models.download_texture_analysis), #on hold
        ("Downloading Image Forgery Detector model (for identifying digitally altered images)", download_models.download_image_forgery_detector)
    ]


    with tqdm(total=len(steps), desc="Overall Progress", unit="step") as pbar:
        for step_name, step_function in steps:
            print(f"➡️ {step_name}...")
            step_function()
            pbar.update(1)  # ✅ Update progress bar after each step
    print("\n✅ All models are set up!")
