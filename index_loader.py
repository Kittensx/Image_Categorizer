import yaml
import json
import os
from PIL import Image
from load_config import load_config
import deepdanbooru 
import numpy as np
from ddb_processor import DeepDanbooruProcessor 
import logging
from config_manager import ConfigManager

class IndexLoader(ConfigManager):
    def __init__(self, image_folder=None, index_file=None):
        super().__init__()        
        self.index_data = self.load_existing_index()  # Load existing index on startup
        
        # ✅ Initialize DeepDanbooru Processor
        self.deepdanbooru = DeepDanbooruProcessor()
        
    def load_existing_index(self):
        """Loads or creates an image index file."""
        if os.path.exists(self.image_index_file):
            with open(self.image_index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}  # Return empty dict if no index exists
        
    def load_image_index(self, index_file=None):
        """Loads the image index file if it exists."""
        if os.path.exists(self.image_index_file):
            with open(self.image_index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def generate_thumbnails(self):
        """Creates missing thumbnails for images in the index."""
        os.makedirs(self.thumbnails_folder, exist_ok=True)  # ✅ Ensure folder exists

        for image_name, data in self.index_data.items():
            image_path = data["image_path"]
            thumbnail_path = os.path.join(self.thumbnails_folder, image_name)

            # ✅ Skip if the thumbnail already exists
            if os.path.exists(thumbnail_path):
                #print(f"✅ Thumbnail already exists: {thumbnail_path}")
                continue

            # ✅ Generate thumbnail
            try:
                img = Image.open(image_path)
                img.thumbnail((100, 100))  # Resize to thumbnail size
                img.save(thumbnail_path)
                #print(f"✅ Created thumbnail: {thumbnail_path}")
            except Exception as e:
                logging.error(f"❌ Error creating thumbnail for {image_name}: {e}")


    def save_index(self):
        """Safely saves the updated image index, ensuring all previous entries remain."""
        temp_file = self.image_index_file + ".dat"  # ✅ Temporary file to prevent corruption

        try:
            # ✅ Load existing index before saving
            if os.path.exists(self.image_index_file):
                with open(self.image_index_file, "r", encoding="utf-8") as f:
                    try:
                        self.index_data = json.load(f)
                    except json.JSONDecodeError:
                        print("⚠️ Warning: `image_index.json` is corrupted. Creating a new one.")
                        self.index_data = {}

            # ✅ Step 1: Write merged data to a temporary file
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.index_data, f, indent=4)

            # ✅ Step 2: Replace the original JSON file with the updated one
            os.replace(temp_file, self.image_index_file)  # Atomic operation

            #print(f"✅ Index safely updated: {self.image_index_file}")

        except Exception as e:
            logging.error(f"❌ Error saving index: {e}")

    def scan_and_log_images(self):
        """Scans the folder for new images, runs DeepDanbooru only if necessary, and updates the index."""
        if os.path.exists(self.image_index_file):
            with open(self.image_index_file, "r", encoding="utf-8") as f:
                try:
                    self.index_data = json.load(f)
                except json.JSONDecodeError:
                    #print("⚠️ Warning: `image_index.json` is corrupted. Creating a new one.")
                    self.index_data = {}
        self.generate_thumbnails()
        for filename in os.listdir(self.image_folder):
            if filename.lower().endswith(("png", "jpg", "jpeg")):
                image_path = os.path.join(self.image_folder, filename)
                

                # ✅ Ensure image entry exists before modification
                if filename not in self.index_data:
                    self.index_data[filename] = {
                        "image_path": image_path,
                        "categories": "Not classified yet",
                        "DeepDanbooru": {},  # ✅ Store top tags here
                        "DeepDanbooru_json": None  # ✅ Store path to JSON file
                    }

                # ✅ Skip processing if categories already exist
                if self.index_data[filename]["categories"] != "Not classified yet":
                    #print(f"⚠️ Skipping {filename} (already classified)")
                    continue

                # ✅ Run DeepDanbooru classification only if missing
                if not self.index_data[filename]["DeepDanbooru_json"]:
                    #print(f"🔍 Running DeepDanbooru classification for {filename}...")
                    ddb_results = self.deepdanbooru.process_and_classify_image(image_path, confidence_threshold=0.35, max_tags=50)

                    if ddb_results:
                        self.index_data[filename]["DeepDanbooru"] = ddb_results["top_tags"]  # ✅ Store top tags
                        self.index_data[filename]["DeepDanbooru_json"] = ddb_results["ddb_json"]  # ✅ Store JSON file path
        
        self.save_index()  # ✅ Save the updated index


    def classify_images(self):
        """Runs `categorize_image()` on unclassified images, updates DeepDanbooru results if missing."""
        if os.path.exists(self.image_index_file):
            with open(self.image_index_file, "r", encoding="utf-8") as f:
                try:
                    self.index_data = json.load(f)
                except json.JSONDecodeError:
                    #print("⚠️ Warning: `image_index.json` is corrupted. Creating a new one.")
                    self.index_data = {}

        for filename, data in self.index_data.items():
            image_path = data["image_path"]

            # ✅ Ensure "DeepDanbooru_json" exists instead of "DeepDanbooru_txt"
            if "DeepDanbooru_json" not in data:
                self.index_data[filename]["DeepDanbooru_json"] = None
            if "DeepDanbooru" not in data:
                self.index_data[filename]["DeepDanbooru"] = {}

            # ✅ Skip processing if categories already exist
            if data.get("categories", "Not classified yet") != "Not classified yet":
                #print(f"⚠️ Skipping {filename} (already classified)")
                continue

            # ✅ If categories are not classified yet, run categorization
            #print(f"🔍 Classifying {filename} with ImageNet model...")
            categories = self.categorize_image(image_path)
            self.index_data[filename]["categories"] = categories

            # ✅ If DeepDanbooru results are missing, re-run DeepDanbooru
            if not self.index_data[filename]["DeepDanbooru_json"]:
                #print(f"🔍 Running DeepDanbooru classification for {filename}...")
                ddb_results = self.deepdanbooru.process_and_classify_image(image_path, confidence_threshold=0.05, max_tags=10)

                if ddb_results:
                    self.index_data[filename]["DeepDanbooru"] = ddb_results["top_tags"]
                    self.index_data[filename]["DeepDanbooru_json"] = ddb_results["ddb_json"]
        
        self.save_index()  # ✅ Save updated index
        #print("✅ All images classified!")
