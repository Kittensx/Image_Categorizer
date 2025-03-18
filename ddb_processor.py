import os
import json
import numpy as np
from PIL import Image
from load_deepdanbooru_model import DeepDanbooruModel
import logging
from config_manager import ConfigManager

class DeepDanbooruProcessor(ConfigManager):
    def __init__(self):  
        super().__init__()
        self.dd_instance = DeepDanbooruModel(self.dd_model_folder)
        #dd_instance.model
        #dd_instance.model_path
        
        self.labels = self.load_labels(self.custom_tags_folder) 

    def load_labels(self, custom_folder="custom_tags"):
        """Loads DeepDanbooru labels from multiple tag files, including custom labels."""
        
        label_files = [
            os.path.join(self.dd_instance.model_path, "tags.txt"),
            os.path.join(self.dd_instance.model_path, "tags-character.txt"),
            os.path.join(self.dd_instance.model_path, "tags-general.txt")
        ]

        # ✅ Load default DeepDanbooru labels
        # Allows custom txt files to be added and loaded alongside dd tags. 
        labels = []
        for label_file in label_files:
            if os.path.exists(label_file):
                with open(label_file, "r", encoding="utf-8") as f:
                    labels.extend(line.strip() for line in f.readlines())
                #print(f"✅ Loaded {len(labels)} labels from {label_file}")
            else:
                logging.warning(f"⚠️ Warning: {label_file} not found. Skipping.")

        # ✅ Check for additional custom `.txt` files
        if os.path.exists(custom_folder):
            for custom_file in os.listdir(custom_folder):
                if custom_file.endswith(".txt"):
                    custom_path = os.path.join(custom_folder, custom_file)
                    with open(custom_path, "r", encoding="utf-8") as f:
                        custom_labels = [line.strip() for line in f.readlines()]
                        labels.extend(custom_labels)
                    #print(f"✅ Loaded {len(custom_labels)} custom labels from {custom_file}")

        # ✅ Ensure labels are unique and properly indexed
        labels = list(dict.fromkeys(labels))  # Remove duplicates while preserving order

        if not labels:
            raise ValueError("❌ Error: No valid labels were loaded.")

        return labels


    def preprocess_image(self, image_path):
        """Loads an image, resizes it to 512x512 for DeepDanbooru, and ensures correct shape."""
        resized_folder = os.path.join(os.path.dirname(image_path), "resized_512")
        os.makedirs(resized_folder, exist_ok=True)

        resized_image_path = os.path.join(resized_folder, os.path.basename(image_path))

        # ✅ Check if resized image already exists
        if os.path.exists(resized_image_path):
            #print(f"✅ Resized image already exists: {resized_image_path}")
            img = Image.open(resized_image_path)
        else:
            # ✅ Resize only if needed
            img = Image.open(image_path).convert("RGB")
            img = img.resize((512, 512))
            img.save(resized_image_path)
            #print(f"✅ Resized and saved: {resized_image_path}")

        # ✅ Convert image to NumPy array and ensure correct shape
        img_array = np.array(img) / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # ✅ Add batch dimension (1, 512, 512, 3)

        #print(f"✅ Image shape before passing to DeepDanbooru: {img_array.shape}")
        return img_array, resized_image_path




    def classify_image(self, image_path):
        """Runs DeepDanbooru classification and stores results in a JSON file."""
        if not os.path.exists(image_path):
            #print(f"❌ Image not found: {image_path}")
            return None

        # ✅ Resize image and get saved path
        img_array, resized_image_path = self.preprocess_image(image_path)
        
        # ✅ Run DeepDanbooru classification
        raw_scores = dd_instance.model.predict(img_array)
        
        dd_instance.unload_model() #unload model when done 
        

        # ✅ Ensure `raw_scores` is a flat list, not a nested list
        if isinstance(raw_scores, np.ndarray):
            raw_scores = raw_scores.flatten().tolist()  # ✅ Ensure it's a flat list

        # ✅ Convert DeepDanbooru predictions to {tag: confidence}
        tag_confidence_map = {
            self.labels[i]: round(float(score), 4)  # ✅ Ensure `score` is a float before rounding
            for i, score in enumerate(raw_scores)
            if i < len(self.labels) and isinstance(score, (float, int))  # ✅ Ignore invalid types
        }

        if not tag_confidence_map:
            tag_confidence_map = {"Unclassified": 0.0}

        # ✅ Save results to JSON      
        #ddb_json_folder = os.path.join(os.path.dirname(image_path), "DDB") 
        os.makedirs(ddb_json_folder, exist_ok=True)        
        ddb_json_path = os.path.join(ddb_json_folder, os.path.basename(image_path).split(".")[0] + ".json")
        

        with open(ddb_json_path, "w", encoding="utf-8") as f:
            json.dump(tag_confidence_map, f, indent=4)

        # ✅ Save reference to `ddb_index.json`
        self.update_ddb_index(image_path, resized_image_path, ddb_json_path)

        #print(f"✅ DeepDanbooru JSON results saved: {ddb_json_path}")
        return ddb_json_path, tag_confidence_map
 
    def update_ddb_index(self, image_path, resized_image_path, ddb_json_path):
        """Stores reference paths for each image in `ddb_index.json`."""
        ddb_index_file = os.path.join(self.ddb_folder, "ddb_index.json")

        # ✅ Load existing data
        if os.path.exists(ddb_index_file):
            with open(ddb_index_file, "r", encoding="utf-8") as f:
                try:
                    ddb_index = json.load(f)
                except json.JSONDecodeError:
                    ddb_index = {}
        else:
            ddb_index = {}

        # ✅ Add/Update entry
        ddb_index[os.path.basename(image_path)] = {
            "original_image": image_path,
            "resized_512": resized_image_path,
            "deepdanbooru_json": ddb_json_path
        }

        # ✅ Save updated index
        with open(ddb_index_file, "w", encoding="utf-8") as f:
            json.dump(ddb_index, f, indent=4)

        #print(f"✅ Updated DeepDanbooru index for {image_path}")
    
    def process_deepdanbooru_results(self, tag_confidence_map, confidence_threshold=0.25, max_tags=40):
        """
        Filters top confidence tags from DeepDanbooru results and returns a cleaned dictionary.
        """
        if not tag_confidence_map:
            return {}

        # ✅ Filter out low-confidence scores
        filtered_tags = {
            tag: confidence for tag, confidence in tag_confidence_map.items() if confidence >= confidence_threshold
        }

        # ✅ Sort by confidence and limit the number of tags
        sorted_tags = dict(sorted(filtered_tags.items(), key=lambda item: item[1], reverse=True)[:max_tags])

        return sorted_tags  # ✅ Return top tags without needing to reload JSON


        
    def process_and_classify_image(self, image_path, confidence_threshold=0.25, max_tags=40):
        """
        Resizes the image, runs DeepDanbooru classification, saves results,
        extracts top confidence tags, and returns them.
        """
        ddb_json_folder = os.path.join(os.path.dirname(image_path), "DDB") 
        os.makedirs(ddb_json_folder, exist_ok=True)    
        ddb_json_path = os.path.join(ddb_json_folder, os.path.basename(image_path).split(".")[0] + ".json")
        
        if not os.path.exists(image_path):                      
            #print(f"❌ Image not found: {image_path}")
            return None       

        # ✅ Step 1: Resize image and get saved path
        image_tensor, resized_image_path = self.preprocess_image(image_path)
        

         # ✅ Step 2: If JSON already exists, load it
        if os.path.exists(ddb_json_path):
            #print(f"✅ Using existing DeepDanbooru JSON: {ddb_json_path}")
            with open(ddb_json_path, "r", encoding="utf-8") as f:
                tag_confidence_map = json.load(f)  # ✅ Load the existing JSON file
        else:
            # ✅ Run DeepDanbooru classification if JSON doesn't exist
            ddb_json_path, tag_confidence_map = self.classify_image(image_path)

        if not tag_confidence_map:
            #print(f"⚠️ No valid DeepDanbooru results for {image_path}")
            return None         
        
        #Step 3: Filter Results
        sorted_tags = self.process_deepdanbooru_results(tag_confidence_map, confidence_threshold, max_tags)
        
        # ✅ Step 4: Save reference to `ddb_index.json`
        self.update_ddb_index(image_path, resized_image_path, ddb_json_path)

        #print(f"✅ Processed DeepDanbooru tags for {image_path}: {sorted_tags}")
        
        return {
            "ddb_json": ddb_json_path,
            "top_tags": sorted_tags  # ✅ Return top tags directly for `index_loader.py`
        }
