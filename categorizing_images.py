import json
import yaml
import os
import re
import torch
import shutil
from torchvision import models, transforms
from PIL import Image
import spacy
from load_config import load_config
from load_master_lists import load_master_lists
from pathlib import Path
from index_loader import IndexLoader
import logging
from load_resnet50 import ResNet50Model
from config_manager import ConfigManager
from load_spacy_model import SpacyModel


class ImageCategorizer(ConfigManager):
    def __init__(self):
        super().__init__()
        
        # Step 1: Merge categories before processing images  
        self.categories = load_master_lists()   # Load category mappings by combining master lists in "categories/_master"
        
        self.nlp = SpacyModel()
               
        resnet_instance = ResNet50Model()
        self.model = resnet_instance.model
        self.labels = resnet_instance.labels
       
        self.index_loader = IndexLoader()
        
        '''
        ensure these folders are replaced with self.thumbnails_folder //3-12-2025
        '''
        
        
        # Scan images, copy them to working folder, and convert if needed
        self.new_images_folder = self.get_all_images()
        self.all_images = self.copy_images_to_working_folder(self.new_images_folder)
        self.converted_images = self.convert_rgba_to_rgb(self.all_images)
        
        #self.category_keywords = self.get_categories(categories_folder="categories")  # Load category mappings / Function not in use?
    
    
    def merge_category_files(self, save_combined_for=None):
        """Scans the `categories/` folder, merges new files into the correct structure, and updates the in-memory list.

        Parameters:
        - save_combined_for (str): If provided, saves a `_combinedlist.json` for this category.
        """
        
        CATEGORIES_FOLDER = "categories/"
        MASTER_LIST_FILE = "master_list.json"
        BASE_IMAGE_FOLDER = self.config.get("base_image_folder", "images")
        #print("🔄 Merging category files...")

        # Load existing master categories
        master_categories = load_master_lists()  # Now loads all master categories

        categorizer = ImageCategorizer()  # Use ImageCategorizer to handle category merging

        for filename in os.listdir(CATEGORIES_FOLDER):
            if filename.endswith((".json", ".yaml", ".yml")):
                category_name = categorizer.extract_top_category(filename, master_categories.keys())

                if category_name:
                    file_path = os.path.join(CATEGORIES_FOLDER, filename)

                    with open(file_path, "r", encoding="utf-8") as f:
                        new_data = json.load(f) if filename.endswith(".json") else yaml.safe_load(f)

                    #print(f"📂 Merging: {filename} into {category_name}")
                    categorizer.merge_into_correct_level(new_data, master_categories, category_name)

        # Save the fully merged master list
        with open(MASTER_LIST_FILE, "w", encoding="utf-8") as f:
            json.dump(master_categories, f, indent=4)
        print(f"✅ Master category list updated in `{MASTER_LIST_FILE}`!")

    def convert_rgba_to_rgb(self, image_folder=None):
        """
        Converts images from RGBA to RGB format and saves them to a new folder.

        :param image_paths: List of image file paths.
        :param destination_folder: Folder where converted images will be saved.
        """       
        image_folder=image_folder if image_folder is not None else self.image_folder
        converted_images = []
        for image_path in image_folder:
            try:
                with Image.open(image_path) as img:
                    if img.mode == "RGBA":
                        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                        rgb_img.paste(img, mask=img.split()[3])
                        save_path = os.path.join(self.thumbnails_folder, os.path.basename(image_path))
                        rgb_img.save(save_path, format="PNG")
                        converted_images.append(save_path)
                    else:
                        converted_images.append(image_path)
                    self.index_loader.scan_and_log_images() # Step 1: Log images
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        return converted_images  # Returns only RGB-converted images

   
    
    def copy_images_to_working_folder(self, image_folder=None): 
        """
        Copies all images from the given list of paths to the destination folder.
        
        :param image_paths: List of image file paths.
        :param destination_folder: Folder where images will be copied.
        """
        image_folder=image_folder if image_folder is not None else self.image_folder
        # Ensure destination folder exists
        copied_images = []
        for image_path in image_folder:
            if os.path.isfile(image_path):
                destination_path = os.path.abspath(os.path.join(self.thumbnails_folder, os.path.basename(image_path)))

                # ✅ Skip copying if the file already exists
                if os.path.exists(destination_path):
                    #print(f"🔄 Skipping copy (already exists): {destination_path}")
                    copied_images.append(destination_path)
                    continue

                try:
                    shutil.copy(image_path, destination_path)
                    copied_images.append(destination_path)
                    #print(f"✅ Copied: {image_path} -> {destination_path}")
                except Exception as e:
                    print(f"❌ Error copying {image_path}: {e}")

        return copied_images  # Returns list of files in the working directory




    def get_all_images(self, image_folder=None):
        """Recursively scans for all image files in the base folder and subfolders."""
        image_folder = image_folder if image_folder is not None else self.image_folder
        image_extensions = {ext.lower() for ext in Image.registered_extensions().keys()}
        image_files = []

        for image_path in Path(self.image_folder).rglob("*.*"):  # Scan recursively
            if image_path.suffix.lower() in image_extensions:
                image_files.append(str(image_path))

        return image_files
    
    
    

    

    def get_categories(self, categories_folder="categories"): #not in use
        """Loads and merges default categories with custom categories from JSON/YAML."""
        categories = {}

        if os.path.exists(categories_folder):
            for filename in os.listdir(categories_folder):
                file_path = os.path.join(categories_folder, filename)
                if filename.endswith(".json"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.merge_dicts(data, categories)
                elif filename.endswith((".yaml", ".yml")):
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                        self.merge_dicts(data, categories)
        else:
            os.makedirs(categories_folder, exist_ok=True)

        return categories if categories else self.default_categories()

    def merge_into_correct_level(self, new_data, master_data, top_category):
        """
        Merges a user-defined file into the correct location in the master list.
        - If values match an existing subcategory, merge them properly.
        - If values should be a subcategory, move them under the correct key.
        """
        for key, value in new_data.items():
            if key in master_data.get(top_category, {}):  # If key exists in the master list
                if isinstance(value, list) and isinstance(master_data[top_category][key], list):
                    master_data[top_category][key] = list(set(master_data[top_category][key] + value))  # Merge lists
                elif isinstance(value, dict) and isinstance(master_data[top_category][key], dict):
                    self.merge_into_correct_level(value, master_data[top_category][key], top_category)  # Recursively merge
            else:
                # If the key is a new top-level but should be deeper, find correct placement
                correct_location = self.find_deepest_match(key, master_data[top_category])
                if correct_location:
                    master_data[top_category][correct_location][key] = value  # Place in correct subcategory
                else:
                    master_data[top_category][key] = value  # Add as a new key
                    
    def find_deepest_match(self, user_key, category_structure, current_path=""):
        """
        Recursively finds the deepest matching subcategory for a given key.
        Ensures the key is placed in the most specific category available.
        """
        deepest_match = None
        current_depth = 0  # Track depth level

        for subcategory, sub_values in category_structure.items():
            new_path = f"{current_path} > {subcategory}" if current_path else subcategory

            if user_key == subcategory:  # Direct match at this level
                return new_path  # Return full hierarchical path

            if isinstance(sub_values, dict):
                deeper_match = self.find_deepest_match(user_key, sub_values, new_path)
                if deeper_match:
                    # Update only if this new match is deeper
                    match_depth = deeper_match.count(" > ")
                    if match_depth > current_depth:
                        deepest_match = deeper_match
                        current_depth = match_depth

        return deepest_match

    

    def save_combined_list(self, merged_data, category): #module not in use
        """Saves the merged data into a new file without modifying the master list."""
        output_file = f"{category}_combinedlist.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, indent=4)
        #print(f"✅ Merged categories saved as: {output_file}")


    def extract_top_category(self, filename, master_categories):
        """Extracts the category name from a file using enforced naming rules."""
        match = re.match(r"([a-zA-Z]+)[_\-\s]?(\d+)", filename)  # Match category + number
        if match:
            category = match.group(1).lower()  # Extract category name
            if category in master_categories:  # Ensure it's a valid top-level category
                return category
        return None  # Invalid file name or category not in master list

    def merge_dicts(self, source, destination):
        """✅ REPLACEMENT: Recursively merges nested dictionaries ensuring structured categories."""
        for key, value in source.items():
            if key in destination:
                if isinstance(destination[key], dict) and isinstance(value, dict):
                    self.merge_dicts(value, destination[key])
                elif isinstance(destination[key], list) and isinstance(value, dict):
                    destination[key] = {"general": destination[key]}
                    self.merge_dicts(value, destination[key])
                elif isinstance(destination[key], dict) and isinstance(value, list):
                    if "general" in destination[key]:
                        destination[key]["general"] = list(set(destination[key]["general"] + value))
                    else:
                        destination[key]["general"] = value
                elif isinstance(destination[key], list) and isinstance(value, list):
                    destination[key] = list(set(destination[key] + value))
                else:
                    destination[key] = value  # Overwrite conflicting values
            else:
                destination[key] = value  # Assign new key-value pair


    @staticmethod
    def default_categories(): #revisit this later to fill out the default more.
        """Returns default category mappings."""
        return {
            "human": {
                "adults": ["man", "woman"],
                "children": ["boy", "girl", "baby"]
            },
            "animal": {
                "mammals": ["dog", "cat", "elephant"],
                "birds": ["sparrow", "eagle"],
                "fish": ["goldfish", "shark"]
            },
            "landscape": {
                "natural": ["mountain", "beach", "forest"],
                "water": ["river", "lake", "ocean"]
            }
        }

    def categorize_image(self, image_path):
        """Categorizes an image based on the most relevant ImageNet class."""
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        image = transform(Image.open(image_path)).unsqueeze(0)

        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            category_idx = probabilities.argmax().item()
            confidence = probabilities[category_idx].item()

        predicted_label = self.labels[category_idx]

        # ✅ Ensure correct unpacking
        matched_categories, _ = self.category_matches(predicted_label)

        if not matched_categories:  # ✅ Handle cases where no matches were found
            print(f"\n⚠️ No category matches found for label: {predicted_label}")
            return [("Unknown", 0.0)]  # ✅ Return a default category instead of failing

        #print(f"\n🔍 Predicted Label: {predicted_label} (Index {category_idx}) with {confidence:.2%} confidence")
        #print(f"✅ Matching Categories:")

        #for category, score in matched_categories:
            #print(f"   - {category}: {score:.2%} similarity")

        return matched_categories


    def category_matches(self, label, top_category_threshold=0.5, subcategory_threshold=0.3):
        """Finds matching categories for an ImageNet label using NLP similarity."""
        label_doc = self.nlp(label.lower())

        if label_doc.vector_norm == 0:
            #print(f"⚠️ Skipping label '{label}' due to missing word vector.")
            return [], {}

        matched_categories = {}
        subcategory_confidence = {}
        
        #print(f"\n🔍 Matching for Label: {label}")

        for category, subcategories in self.categories.items():
            if isinstance(subcategories, list):
                for keyword in subcategories:
                    keyword_doc = self.nlp(keyword)
                    
                    if keyword_doc.vector_norm == 0:
                        #print(f"⚠️ Skipping keyword '{keyword}' due to missing word vector.")
                        continue  # Skip words without vectors
                    
                    similarity = label_doc.similarity(keyword_doc)
                    if similarity >= top_category_threshold:
                        matched_categories[category] = max(matched_categories.get(category, 0), similarity)

            elif isinstance(subcategories, dict):
                for subcategory, keywords in subcategories.items():
                    for keyword in keywords:
                        keyword_doc = self.nlp(keyword)
                        
                        if keyword_doc.vector_norm == 0:
                            #print(f"⚠️ Skipping keyword '{keyword}' due to missing word vector.")
                            continue  # Skip words without vectors
                        
                        similarity = label_doc.similarity(keyword_doc)
                        if similarity >= subcategory_threshold:
                            key = f"{category} > {subcategory}"
                            matched_categories[key] = max(matched_categories.get(key, 0), similarity)

        sorted_matches = sorted(matched_categories.items(), key=lambda x: x[1], reverse=True)

        return sorted_matches, subcategory_confidence



    def save_results_to_index(self, image_path, categories):
        """Saves image classification results to an index file."""
        image_name = os.path.basename(image_path)

        if os.path.exists(self.image_index_file):
            with open(self.image_index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)
        else:
            index_data = {}

        index_data[image_name] = {
            "image_path": image_path,
            "categories": categories
        }
       
        with open(self.image_index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=4)
        #print(f"✅ Classification saved for {image_name}.")
    
   
       
                
    def process_all_images(self):
        """Scans all images and runs categorization."""
        #image_files = self.get_all_images()  # ✅ Get all images recursively
        image_files = self.converted_images
        if not image_files:
            #print("⚠️ No images found in the specified directory.")
            return

        print(f"📂 Found {len(image_files)} images for processing.")
       
        for image_path in image_files:
            #print(f"🔍 Processing: {image_path}")
            
            #print(json.dumps(tags, indent=4))
            categories = self.categorize_image(image_path)
            # ✅ Run DeepDanbooru classification using IndexLoader
            self.index_loader.scan_and_log_images()            
            self.index_loader.classify_images()  # Step 2: Classify them - adds to index file. 
            
            # ✅ Save results to index
            self.save_results_to_index(image_path, categories)
            # ✅ Store or use categorized results as needed
            logging.info(f"✅ Categorized {image_path} as: {categories}")

        logging.info("🎯 All images processed successfully.")
   

'''
if __name__ == "__main__":
    categorizer = ImageCategorizer()
    categorizer.process_all_images()  # Automatically process all images
'''
