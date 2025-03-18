import os
import json
import yaml
from load_config import load_config
from categorizing_images import ImageCategorizer
from ddb_index_updater import DeepDanbooruUpdater
from thumbnails import ThumbnailCreator
from tqdm import tqdm
from load_deepdanbooru_model import DeepDanbooruModel
from gallery_rename import GalleryMaker
from config_manager import ConfigManager

class RunCat(ConfigManager):
    
        
    def run_main_program(self):   
       
        
        """Executes the main image classification pipeline."""
        print("\n🚀 Running Image Categorization...")   
        categorizer = ImageCategorizer()  
        ddb_index_updater = DeepDanbooruUpdater()   
        gallery_maker = GalleryMaker()
        ddb_model = DeepDanbooruModel(self.dd_model_folder)
        thumbnail_creator = ThumbnailCreator()
        

        steps = [
        ("Combining categories into one file", categorizer.merge_category_files),  #Necessary to categorize using custom categories
        ("Adding Categories for all images", categorizer.process_all_images),
        ("Finding DeepDanbooru Tags", categorizer.index_loader.classify_images),
        ("Unloading DeepDanBooru Model", ddb_model.unload_model), #Unloads DDB Model from memory after classify images completes
        ("Updating DeepDanbooru Tags", ddb_index_updater.update_image_index), #Updates the image_index.json with "DDB/DDB_index.json"
        ("Generating Missing Thumbnails", lambda: thumbnail_creator.generate_thumbnails(xy=512)),
        ("Organizing Galleries", gallery_maker.organize_new_gallery)
        ]

       
        with tqdm(total=len(steps), desc="Overall Progress", unit="step") as pbar:
            for step_name, step_function in steps:
                print(f"➡️ {step_name}...")
                step_function()
                pbar.update(1)  # ✅ Update progress bar after each step

         


if __name__ == "__main__":
    runcat = RunCat()
    runcat.run_main_program()
    # Example of saving merged lists per category
    # merge_category_files(save_combined_for="human")
    # merge_category_files(save_combined_for="animal")
