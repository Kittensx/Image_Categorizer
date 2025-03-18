import os
from load_config import load_config

class ConfigManager:
    """Holds all common paths and settings to be inherited by other classes."""
    
    def __init__(self):
        
        
        # ✅ Load main config file (if applicable)       
        self.root_path = os.path.dirname(os.path.abspath(__file__))
        self.program_path = os.getcwd()
        
        # User config directory
        self.user_config_path = os.path.abspath(os.path.join(self.root_path, "user_config"))   
        self.config = load_config(os.path.abspath(os.path.join(self.user_config_path, "config.yaml")))
        self.repo_path = os.path.abspath(os.path.join(self.root_path, "repositories"))
        
        
        #self.repo_folder= self.config.get("repositories", "repositories").lower() #folder creation should occur during donwload_models.py
        self.dd_model_folder = os.path.abspath(os.path.join(self.repo_path, "deepdanbooru")) #folder creation should occur during donwload_models.py
        self.spacy_model_folder = os.path.abspath(os.path.join(self.repo_path, "Spacy")) #folder creation should occur during download_models.py
        self.base_image_folder = self.config.get("base_image_folder", "images").lower() #redundant name
        self.image_folder = self.config.get("image_folder", "images").lower() #this is the place where we scan for categories / image scanning
        
        
        self.custom_tags_folder = os.path.abspath(os.path.join(self.user_config_path, "custom_tags"))
        
           
        self.gallery_folder = self.config.get("gallery_location", "static").lower()        
        self.gallery_path = os.path.join(self.root_path, self.gallery_folder)
        
       
             
        
        # Common file paths
        self.ddb_index_file = os.path.abspath(os.path.join(self.gallery_path, "ddb_index.json"))
        self.ddb_folder = os.path.abspath(os.path.join(self.gallery_path, "DDB"))
        self.image_index_file = os.path.abspath(os.path.join(self.gallery_path, "image_index.json"))
        self.thumbnails_folder = os.path.abspath(os.path.join(self.gallery_path, "Thumbnails"))
        self.ddb_json_folder = self.ddb_folder        
        #ddb_json_folder = os.path.join(os.path.dirname(image_path), "DDB") originally line 116 in ddb_processor
        
        #Select Preferred Models
        self.spacy_model = self.config.get("spacy_model", "en_core_web_md")
        
        
        
        #Move the folder creation to program init once the  program is complete
        os.makedirs(self.gallery_folder, exist_ok=True)
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.ddb_folder, exist_ok=True)
        os.makedirs(self.ddb_json_folder, exist_ok=True)
        os.makedirs(self.user_config_path, exist_ok=True)        
        os.makedirs(self.custom_tags_folder, exist_ok=True)
        os.makedirs(self.thumbnails_folder, exist_ok=True)
        #os.makedirs(self.working_folder, exist_ok=True)
       
        