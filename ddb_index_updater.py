import json
import os
from config_manager import ConfigManager

class DeepDanbooruUpdater(ConfigManager):
    def __init__(self, confidence_threshold=0.25, max_tags=40):   
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.max_tags = max_tags

        # ✅ Load indexes
        self.ddb_index = self.load_json(self.ddb_index_file)
        self.image_index = self.load_json(self.image_index_file)

    def load_json(self, file_path):
        """Loads JSON file if it exists, otherwise returns an empty dictionary."""
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    #print(f"⚠️ Warning: {file_path} is corrupted. Creating a new one.")
                    return {}
        return {}

    def extract_top_tags(self, ddb_json_path):
        """Extracts and filters top confidence tags from a DeepDanbooru JSON file."""
        if not os.path.exists(ddb_json_path):
            #print(f"⚠️ No DeepDanbooru JSON file found at {ddb_json_path}")
            return {}

        with open(ddb_json_path, "r", encoding="utf-8") as f:
            raw_tags = json.load(f)

        # ✅ Filter out low-confidence scores
        filtered_tags = {
            tag: confidence for tag, confidence in raw_tags.items() if confidence >= self.confidence_threshold
        }

        # ✅ Sort by confidence and limit the number of tags
        sorted_tags = dict(sorted(filtered_tags.items(), key=lambda item: item[1], reverse=True)[:self.max_tags])

        return sorted_tags
        
    
    def update_image_index(self):
        """Updates `image_index.json` with DeepDanbooru tags for each image."""
        for filename, data in self.image_index.items():
            if filename in self.ddb_index:
                ddb_json_path = self.ddb_index[filename]["deepdanbooru_json"]

                # ✅ Extract top tags
                top_tags = self.extract_top_tags(ddb_json_path)

                if top_tags:
                    self.image_index[filename]["DeepDanbooru"] = top_tags
                    #print(f"✅ Updated {filename} with DeepDanbooru tags.")

        # ✅ Save updated `image_index.json`
        self.save_json(self.image_index_file, self.image_index)
        
       

    def save_json(self, file_path, data):
        """Saves the JSON data to a file."""
        temp_file = file_path + ".dat"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        os.replace(temp_file, file_path)
        #print(f"✅ Index updated: {file_path}")


if __name__ == "__main__":
    updater = DeepDanbooruUpdater()
    updater.update_image_index()
