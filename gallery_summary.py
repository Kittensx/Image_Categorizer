import json
import os
from collections import Counter
from config_manager import ConfigManager

class GallerySummary(ConfigManager):

    
    def load_json(self, file_path):
        """Loads a JSON file if it exists, otherwise returns an empty dictionary."""
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print(f"⚠️ Warning: {file_path} is corrupted. Creating a new one.")
                    return {}
        return {}

    def generate_gallery_summary(self, ddb_index_path=None, confidence_threshold=0.85):
        """Generates a summary of the most common high-confidence DeepDanbooru tags in the gallery."""
        ddb_index_path = ddb_index_path if ddb_index_path is not None else self.ddb_index_file   
        confidence_threshold = 0.85 if confidence_threshold is None else confidence_threshold
        ddb_index = self.load_json(ddb_index_path)
        tag_counter = Counter()
        total_images = len(ddb_index)

        for filename, data in ddb_index.items():
            ddb_json_path = data.get("deepdanbooru_json")
            if not ddb_json_path or not os.path.exists(ddb_json_path):
                continue
            
            with open(ddb_json_path, "r", encoding="utf-8") as f:
                tags = json.load(f)

            # ✅ Filter tags by confidence threshold and count occurrences
            high_conf_tags = [tag for tag, confidence in tags.items() if confidence >= confidence_threshold]
            tag_counter.update(high_conf_tags)
        
        # ✅ Get the most common tags
        most_common_tags = tag_counter.most_common(20)  # Limit to top 20 tags
        
        # ✅ Format the summary
        summary = {
            "total_images": total_images,
            "top_tags": most_common_tags
        }
        
        print(f"✅ Gallery Summary: {summary}")
        return summary

if __name__ == "__main__":
    gallery_summary = GallerySummary()
    summary = gallery_summary.generate_gallery_summary()
    print("\nGallery Summary Tags:")
    for tag, count in summary["top_tags"]:
        print(f"{tag}: {count} occurrences")