import os
from PIL import Image
import json
from config_manager import ConfigManager


class ThumbnailCreator(ConfigManager):

    def load_image_index(self, image_index_file=None):
        """Loads the image index file if it exists."""
        image_index_file = image_index_file if image_index_file is not None else self.image_index_file
        if os.path.exists(image_index_file):
            with open(image_index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def generate_thumbnails(self, image_index_file=None, thumbnails_folder=None, xy=None, x=100, y=100):
        """Creates missing thumbnails for images in the index."""
        image_index_file = image_index_file if image_index_file is not None else self.image_index_file
        thumbnails_folder = thumbnails_folder if thumbnails_folder is not None else self.thumbnails_folder
        x = 100 if x is None else x #default value if x is None
        y = 100 if y is None else y #default value if y is None
        if xy is not None:
            x,y = xy, xy #sets both to the same value
        
        image_index = load_image_index()
        
        for image_name, data in image_index.items():
            image_path = data["image_path"]
            thumbnail_path = os.path.join(thumbnails_folder, image_name)
           
           # ✅ Skip if the thumbnail already exists
            if os.path.exists(thumbnail_path):
                print(f"✅ Thumbnail already exists: {thumbnail_path}")
                continue

            
            try:
                img = Image.open(image_path)
                img.thumbnail((x, y))  # Resize to thumbnail size
                img.save(thumbnail_path)
                print(f"✅ Created thumbnail: {thumbnail_path}")
            except Exception as e:
                print(f"❌ Error creating thumbnail for {image_name}: {e}")
           

if __name__ == "__main__":
    generate_thumbnails()
