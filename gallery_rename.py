import os
import shutil
from config_manager import ConfigManager

class GalleryMaker(ConfigManager):
    def __init__(self):
        super().__init__()
        #//update these with config values
        self.gallery_root_path = os.path.abspath(os.path.join(self.root_path, "static"))

    def organize_new_gallery(self, gallery_root_path=None):
        """Creates a new gallery folder inside the root directory and moves index & thumbnail files inside it."""
        gallery_root_path = gallery_root_path if gallery_root_path is not None else self.gallery_root_path
        # Ensure 'static' directory exists
        static_dir = gallery_root_path
        os.makedirs(static_dir, exist_ok=True)

        # Count existing Gallery folders to determine the next number
        existing_galleries = [f for f in os.listdir(static_dir)]
        next_gallery_num = len(existing_galleries)

        # Format gallery name (4-digit padding)
        new_gallery_folder = os.path.join(static_dir, f"Gallery_{next_gallery_num:04d}")
        os.makedirs(new_gallery_folder, exist_ok=True)
                  
        # Rename Thumbnails folder (if it exists)
        new_thumbnails_folder = os.path.join(new_gallery_folder, f"Thumbnails_{next_gallery_num:04d}")
        if os.path.exists(self.thumbnails_folder):
            shutil.move(self.thumbnails_folder, new_thumbnails_folder)

        # Move index files into the new gallery folder
        for file in [self.ddb_index_file, self.image_index_file]:
            if os.path.exists(file):
                shutil.move(file, os.path.join(new_gallery_folder, os.path.basename(file)))

        print(f"✅ New gallery created: {new_gallery_folder}")
