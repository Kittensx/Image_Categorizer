import os
import json
import shutil
from datetime import datetime
import logging



# Define paths for master lists and backups
MASTER_LIST_FOLDER = os.path.join("categories", "_master")
BACKUP_FOLDER = os.path.join("categories", "_backup")

# Ensure the backup folder exists
os.makedirs(BACKUP_FOLDER, exist_ok=True)




def backup_file(file_path):
    """Creates a backup of the original JSON file before modification."""
    if not os.path.exists(file_path):
        return

    filename = os.path.basename(file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{timestamp}_{filename}"
    backup_path = os.path.join(BACKUP_FOLDER, backup_filename)

    shutil.copy2(file_path, backup_path)
    #print(f"🔄 Backup saved: {backup_filename}")

def auto_correct_top_level_key(data, expected_key, file_path):
    """
    Auto-corrects the top-level key in the JSON if it is incorrect.
    - If only one key exists, rename it to match the expected category.
    - If multiple keys exist and the expected key is missing, wrap everything inside the expected key.
    - If the expected key exists, return the data unchanged.
    """
    keys = list(data.keys())

    if len(keys) == 1 and keys[0] != expected_key:
        print(f"🔄 Auto-correcting key: '{keys[0]}' → '{expected_key}' in {file_path}")
        corrected_data = {expected_key: data[keys[0]]}
    elif expected_key not in data:
        print(f"⚠️ No matching key found for '{expected_key}'. Wrapping all contents inside '{expected_key}'.")
        corrected_data = {expected_key: data}  # Wrap everything inside the correct key
    else:
        return data  # Already correct

    # Backup file before modifying
    backup_file(file_path)

    # Save the corrected file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(corrected_data, f, indent=4)

    return corrected_data

def load_master_lists():
    """
    Loads all master category lists from the `_master` folder.
    - Scans for files prefixed with `master_`
    - Extracts the category name
    - Ensures the first key in the JSON matches the expected category
    - Returns a dictionary with all master categories
    """
    master_categories = {}

    if not os.path.exists(MASTER_LIST_FOLDER):
        #print(f"❌ Master list folder not found: {MASTER_LIST_FOLDER}")
        return {}

    for filename in os.listdir(MASTER_LIST_FOLDER):
        if filename.startswith("master_") and filename.endswith(".json"):
            category_name = filename.replace("master_", "").replace(".json", "")
            file_path = os.path.join(MASTER_LIST_FOLDER, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Auto-correct top-level key if necessary
                corrected_data = auto_correct_top_level_key(data, category_name, file_path)

                if category_name in corrected_data:
                    master_categories[category_name] = corrected_data[category_name]
                    #print(f"✅ Loaded: {filename} (Corrected key: {category_name})")
                else:
                    logging.warning(f"⚠️ Warning: {filename} does not have '{category_name}' as its top-level key.")

            except json.JSONDecodeError:
                print(f"❌ Error: {filename} is not a valid JSON file.")

    return master_categories

if __name__ == "__main__":
    loaded_categories = load_master_lists()
    logging.info("\n📜 Master Categories Loaded:")
    for key in loaded_categories:
        logging.info(f" - {key} ({len(loaded_categories[key])} subcategories)")
