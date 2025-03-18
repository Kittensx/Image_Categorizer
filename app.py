from flask import Flask, render_template
import json
import os
from PIL import Image
from gallery_summary import GallerySummary  # ✅ Import the summary function

app = Flask(__name__)

# ✅ Define paths
IMAGE_INDEX_FILE = "image_index.json"
THUMBNAIL_FOLDER = "static/thumbnails"
gallery_summary = GallerySummary()

def load_image_index():
    """Loads the image index file if it exists."""
    if os.path.exists(IMAGE_INDEX_FILE):
        with open(IMAGE_INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@app.route("/")
def index():
    """Displays all classified images with their results, including DeepDanbooru tags and gallery summary."""
    image_index = load_image_index()
    images = []

    os.makedirs(THUMBNAIL_FOLDER, exist_ok=True)  # Ensure thumbnail folder exists

    for image_name, data in image_index.items():
        image_path = data["image_path"]  # ✅ Use full original image path
        categories = data.get("categories", "No category found")
        deepdanbooru_tags = data.get("DeepDanbooru", {})

        # ✅ Convert DeepDanbooru tags to a sorted list
        deepdanbooru_sorted = sorted(deepdanbooru_tags.items(), key=lambda item: item[1], reverse=True)

        # ✅ Create a thumbnail if it doesn't exist
        thumbnail_path = os.path.join(THUMBNAIL_FOLDER, image_name)
        if not os.path.exists(thumbnail_path):
            try:
                img = Image.open(image_path)
                img.thumbnail((512, 512))
                img.save(thumbnail_path)
            except Exception as e:
                print(f"❌ Error creating thumbnail for {image_name}: {e}")
                thumbnail_path = None  # ✅ Avoid displaying a broken image

        images.append({
            "filename": image_name,
            "thumbnail": thumbnail_path,
            "full_image": image_path,  # ✅ Now correctly passing full image path
            "results": categories,
            "deepdanbooru": deepdanbooru_sorted
        })

    # ✅ Generate gallery-wide DeepDanbooru summary
    gallery_summary = gallery_summary.generate_gallery_summary()

    return render_template("index.html", images=images, gallery_summary=gallery_summary)


if __name__ == "__main__":
    app.run(debug=True)
