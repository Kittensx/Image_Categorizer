import os
import tensorflow as tf
import gc
from config_manager import ConfigManager

class DeepDanbooruModel(ConfigManager):
    _instance = None  # Stores the single loaded model
    def __init__(self, model_path):
        super().__init__()  # ✅ Do NOT pass model_path to ConfigManager
        self.model_path = model_path  
        self.model = self._load_model()
        
    def __new__(cls, model_path):
        """Ensures only one model instance is created (Singleton)."""
        if cls._instance is None:
            print("🔄 Loading DeepDanbooru model (first time)...")
            cls._instance = super(DeepDanbooruModel, cls).__new__(cls)
            cls._instance.model_path = model_path
            cls._instance.model = cls._instance._load_model()
        else:
            print("✅ Using cached DeepDanbooru model instance.")
        return cls._instance  # Return the same instance for all new objects

    def _load_model(self):
        """Loads DeepDanbooru model from the local repository."""
        model_file = os.path.join(self.model_path, "model-resnet_custom_v3.h5")

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"❌ DeepDanbooru model not found at {model_file}")

        print(f"✅ Loading DeepDanbooru model from {model_file}...")
        model = tf.keras.models.load_model(model_file, compile=False)
        return model

    def unload_model(self):
        """Safely unloads the DeepDanbooru model from GPU memory."""
        if self.model is not None:
            print("🔄 Unloading DeepDanbooru model from GPU...")

            # Delete model from memory
            del self.model
            self.model = None

            # Force garbage collection to clean up memory
            gc.collect()

            # Clear TensorFlow GPU memory cache
            tf.keras.backend.clear_session()

            print("✅ DeepDanbooru model successfully unloaded!")
        else:
            print("⚠️ No model found to unload.")

'''
# ✅ Usage Example
#MODEL_PATH = "/path/to/deepdanbooru"  # Replace with actual path

dd_model = DeepDanbooruModel(MODEL_PATH)  # Loads the model

# ✅ Unload when finished
dd_model.unload_model()
'''
