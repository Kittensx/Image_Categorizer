import torch
from torchvision import models
from config_manager import ConfigManager

class ResNet50Model(ConfigManager):
    """Singleton class for loading and managing the ResNet50 model."""

    _instance = None  # Holds the single instance

    def __new__(cls):
        if cls._instance is None:
            print("🔄 Loading ResNet50 model (first time)...")
            cls._instance = super(ResNet50Model, cls).__new__(cls)
            cls._instance.model, cls._instance.labels = cls._load_model()
        else:
            print("✅ Using cached ResNet50 model instance.")
        return cls._instance  # Return the same instance for all future calls

    @staticmethod
    def _load_model():
        """Load the ResNet50 model and labels."""
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.eval()
        labels = models.ResNet50_Weights.DEFAULT.meta["categories"]
        return model, labels

    def unload_model(self):
        """Unloads the model from memory."""
        if self.model is not None:
            print("🔄 Unloading ResNet50 model from memory...")
            del self.model
            self.model = None
            torch.cuda.empty_cache()  # Clear CUDA memory if used
            print("✅ ResNet50 model successfully unloaded!")

'''
# ✅ Example Usage:
resnet1 = ResNet50Model()  # First call: Loads the model
resnet2 = ResNet50Model()  # Second call: Uses the same model instance

print(resnet1 is resnet2)  # ✅ True (Singleton check)

# ✅ Unload the model when done
resnet1.unload_model()
'''
