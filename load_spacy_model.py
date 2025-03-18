import spacy
from config_manager import ConfigManager
from download_spacy import SpacyInstall

class SpacyModel(ConfigManager):     
    """Singleton class for loading and managing the spaCy model."""
    _instance = None  # Holds the single instance
    
    def __new__(cls, spacy_model="en_core_web_md"):  # ✅ Add a default model name
        if cls._instance is None:
            cls._instance = super(SpacyModel, cls).__new__(cls)
            cls._instance.spacy_model = spacy_model  # ✅ Define spacy_model before using it
            print(f"🔄 Loading spaCy model: {cls._instance.spacy_model}")  
            cls._instance.nlp = cls._instance._load_model()  # ✅ Fix method call
        else:
            print("✅ Using cached spaCy model instance.")
        return cls._instance  # ✅ Return the singleton instance

    def _load_model(self):
        """Load the spaCy model, with fallback if missing."""
        try:
            return spacy.load(self.spacy_model)  # ✅ Load spaCy model
        except OSError:
            spacy_install = SpacyInstall() 
            print(f"❌ Failed to load spaCy model: {self.spacy_model}. Attempting reinstall...")
            spacy_install.install_spacy_model()  # ✅ Install missing model
            return spacy.load(self.spacy_model)  # ✅ Retry loading

    def unload_model(self):
        """Unloads the spaCy model from memory."""
        if self.nlp is not None:
            print("🔄 Unloading spaCy model from memory...")
            del self.nlp
            self.nlp = None
            print("✅ spaCy model successfully unloaded!")


'''
# ✅ Example Usage:
spacy1 = SpacyModel("en_core_web_md")  # First call: Loads the model
spacy2 = SpacyModel("en_core_web_md")  # Second call: Uses the same model instance

print(spacy1 is spacy2)  # ✅ True (Singleton check)

# ✅ Unload the model when done
spacy1.unload_model()
'''