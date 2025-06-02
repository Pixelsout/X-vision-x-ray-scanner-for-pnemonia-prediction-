# radiology_model/config.py
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 5  # Example: pneumonia, covid, etc.
DATA_PATH = "data/chest_xray"
MODEL_SAVE_PATH = "models/classifier.pth"
