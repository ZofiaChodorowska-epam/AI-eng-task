import re
import os
import logging
import warnings

# Suppress logs and progress bars globally
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*truncate.*")

# Try importing GLiNER
try:
    from gliner import GLiNER
    HAS_GLINER = True
except ImportError:
    HAS_GLINER = False

# Lazy loader for model
MODEL_NAME = "urchade/gliner_small-v2.1"
_model = None

def get_model():
    global _model
    if _model is None and HAS_GLINER:
        try:
            _model = GLiNER.from_pretrained(MODEL_NAME)
        except Exception:
            _model = None
    return _model

def filter_sensitive_data_regex(text: str) -> str:
    """
    Fallback: Analyze and redact sensitive PII from the text using Regex.
    """
    if not text:
        return ""
        
    # Email
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL REDACTED]', text)
    # Phone (US/Intl formats)
    phone_pattern = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b' 
    text = re.sub(phone_pattern, '[PHONE REDACTED]', text)
    # Credit Card
    text = re.sub(r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '[CARD REDACTED]', text)
    
    return text

def filter_sensitive_data(text: str) -> str:
    """
    Primary interface: Redact PII using GLiNER (NLP) if available, else Regex.
    """
    if not text:
        return ""

    model = get_model()
    if model:
        try:
            # Define labels we want to catch
            # User request: Name and Car Plate are NOT sensitive. 
            # Sensitive: Phone, Email, Address. 
            # We include "person", "location" and "license plate" to avoid misclassification
            labels = ["person", "phone number", "email address", "address", "location", "license plate"]
            
            # Predict
            entities = model.predict_entities(text, labels, threshold=0.3)
            entities.sort(key=lambda x: x["start"], reverse=True)
            
            redacted_text = text
            for entity in entities:
                start = entity["start"]
                end = entity["end"]
                label = entity["label"]
                text_segment = text[start:end]
                
                # Heuristic: If label is "license plate" but looks like a phone number (e.g. 555-0199),
                # force redaction as phone number.
                # Regex for 3-4 digit pattern often seen in local US numbers or short codes
                if label == "license plate" and re.search(r'^\d{3}-\d{4}$', text_segment.strip()):
                    label = "phone number"

                if label == "phone number":
                    # Only redact if it actually contains digits
                    if not re.search(r'\d', text_segment):
                        continue
                        
                # 3. SKIP redaction for non-sensitive types per user request
                # Names, Plates, and generic Locations (e.g. "New York") are visible.
                # Specific "address" (e.g. "123 Main St") will still be redacted.
                if label in ["person", "license plate", "location"]:
                    continue
                    
                replacement = f"[{label.upper()} REDACTED]"
                
                # Apply replacement (be careful with offsets if multiple replaces happen, 
                # but here we iterate sorted reverse so it's safe)
                redacted_text = redacted_text[:start] + replacement + redacted_text[end:]
                
            return redacted_text
        except Exception:
            # Fallback if prediction errors
            return filter_sensitive_data_regex(text)
    
    # Fallback to Regex if model not available or loaded
    return filter_sensitive_data_regex(text)

def check_input_safety(text: str) -> bool:
    """
    Basic check intent using regex.
    """
    if re.search(r'\b(?:\d{4}[-\s]?){3}\d{4}\b', text):
        return False
    return True
