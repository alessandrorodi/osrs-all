"""
OCR Text Recognition for OSRS

This module provides optical character recognition capabilities for reading
text in OSRS interfaces, chat messages, item names, and other game elements.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import re

# Fix for newer Pillow versions (10.x) - EasyOCR compatibility
try:
    import PIL.Image
    if not hasattr(PIL.Image, 'ANTIALIAS'):
        PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
        logging.getLogger(__name__).debug("Applied PIL.Image.ANTIALIAS compatibility fix")
except ImportError:
    pass

import easyocr
import pytesseract

from core.computer_vision import Detection

logger = logging.getLogger(__name__)


@dataclass
class TextDetection(Detection):
    """Text detection with OCR-specific information"""
    text: str = ""  # Recognized text
    text_confidence: float = 0.0  # OCR confidence
    language: str = "en"  # Detected language
    text_type: str = ""  # chat, interface, item_name, etc.
    is_clickable: bool = False  # Whether text is clickable UI element
    
    @property
    def clean_text(self) -> str:
        """Return cleaned text without special characters"""
        # Remove non-alphanumeric characters and extra spaces
        cleaned = re.sub(r'[^\w\s-]', '', self.text)
        return ' '.join(cleaned.split())


class OCRDetector:
    """
    OCR text detection for OSRS game elements
    
    Detects and reads:
    - Chat messages
    - Interface text (buttons, labels)
    - Item names and descriptions
    - Player names
    - Quest dialogue
    - Skill levels and XP values
    """
    
    def __init__(self, languages: Optional[List[str]] = None, use_gpu: bool = True):
        """
        Initialize OCR detector
        
        Args:
            languages: List of languages to recognize (default: ['en'])
            use_gpu: Whether to use GPU acceleration if available
        """
        self.languages = languages or ['en']
        self.use_gpu = use_gpu
        
        # Initialize EasyOCR reader
        try:
            self.easyocr_reader = easyocr.Reader(
                self.languages, 
                gpu=self.use_gpu
            )
            logger.info(f"EasyOCR initialized with languages: {self.languages}")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.easyocr_reader = None
        
        # Text preprocessing settings
        self.preprocessing_methods = {
            'chat': self._preprocess_chat_text,
            'interface': self._preprocess_interface_text,
            'item_name': self._preprocess_item_text,
            'player_name': self._preprocess_player_text,
            'numbers': self._preprocess_numbers
        }
        
        # Known OSRS text patterns
        self.osrs_patterns = {
            'item_name': r'^[A-Za-z\s\'-]+$',
            'player_name': r'^[A-Za-z0-9\s_-]{1,12}$',
            'number': r'^\d{1,3}(,\d{3})*$',
            'skill_level': r'^Level \d{1,2}: \d{1,8} XP$',
            'chat_message': r'^[A-Za-z0-9\s\!\?\.\,\:\;\'\"_-]+$'
        }
    
    def detect_text(self, image: np.ndarray, 
                   text_types: Optional[List[str]] = None,
                   confidence_threshold: float = 0.5,
                   region: Optional[Tuple[int, int, int, int]] = None) -> List[TextDetection]:
        """
        Detect and recognize text in image
        
        Args:
            image: Input image (BGR format)
            text_types: Types of text to look for (None for all)
            confidence_threshold: Minimum confidence for text recognition
            region: Region of interest (x, y, width, height)
            
        Returns:
            List of detected text elements
        """
        if self.easyocr_reader is None:
            logger.warning("EasyOCR not available")
            return []
        
        try:
            # Use region of interest if specified
            search_image = image
            offset_x, offset_y = 0, 0
            
            if region:
                x, y, w, h = region
                search_image = image[y:y+h, x:x+w]
                offset_x, offset_y = x, y
            
            # Run OCR
            results = self.easyocr_reader.readtext(search_image)
            
            detections = []
            
            for (bbox, text, confidence) in results:
                if confidence < confidence_threshold:
                    continue
                
                # Extract bounding box coordinates
                bbox = np.array(bbox, dtype=int)
                x_min, y_min = bbox.min(axis=0)
                x_max, y_max = bbox.max(axis=0)
                
                # Adjust for region offset
                x_min += offset_x
                y_min += offset_y
                x_max += offset_x
                y_max += offset_y
                
                # Determine text type
                text_type = self._classify_text_type(text, text_types)
                
                # Create detection
                detection = TextDetection(
                    x=x_min,
                    y=y_min,
                    width=x_max - x_min,
                    height=y_max - y_min,
                    confidence=confidence,
                    label=f"text_{text_type}",
                    text=text,
                    text_confidence=confidence,
                    text_type=text_type,
                    is_clickable=self._is_clickable_text(text, text_type)
                )
                
                detections.append(detection)
            
            logger.debug(f"OCR detected {len(detections)} text elements")
            return detections
            
        except Exception as e:
            logger.error(f"OCR detection failed: {e}")
            return []
    
    def read_chat_messages(self, image: np.ndarray,
                          chat_region: Optional[Tuple[int, int, int, int]] = None) -> List[str]:
        """
        Read chat messages from the chat box
        
        Args:
            image: Input image
            chat_region: Chat box region (x, y, width, height)
            
        Returns:
            List of chat messages
        """
        # Preprocess image for chat text
        processed_image = self._preprocess_chat_text(image)
        
        # Detect text in chat region
        detections = self.detect_text(
            processed_image,
            text_types=['chat'],
            confidence_threshold=0.3,  # Lower threshold for chat
            region=chat_region
        )
        
        # Extract and clean chat messages
        messages = []
        for detection in detections:
            if detection.text_type == 'chat':
                clean_msg = detection.clean_text
                if len(clean_msg) > 2:  # Filter out very short text
                    messages.append(clean_msg)
        
        return messages
    
    def read_interface_text(self, image: np.ndarray,
                           interface_region: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Read text from game interface elements
        
        Args:
            image: Input image
            interface_region: Interface region to scan
            
        Returns:
            Dictionary with interface text elements
        """
        # Preprocess for interface text
        processed_image = self._preprocess_interface_text(image)
        
        # Detect interface text
        detections = self.detect_text(
            processed_image,
            text_types=['interface'],
            confidence_threshold=0.6,
            region=interface_region
        )
        
        interface_data = {
            'buttons': [],
            'labels': [],
            'values': [],
            'clickable_elements': []
        }
        
        for detection in detections:
            text = detection.clean_text
            
            # Categorize interface elements
            if detection.is_clickable:
                interface_data['clickable_elements'].append({
                    'text': text,
                    'position': detection.center,
                    'bbox': detection.bbox
                })
            
            if self._is_button_text(text):
                interface_data['buttons'].append(text)
            elif self._is_value_text(text):
                interface_data['values'].append(text)
            else:
                interface_data['labels'].append(text)
        
        return interface_data
    
    def read_item_names(self, image: np.ndarray,
                       inventory_region: Optional[Tuple[int, int, int, int]] = None) -> List[Dict[str, Any]]:
        """
        Read item names from inventory or ground items
        
        Args:
            image: Input image
            inventory_region: Inventory region to scan
            
        Returns:
            List of item information dictionaries
        """
        # Preprocess for item text
        processed_image = self._preprocess_item_text(image)
        
        # Detect item text
        detections = self.detect_text(
            processed_image,
            text_types=['item_name'],
            confidence_threshold=0.4,
            region=inventory_region
        )
        
        items = []
        for detection in detections:
            if detection.text_type == 'item_name':
                item_info = {
                    'name': detection.clean_text,
                    'position': detection.center,
                    'bbox': detection.bbox,
                    'confidence': detection.text_confidence,
                    'is_valuable': self._is_valuable_item(detection.clean_text)
                }
                items.append(item_info)
        
        return items
    
    def read_numbers(self, image: np.ndarray,
                    region: Optional[Tuple[int, int, int, int]] = None) -> List[Tuple[int, Tuple[int, int]]]:
        """
        Read numeric values (XP, levels, quantities, etc.)
        
        Args:
            image: Input image
            region: Region to scan for numbers
            
        Returns:
            List of (number, position) tuples
        """
        # Preprocess for numbers
        processed_image = self._preprocess_numbers(image)
        
        # Use Tesseract for better number recognition
        numbers = []
        try:
            # Configure Tesseract for numbers only
            config = '--psm 6 -c tessedit_char_whitelist=0123456789,'
            
            if region:
                x, y, w, h = region
                roi = processed_image[y:y+h, x:x+w]
            else:
                roi = processed_image
                x, y = 0, 0
            
            # Extract text
            text = pytesseract.image_to_string(roi, config=config)
            
            # Find numbers in text
            number_matches = re.findall(r'\d{1,3}(?:,\d{3})*', text)
            
            for match in number_matches:
                try:
                    # Convert to integer (remove commas)
                    number = int(match.replace(',', ''))
                    
                    # For now, use center of region as position
                    # In practice, you'd want to get exact position from OCR
                    pos_x = x + (roi.shape[1] // 2) if region else roi.shape[1] // 2
                    pos_y = y + (roi.shape[0] // 2) if region else roi.shape[0] // 2
                    
                    numbers.append((number, (pos_x, pos_y)))
                    
                except ValueError:
                    continue
                    
        except Exception as e:
            logger.error(f"Number recognition failed: {e}")
        
        return numbers
    
    def _classify_text_type(self, text: str, allowed_types: Optional[List[str]] = None) -> str:
        """Classify the type of detected text"""
        text_lower = text.lower().strip()
        
        # Check specific patterns first (most specific to least specific)
        if re.match(self.osrs_patterns['number'], text):
            text_type = 'number'
        elif re.match(self.osrs_patterns['skill_level'], text):
            text_type = 'skill_level'
        elif any(word in text_lower for word in ['says', 'yells', 'whispers']):
            text_type = 'chat'
        elif any(word in text_lower for word in ['click', 'press', 'select', 'ok', 'cancel', 'attack', 'examine', 'use', 'talk-to']):
            text_type = 'interface'
        elif re.match(self.osrs_patterns['player_name'], text) and len(text) >= 3 and len(text) <= 12:
            # More restrictive player name check - avoid single words that could be interface
            # Single words are more likely interface elements unless they contain numbers/mixed case
            if ' ' in text or any(c.isdigit() for c in text) or (text != text.lower() and text != text.upper()):
                # Multi-word, contains numbers, or mixed case - likely player name
                if not any(interface_word in text_lower for interface_word in ['click', 'here', 'press', 'button', 'menu']):
                    text_type = 'player_name'
                else:
                    text_type = 'interface'
            else:
                # Single word, no numbers, uniform case - likely interface
                text_type = 'interface'
        else:
            # Default classification
            if len(text) < 3:
                text_type = 'interface'  # Very short text likely interface elements
            elif len(text) < 20:
                text_type = 'item_name'
            else:
                text_type = 'chat'
        
        # Filter by allowed types
        if allowed_types and text_type not in allowed_types:
            text_type = 'unknown'
        
        return text_type
    
    def _is_clickable_text(self, text: str, text_type: str) -> bool:
        """Determine if text represents a clickable element"""
        clickable_keywords = [
            'click', 'press', 'select', 'ok', 'cancel', 'yes', 'no',
            'buy', 'sell', 'trade', 'bank', 'deposit', 'withdraw',
            'attack', 'talk-to', 'examine', 'use', 'walk here'
        ]
        
        text_lower = text.lower()
        return (text_type == 'interface' or 
                any(keyword in text_lower for keyword in clickable_keywords))
    
    def _is_button_text(self, text: str) -> bool:
        """Check if text is likely a button"""
        button_patterns = [
            r'^(OK|Cancel|Yes|No|Accept|Decline)$',
            r'^(Buy|Sell|Trade|Exchange)$',
            r'^(Bank|Deposit|Withdraw)$',
            r'^(Attack|Talk-to|Examine|Use)$'
        ]
        
        return any(re.match(pattern, text, re.IGNORECASE) for pattern in button_patterns)
    
    def _is_value_text(self, text: str) -> bool:
        """Check if text represents a numeric value"""
        return re.match(r'^\d{1,3}(,\d{3})*$', text) is not None
    
    def _is_valuable_item(self, item_name: str) -> bool:
        """Check if item is potentially valuable"""
        valuable_keywords = [
            'rune', 'dragon', 'barrows', 'godsword', 'whip', 'staff',
            'ring', 'amulet', 'boots', 'gloves', 'cape', 'shield',
            'ore', 'bar', 'log', 'seed', 'herb', 'potion'
        ]
        
        item_lower = item_name.lower()
        return any(keyword in item_lower for keyword in valuable_keywords)
    
    # Image preprocessing methods
    
    def _preprocess_chat_text(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for chat text recognition"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast for chat text (usually white on dark background)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        
        # Threshold to get white text on black background
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def _preprocess_interface_text(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for interface text recognition"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return blurred
    
    def _preprocess_item_text(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for item name recognition"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance small text
        enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4,4)).apply(gray)
        
        # Sharpen text
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    def _preprocess_player_text(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for player name recognition"""
        # Similar to chat text but less aggressive
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8)).apply(gray)
        
        return enhanced
    
    def _preprocess_numbers(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for number recognition"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Strong contrast enhancement for numbers
        enhanced = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4)).apply(gray)
        
        # Binary threshold for clear number recognition
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned