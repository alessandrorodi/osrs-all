"""
OSRS Text Intelligence System

Enhanced OCR system specifically designed for Old School RuneScape text recognition.
Integrates with existing YOLOv8 vision system and provides comprehensive text analysis.

Key Features:
- GPU-accelerated OCR processing (RTX 4090 optimized)
- OSRS-specific text parsing and understanding
- Combat level extraction, item quantities, GP amounts
- Chat message analysis and filtering
- XP rate calculations and skill notifications
- Performance optimized for <100ms latency
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import logging
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vision.detectors.ocr_detector import OCRDetector, TextDetection
from core.computer_vision import Detection

logger = logging.getLogger(__name__)


@dataclass
class OSRSTextRegion:
    """OSRS-specific text region configuration"""
    name: str
    region: Tuple[int, int, int, int]  # x, y, width, height
    text_type: str
    preprocessing: str = "default"
    confidence_threshold: float = 0.5
    scan_frequency: float = 1.0  # seconds between scans
    priority: int = 1  # 1=high, 2=medium, 3=low


@dataclass
class OSRSTextData:
    """Structured OSRS text information"""
    raw_text: str
    processed_text: str
    text_type: str
    confidence: float
    position: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    timestamp: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChatMessage:
    """OSRS chat message structure"""
    player_name: str
    message: str
    chat_type: str  # public, private, clan, game, trade
    timestamp: float
    color: Optional[str] = None
    is_system: bool = False
    
    def __post_init__(self):
        # Clean and validate message
        self.message = self.message.strip()
        self.player_name = self.player_name.strip()


@dataclass
class ItemInfo:
    """OSRS item information from OCR"""
    name: str
    quantity: int = 1
    noted: bool = False
    position: Tuple[int, int] = (0, 0)
    ge_price: Optional[int] = None
    high_alch: Optional[int] = None
    is_valuable: bool = False
    category: str = "misc"


@dataclass
class PlayerStats:
    """Player statistics from OCR"""
    health_current: Optional[int] = None
    health_max: Optional[int] = None
    prayer_current: Optional[int] = None
    prayer_max: Optional[int] = None
    energy_percent: Optional[int] = None
    combat_level: Optional[int] = None
    skill_levels: Optional[Dict[str, int]] = None
    
    def __post_init__(self):
        if self.skill_levels is None:
            self.skill_levels = {}


class OSRSTextIntelligence:
    """
    Advanced OSRS text recognition and intelligence system
    
    Provides comprehensive text analysis including:
    - Chat message processing and filtering
    - Item name and quantity recognition
    - Player stats and combat levels
    - Interface text and navigation elements
    - XP tracking and calculations
    - Price checking and value estimation
    """
    
    def __init__(self, device: str = "auto", batch_size: int = 4):
        """
        Initialize OSRS text intelligence system
        
        Args:
            device: Processing device ('cpu', 'cuda', 'auto')
            batch_size: Batch size for GPU processing
        """
        self.device = device
        self.batch_size = batch_size
        
        # Initialize base OCR detector with GPU acceleration
        self.ocr = OCRDetector(
            languages=['en'], 
            use_gpu=(device != 'cpu')
        )
        
        # OSRS-specific text regions (calibrated for fixed/resizable modes)
        self.text_regions = self._initialize_text_regions()
        
        # Text processing patterns
        self.osrs_patterns = self._initialize_patterns()
        
        # Caching for performance
        self.text_cache = {}
        self.cache_timeout = 1.0  # seconds
        
        # Performance tracking
        self.performance_stats = {
            'processing_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'total_ocr_calls': 0,
            'avg_latency': 0.0
        }
        
        # Threading for background processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.processing_lock = threading.Lock()
        
        # Item database for value estimation
        self.item_db = self._load_item_database()
        
        logger.info(f"OSRS Text Intelligence initialized (device: {device})")
    
    def _initialize_text_regions(self) -> Dict[str, OSRSTextRegion]:
        """Initialize OSRS-specific text regions"""
        regions = {
            # Chat regions
            'public_chat': OSRSTextRegion(
                name='public_chat',
                region=(7, 345, 506, 40),  # Main chat area
                text_type='chat',
                preprocessing='chat',
                confidence_threshold=0.4,
                scan_frequency=0.5,
                priority=1
            ),
            'private_chat': OSRSTextRegion(
                name='private_chat',
                region=(7, 385, 506, 20),  # Private message area
                text_type='private_chat',
                preprocessing='chat',
                confidence_threshold=0.4,
                scan_frequency=0.5,
                priority=1
            ),
            'game_messages': OSRSTextRegion(
                name='game_messages',
                region=(7, 405, 506, 40),  # Game messages
                text_type='game_message',
                preprocessing='chat',
                confidence_threshold=0.5,
                scan_frequency=1.0,
                priority=2
            ),
            
            # Interface regions
            'inventory_items': OSRSTextRegion(
                name='inventory_items',
                region=(548, 205, 186, 262),  # Inventory area
                text_type='item_name',
                preprocessing='items',
                confidence_threshold=0.3,
                scan_frequency=2.0,
                priority=2
            ),
            'equipment_stats': OSRSTextRegion(
                name='equipment_stats',
                region=(548, 205, 186, 262),  # Equipment tab
                text_type='equipment',
                preprocessing='interface',
                confidence_threshold=0.6,
                scan_frequency=5.0,
                priority=3
            ),
            
            # Player stats
            'health_orb': OSRSTextRegion(
                name='health_orb',
                region=(8, 45, 60, 60),  # Health orb area
                text_type='health',
                preprocessing='numbers',
                confidence_threshold=0.7,
                scan_frequency=0.5,
                priority=1
            ),
            'prayer_orb': OSRSTextRegion(
                name='prayer_orb',
                region=(8, 85, 60, 60),  # Prayer orb area
                text_type='prayer',
                preprocessing='numbers',
                confidence_threshold=0.7,
                scan_frequency=1.0,
                priority=2
            ),
            'energy_orb': OSRSTextRegion(
                name='energy_orb',
                region=(8, 125, 60, 60),  # Energy orb area
                text_type='energy',
                preprocessing='numbers',
                confidence_threshold=0.7,
                scan_frequency=2.0,
                priority=2
            ),
            
            # Combat and targeting
            'npc_examine': OSRSTextRegion(
                name='npc_examine',
                region=(4, 4, 512, 50),  # Top of main screen for examine text
                text_type='examine',
                preprocessing='interface',
                confidence_threshold=0.5,
                scan_frequency=1.0,
                priority=2
            ),
            'combat_level': OSRSTextRegion(
                name='combat_level',
                region=(200, 50, 200, 100),  # Combat level display area
                text_type='combat_level',
                preprocessing='numbers',
                confidence_threshold=0.6,
                scan_frequency=3.0,
                priority=2
            ),
            
            # Trading and GE
            'trade_screen': OSRSTextRegion(
                name='trade_screen',
                region=(100, 100, 400, 300),  # Trade interface
                text_type='trade',
                preprocessing='interface',
                confidence_threshold=0.6,
                scan_frequency=1.0,
                priority=1
            ),
            'ge_prices': OSRSTextRegion(
                name='ge_prices',
                region=(100, 150, 400, 200),  # Grand Exchange prices
                text_type='price',
                preprocessing='numbers',
                confidence_threshold=0.7,
                scan_frequency=2.0,
                priority=2
            )
        }
        
        return regions
    
    def _initialize_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize OSRS-specific text patterns"""
        return {
            'chat_patterns': {
                'public_chat': r'^([A-Za-z0-9\s_-]+):\s*(.+)$',
                'private_from': r'^From\s+([A-Za-z0-9\s_-]+):\s*(.+)$',
                'private_to': r'^To\s+([A-Za-z0-9\s_-]+):\s*(.+)$',
                'clan_chat': r'^\[([A-Za-z0-9\s_-]+)\]\s*([A-Za-z0-9\s_-]+):\s*(.+)$',
                'game_message': r'^(You\s+.+|Your\s+.+|\w+\s+(gains?|receives?|loses?).+)$',
                'trade_message': r'.*(wishes to trade with you|declined trade|accepted trade).*'
            },
            'item_patterns': {
                'item_with_quantity': r'^(.+?)\s*[x×]?\s*(\d{1,8})$',
                'item_noted': r'^(.+?)\s*\(noted\)$',
                'item_name_only': r'^([A-Za-z\s\'-]+)$',
                'stack_quantity': r'^(\d{1,3}(?:,\d{3})*[KMB]?)$'
            },
            'number_patterns': {
                'health': r'^(\d+)/(\d+)$',
                'prayer': r'^(\d+)/(\d+)$',
                'energy': r'^(\d+)%?$',
                'combat_level': r'^Level-(\d+)$|^Lvl-(\d+)$|^(\d+)$',
                'xp_drop': r'^\+(\d{1,6})\s+XP$',
                'gp_amount': r'^(\d{1,3}(?:,\d{3})*)\s*gp$',
                'price': r'^(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:gp|coins?)?$'
            },
            'interface_patterns': {
                'button': r'^(OK|Cancel|Yes|No|Accept|Decline|Buy|Sell|Bank)$',
                'menu_option': r'^(Attack|Talk-to|Examine|Use|Walk here|Take|Drop)$',
                'quest_dialogue': r'^([A-Za-z\s]+):\s*(.+)$',
                'level_up': r'^Congratulations, you just advanced (\w+) level\.$'
            }
        }
    
    def _load_item_database(self) -> Dict[str, Dict[str, Any]]:
        """Load OSRS item database for price checking"""
        # This would load from a JSON file or API
        # For now, return basic valuable items
        return {
            'Dragon scimitar': {'ge_price': 100000, 'high_alch': 60000, 'category': 'weapon'},
            'Rune platebody': {'ge_price': 38000, 'high_alch': 38400, 'category': 'armour'},
            'Shark': {'ge_price': 800, 'high_alch': 0, 'category': 'food'},
            'Prayer potion(4)': {'ge_price': 12000, 'high_alch': 0, 'category': 'potion'},
            'Nature rune': {'ge_price': 200, 'high_alch': 0, 'category': 'rune'},
            'Coal': {'ge_price': 150, 'high_alch': 0, 'category': 'resource'}
        }
    
    def analyze_game_text(self, screenshot: np.ndarray, 
                         regions: Optional[List[str]] = None,
                         force_refresh: bool = False) -> Dict[str, Any]:
        """
        Comprehensive text analysis of OSRS screenshot
        
        Args:
            screenshot: Game screenshot (BGR format)
            regions: Specific regions to analyze (None for all)
            force_refresh: Skip cache and force new OCR
            
        Returns:
            Dictionary with all extracted text information
        """
        start_time = time.time()
        
        try:
            # Determine regions to process
            target_regions = regions or list(self.text_regions.keys())
            
            # Initialize results
            results = {
                'timestamp': start_time,
                'chat_messages': [],
                'items': [],
                'player_stats': PlayerStats(),
                'interface_elements': [],
                'npc_info': [],
                'trade_info': {},
                'performance': {}
            }
            
            # Process regions in parallel by priority
            priority_groups = self._group_regions_by_priority(target_regions)
            
            for priority, region_names in priority_groups.items():
                if priority == 1:  # High priority - process immediately
                    self._process_regions_batch(screenshot, region_names, results, force_refresh)
                else:  # Lower priority - can be processed in background
                    future = self.executor.submit(
                        self._process_regions_batch, 
                        screenshot, region_names, results, force_refresh
                    )
            
            # Post-process and enhance results
            self._enhance_results(results)
            
            # Update performance stats
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
            results['performance'] = {
                'processing_time': processing_time,
                'regions_processed': len(target_regions),
                'cache_efficiency': self._get_cache_efficiency()
            }
            
            logger.debug(f"OSRS text analysis completed in {processing_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"OSRS text analysis failed: {e}")
            return {
                'timestamp': start_time,
                'error': str(e),
                'chat_messages': [],
                'items': [],
                'player_stats': PlayerStats(),
                'interface_elements': [],
                'npc_info': [],
                'trade_info': {},
                'performance': {'processing_time': time.time() - start_time}
            }
    
    def _group_regions_by_priority(self, region_names: List[str]) -> Dict[int, List[str]]:
        """Group regions by processing priority"""
        groups = {}
        for name in region_names:
            if name in self.text_regions:
                priority = self.text_regions[name].priority
                if priority not in groups:
                    groups[priority] = []
                groups[priority].append(name)
        return groups
    
    def _process_regions_batch(self, screenshot: np.ndarray, 
                              region_names: List[str],
                              results: Dict[str, Any],
                              force_refresh: bool = False) -> None:
        """Process a batch of regions"""
        with self.processing_lock:
            for region_name in region_names:
                if region_name not in self.text_regions:
                    continue
                
                region_config = self.text_regions[region_name]
                
                # Check cache first
                cache_key = f"{region_name}_{hash(screenshot.tobytes())}"
                if not force_refresh and cache_key in self.text_cache:
                    cache_entry = self.text_cache[cache_key]
                    if time.time() - cache_entry['timestamp'] < self.cache_timeout:
                        self.performance_stats['cache_hits'] += 1
                        self._merge_region_results(results, cache_entry['data'], region_name)
                        continue
                
                # Process with OCR
                region_data = self._process_single_region(screenshot, region_config)
                
                # Cache results
                self.text_cache[cache_key] = {
                    'data': region_data,
                    'timestamp': time.time()
                }
                self.performance_stats['cache_misses'] += 1
                
                # Merge results
                self._merge_region_results(results, region_data, region_name)
    
    def _process_single_region(self, screenshot: np.ndarray, 
                              region_config: OSRSTextRegion) -> Dict[str, Any]:
        """Process a single text region"""
        try:
            # Extract region
            x, y, w, h = region_config.region
            roi = screenshot[y:y+h, x:x+w]
            
            # Preprocess based on region type
            if region_config.preprocessing == 'chat':
                processed_roi = self.ocr._preprocess_chat_text(roi)
            elif region_config.preprocessing == 'numbers':
                processed_roi = self.ocr._preprocess_numbers(roi)
            elif region_config.preprocessing == 'items':
                processed_roi = self.ocr._preprocess_item_text(roi)
            else:
                processed_roi = self.ocr._preprocess_interface_text(roi)
            
            # Run OCR
            detections = self.ocr.detect_text(
                processed_roi,
                text_types=[region_config.text_type],
                confidence_threshold=region_config.confidence_threshold
            )
            
            # Process detections based on region type
            if region_config.text_type in ['chat', 'private_chat', 'game_message']:
                return self._process_chat_detections(detections, region_config.text_type)
            elif region_config.text_type == 'item_name':
                return self._process_item_detections(detections)
            elif region_config.text_type in ['health', 'prayer', 'energy']:
                return self._process_stat_detections(detections, region_config.text_type)
            elif region_config.text_type == 'combat_level':
                return self._process_combat_level_detections(detections)
            else:
                return self._process_interface_detections(detections)
                
        except Exception as e:
            logger.error(f"Failed to process region {region_config.name}: {e}")
            return {}
    
    def _process_chat_detections(self, detections: List[TextDetection], 
                                chat_type: str) -> Dict[str, Any]:
        """Process chat message detections"""
        messages = []
        
        for detection in detections:
            text = detection.text.strip()
            if len(text) < 2:
                continue
            
            # Parse chat message based on type
            if chat_type == 'chat':
                # Public chat: "PlayerName: message"
                match = re.match(self.osrs_patterns['chat_patterns']['public_chat'], text)
                if match:
                    player_name, message = match.groups()
                    messages.append(ChatMessage(
                        player_name=player_name,
                        message=message,
                        chat_type='public',
                        timestamp=time.time()
                    ))
            elif chat_type == 'private_chat':
                # Private messages
                from_match = re.match(self.osrs_patterns['chat_patterns']['private_from'], text)
                to_match = re.match(self.osrs_patterns['chat_patterns']['private_to'], text)
                
                if from_match:
                    player_name, message = from_match.groups()
                    messages.append(ChatMessage(
                        player_name=player_name,
                        message=message,
                        chat_type='private_from',
                        timestamp=time.time()
                    ))
                elif to_match:
                    player_name, message = to_match.groups()
                    messages.append(ChatMessage(
                        player_name=player_name,
                        message=message,
                        chat_type='private_to',
                        timestamp=time.time()
                    ))
            elif chat_type == 'game_message':
                # Game system messages
                messages.append(ChatMessage(
                    player_name='System',
                    message=text,
                    chat_type='game',
                    timestamp=time.time(),
                    is_system=True
                ))
        
        return {'messages': messages}
    
    def _process_item_detections(self, detections: List[TextDetection]) -> Dict[str, Any]:
        """Process item detection results"""
        items = []
        
        for detection in detections:
            text = detection.clean_text
            if len(text) < 2:
                continue
            
            # Parse item name and quantity
            item_info = self._parse_item_text(text)
            if item_info:
                # Add position information
                item_info.position = detection.center
                
                # Add value information from database
                if item_info.name in self.item_db:
                    db_info = self.item_db[item_info.name]
                    item_info.ge_price = db_info.get('ge_price')
                    item_info.high_alch = db_info.get('high_alch')
                    item_info.category = db_info.get('category', 'misc')
                    item_info.is_valuable = bool(item_info.ge_price and item_info.ge_price > 1000)
                
                items.append(item_info)
        
        return {'items': items}
    
    def _parse_item_text(self, text: str) -> Optional[ItemInfo]:
        """Parse item text to extract name and quantity"""
        # Try item with quantity pattern
        match = re.match(self.osrs_patterns['item_patterns']['item_with_quantity'], text)
        if match:
            name, quantity_str = match.groups()
            try:
                quantity = int(quantity_str.replace(',', ''))
                return ItemInfo(name=name.strip(), quantity=quantity)
            except ValueError:
                pass
        
        # Try noted item pattern
        match = re.match(self.osrs_patterns['item_patterns']['item_noted'], text)
        if match:
            name = match.group(1)
            return ItemInfo(name=name.strip(), noted=True)
        
        # Try simple item name
        match = re.match(self.osrs_patterns['item_patterns']['item_name_only'], text)
        if match:
            name = match.group(1)
            return ItemInfo(name=name.strip())
        
        return None
    
    def _process_stat_detections(self, detections: List[TextDetection], 
                                stat_type: str) -> Dict[str, Any]:
        """Process player stat detections"""
        stats = {}
        
        for detection in detections:
            text = detection.text.strip()
            
            if stat_type == 'health':
                # Parse health: "50/99" or just "50"
                match = re.match(self.osrs_patterns['number_patterns']['health'], text)
                if match:
                    current, maximum = match.groups()
                    stats['health_current'] = int(current)
                    stats['health_max'] = int(maximum)
                else:
                    # Try single number
                    try:
                        stats['health_current'] = int(text)
                    except ValueError:
                        pass
            
            elif stat_type == 'prayer':
                match = re.match(self.osrs_patterns['number_patterns']['prayer'], text)
                if match:
                    current, maximum = match.groups()
                    stats['prayer_current'] = int(current)
                    stats['prayer_max'] = int(maximum)
            
            elif stat_type == 'energy':
                match = re.match(self.osrs_patterns['number_patterns']['energy'], text)
                if match:
                    energy = match.group(1)
                    stats['energy_percent'] = int(energy)
        
        return {'stats': stats}
    
    def _process_combat_level_detections(self, detections: List[TextDetection]) -> Dict[str, Any]:
        """Process combat level detections"""
        for detection in detections:
            text = detection.text.strip()
            
            # Try different combat level patterns
            for pattern in self.osrs_patterns['number_patterns']['combat_level'].split('|'):
                match = re.match(pattern, text)
                if match:
                    # Find the non-None group (different patterns have different group positions)
                    for group in match.groups():
                        if group is not None:
                            try:
                                return {'combat_level': int(group)}
                            except ValueError:
                                continue
        
        return {}
    
    def _process_interface_detections(self, detections: List[TextDetection]) -> Dict[str, Any]:
        """Process interface element detections"""
        elements = []
        
        for detection in detections:
            text = detection.clean_text
            
            element = {
                'text': text,
                'position': detection.center,
                'bbox': detection.bbox,
                'confidence': detection.text_confidence,
                'is_clickable': detection.is_clickable
            }
            
            # Classify interface element type
            if re.match(self.osrs_patterns['interface_patterns']['button'], text, re.IGNORECASE):
                element['element_type'] = 'button'
            elif re.match(self.osrs_patterns['interface_patterns']['menu_option'], text, re.IGNORECASE):
                element['element_type'] = 'menu_option'
            else:
                element['element_type'] = 'label'
            
            elements.append(element)
        
        return {'interface_elements': elements}
    
    def _merge_region_results(self, main_results: Dict[str, Any], 
                             region_results: Dict[str, Any],
                             region_name: str) -> None:
        """Merge region results into main results"""
        if 'messages' in region_results:
            main_results['chat_messages'].extend(region_results['messages'])
        
        if 'items' in region_results:
            main_results['items'].extend(region_results['items'])
        
        if 'stats' in region_results:
            stats = region_results['stats']
            for key, value in stats.items():
                setattr(main_results['player_stats'], key, value)
        
        if 'combat_level' in region_results:
            main_results['player_stats'].combat_level = region_results['combat_level']
        
        if 'interface_elements' in region_results:
            main_results['interface_elements'].extend(region_results['interface_elements'])
    
    def _enhance_results(self, results: Dict[str, Any]) -> None:
        """Post-process and enhance results with additional intelligence"""
        # Calculate item values
        total_value = 0
        for item in results['items']:
            if hasattr(item, 'ge_price') and item.ge_price:
                total_value += item.ge_price * item.quantity
        results['inventory_value'] = total_value
        
        # Analyze chat for important events
        important_messages = []
        for msg in results['chat_messages']:
            if self._is_important_message(msg.message):
                important_messages.append(msg)
        results['important_messages'] = important_messages
        
        # Calculate health percentage
        stats = results['player_stats']
        if stats.health_current and stats.health_max:
            results['health_percentage'] = (stats.health_current / stats.health_max) * 100
    
    def _is_important_message(self, message: str) -> bool:
        """Check if a chat message is important"""
        important_keywords = [
            'level', 'died', 'attack', 'trade', 'follow', 'duel',
            'treasure', 'clue', 'rare', 'drop', 'pk', 'skull'
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in important_keywords)
    
    def _update_performance_stats(self, processing_time: float) -> None:
        """Update performance statistics"""
        self.performance_stats['processing_times'].append(processing_time)
        self.performance_stats['total_ocr_calls'] += 1
        
        # Keep only recent times for rolling average
        if len(self.performance_stats['processing_times']) > 100:
            self.performance_stats['processing_times'] = \
                self.performance_stats['processing_times'][-100:]
        
        # Calculate average latency
        times = self.performance_stats['processing_times']
        self.performance_stats['avg_latency'] = sum(times) / len(times)
    
    def _get_cache_efficiency(self) -> float:
        """Calculate cache hit efficiency"""
        total = self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']
        if total == 0:
            return 0.0
        return self.performance_stats['cache_hits'] / total
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = self.performance_stats.copy()
        stats['cache_efficiency'] = self._get_cache_efficiency()
        stats['gpu_enabled'] = self.device != 'cpu'
        return stats
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        self.text_cache.clear()
        logger.info("OSRS Text Intelligence cleaned up")


# Initialize global instance
osrs_text_intelligence = OSRSTextIntelligence()