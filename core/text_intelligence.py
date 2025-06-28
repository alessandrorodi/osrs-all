"""
OSRS Text Intelligence Core Module

Advanced text parsing and understanding system for OSRS game elements.
Provides contextual analysis, XP tracking, price calculations, and intelligent
text interpretation for enhanced bot decision-making.

Key Features:
- XP rate calculations and skill progression tracking
- Grand Exchange price analysis and profit calculations
- Combat effectiveness analysis from text cues
- Quest progression and dialogue understanding
- Trade negotiation and market analysis
- Anti-detection through natural language processing
"""

import re
import time
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
from enum import Enum

from vision.osrs_ocr import ChatMessage, ItemInfo, PlayerStats, OSRSTextData

logger = logging.getLogger(__name__)


class TextPriority(Enum):
    """Text analysis priority levels"""
    CRITICAL = 1    # Immediate action required (combat, death, trade)
    HIGH = 2        # Important but not urgent (level up, rare drop)
    MEDIUM = 3      # Useful information (chat, prices)
    LOW = 4         # Background information (general messages)


@dataclass
class XPEvent:
    """Experience point event tracking"""
    skill: str
    xp_gained: int
    timestamp: float
    current_level: Optional[int] = None
    source: str = "unknown"  # combat, skilling, quest, etc.
    
    def __post_init__(self):
        if not self.skill:
            self.skill = "unknown"


@dataclass
class MarketAnalysis:
    """Market price analysis and trends"""
    item_name: str
    current_price: int
    price_trend: str  # "rising", "falling", "stable"
    profit_margin: Optional[int] = None
    high_alch_profit: Optional[int] = None
    recommended_action: str = "hold"  # "buy", "sell", "hold"
    confidence: float = 0.0
    last_updated: float = 0.0


@dataclass
class CombatAnalysis:
    """Combat effectiveness analysis"""
    dps_estimate: float = 0.0
    hit_accuracy: float = 0.0
    damage_taken: int = 0
    healing_used: int = 0
    prayer_used: int = 0
    combat_efficiency: float = 0.0
    threat_level: str = "low"  # "low", "medium", "high", "critical"


@dataclass
class QuestContext:
    """Quest progression and dialogue context"""
    quest_name: Optional[str] = None
    npc_name: Optional[str] = None
    dialogue_stage: int = 0
    expected_responses: Optional[List[str]] = None
    quest_items_needed: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.expected_responses is None:
            self.expected_responses = []
        if self.quest_items_needed is None:
            self.quest_items_needed = []


class OSRSTextIntelligenceCore:
    """
    Core text intelligence system for advanced OSRS text understanding
    
    Provides high-level analysis and contextual understanding of game text,
    enabling intelligent decision-making and automated responses.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize text intelligence core
        
        Args:
            data_dir: Directory for storing intelligence data
        """
        self.data_dir = data_dir or Path("data/intelligence")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # XP tracking
        self.xp_events: List[XPEvent] = []
        self.skill_rates: Dict[str, float] = {}
        self.session_start = time.time()
        
        # Market intelligence
        self.market_data: Dict[str, MarketAnalysis] = {}
        self.price_history: Dict[str, List[Tuple[float, int]]] = {}
        
        # Combat tracking
        self.combat_log: List[Dict[str, Any]] = []
        self.combat_analysis = CombatAnalysis()
        
        # Quest and dialogue tracking
        self.quest_context = QuestContext()
        self.dialogue_history: List[Dict[str, Any]] = []
        
        # Pattern recognition
        self.text_patterns = self._initialize_patterns()
        self.context_memory: Dict[str, Any] = {}
        
        # Load historical data
        self._load_intelligence_data()
        
        logger.info("OSRS Text Intelligence Core initialized")
    
    def _initialize_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize advanced text patterns for intelligence analysis"""
        return {
            'xp_patterns': {
                'xp_drop': r'^\+(\d{1,6})\s+([\w\s]+)\s+XP$',
                'level_up': r'^Congratulations, you just advanced ([\w\s]+) level\.$',
                'level_milestone': r'^Your ([\w\s]+) level is now (\d+)\.$',
                'total_xp': r'^Total XP: ([\d,]+)$',
                'skill_xp': r'^([\w\s]+): ([\d,]+) XP$'
            },
            'combat_patterns': {
                'damage_dealt': r'^You hit (\d+)\.$',
                'damage_taken': r'^You have been hit for (\d+) damage\.$',
                'death': r'^Oh dear, you are dead!$',
                'poison': r'^You have been poisoned\.$',
                'heal': r'^You eat the .+ and heal (\d+) hitpoints\.$',
                'special_attack': r'^You perform a special attack\.$',
                'prayer_drain': r'^You are out of prayer points\.$'
            },
            'trading_patterns': {
                'trade_request': r'^(.+) wishes to trade with you\.$',
                'trade_offer': r'^(.+) is offering: (.+)$',
                'trade_accept': r'^(.+) has accepted the trade\.$',
                'trade_decline': r'^(.+) has declined the trade\.$',
                'ge_offer': r'^Grand Exchange: (.+) for ([\d,]+) coins$',
                'price_check': r'^(.+): ([\d,]+) coins$'
            },
            'social_patterns': {
                'private_message': r'^From (.+): (.+)$',
                'clan_message': r'^\[(.+)\] (.+): (.+)$',
                'public_chat': r'^(.+): (.+)$',
                'system_message': r'^(You .+|Your .+|The .+)$',
                'moderator': r'^\*\*(.+)\*\*: (.+)$'
            },
            'quest_patterns': {
                'quest_start': r'^You have started the (.+) quest\.$',
                'quest_complete': r'^Quest complete: (.+)$',
                'quest_update': r'^Quest journal updated: (.+)$',
                'npc_dialogue': r'^(.+): (.+)$',
                'dialogue_option': r'^\d+\. (.+)$',
                'item_required': r'^You need (.+) to continue\.$'
            },
            'skill_patterns': {
                'resource_gathered': r'^You get some (.+)\.$',
                'crafting_success': r'^You successfully make (.+)\.$',
                'cooking_burn': r'^You accidentally burn the (.+)\.$',
                'mining_success': r'^You manage to mine some (.+)\.$',
                'fishing_catch': r'^You catch (.+)\.$',
                'woodcutting': r'^You get some (.+) log.$'
            }
        }
    
    def analyze_text_intelligence(self, text_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive text intelligence analysis
        
        Args:
            text_data: Text data from OSRS OCR system
            
        Returns:
            Intelligence analysis results
        """
        start_time = time.time()
        
        try:
            analysis_results = {
                'timestamp': start_time,
                'xp_analysis': {},
                'combat_analysis': {},
                'market_analysis': {},
                'quest_analysis': {},
                'social_analysis': {},
                'recommendations': [],
                'alerts': [],
                'context_updates': {}
            }
            
            # Analyze chat messages
            if 'chat_messages' in text_data:
                self._analyze_chat_intelligence(text_data['chat_messages'], analysis_results)
            
            # Analyze items and inventory
            if 'items' in text_data:
                self._analyze_item_intelligence(text_data['items'], analysis_results)
            
            # Analyze player stats
            if 'player_stats' in text_data:
                self._analyze_stats_intelligence(text_data['player_stats'], analysis_results)
            
            # Generate contextual recommendations
            self._generate_recommendations(analysis_results)
            
            # Update context memory
            self._update_context_memory(analysis_results)
            
            # Calculate processing time
            analysis_results['processing_time'] = time.time() - start_time
            
            logger.debug(f"Text intelligence analysis completed in {analysis_results['processing_time']:.3f}s")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Text intelligence analysis failed: {e}")
            return {
                'timestamp': start_time,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _analyze_chat_intelligence(self, messages: List[ChatMessage], results: Dict[str, Any]) -> None:
        """Analyze chat messages for intelligence"""
        xp_events = []
        combat_events = []
        trade_events = []
        quest_events = []
        social_events = []
        
        for msg in messages:
            message_text = msg.message
            
            # XP and leveling analysis
            xp_match = re.search(self.text_patterns['xp_patterns']['xp_drop'], message_text)
            if xp_match:
                xp_amount, skill = xp_match.groups()
                xp_event = XPEvent(
                    skill=skill.strip(),
                    xp_gained=int(xp_amount),
                    timestamp=msg.timestamp,
                    source=self._determine_xp_source(message_text)
                )
                xp_events.append(xp_event)
                self.xp_events.append(xp_event)
            
            # Level up detection
            level_match = re.search(self.text_patterns['xp_patterns']['level_up'], message_text)
            if level_match:
                skill = level_match.group(1).strip()
                results['alerts'].append({
                    'type': 'level_up',
                    'skill': skill,
                    'message': f"Level up in {skill}!",
                    'priority': TextPriority.HIGH,
                    'timestamp': msg.timestamp
                })
            
            # Combat analysis
            for pattern_name, pattern in self.text_patterns['combat_patterns'].items():
                match = re.search(pattern, message_text)
                if match:
                    combat_events.append({
                        'type': pattern_name,
                        'data': match.groups(),
                        'timestamp': msg.timestamp,
                        'message': message_text
                    })
            
            # Trading analysis
            for pattern_name, pattern in self.text_patterns['trading_patterns'].items():
                match = re.search(pattern, message_text)
                if match:
                    trade_events.append({
                        'type': pattern_name,
                        'data': match.groups(),
                        'timestamp': msg.timestamp,
                        'message': message_text
                    })
            
            # Quest analysis
            for pattern_name, pattern in self.text_patterns['quest_patterns'].items():
                match = re.search(pattern, message_text)
                if match:
                    quest_events.append({
                        'type': pattern_name,
                        'data': match.groups(),
                        'timestamp': msg.timestamp,
                        'message': message_text
                    })
            
            # Social analysis
            if not msg.is_system:
                social_events.append({
                    'player': msg.player_name,
                    'message': msg.message,
                    'type': msg.chat_type,
                    'timestamp': msg.timestamp,
                    'importance': self._assess_message_importance(msg.message)
                })
        
        # Update results
        results['xp_analysis'] = {
            'events': xp_events,
            'skill_rates': self._calculate_xp_rates(),
            'session_xp': self._calculate_session_xp()
        }
        
        results['combat_analysis'] = {
            'events': combat_events,
            'combat_stats': self._analyze_combat_performance(combat_events)
        }
        
        results['market_analysis'] = {
            'trade_events': trade_events,
            'market_opportunities': self._identify_market_opportunities(trade_events)
        }
        
        results['quest_analysis'] = {
            'events': quest_events,
            'context': self._update_quest_context(quest_events)
        }
        
        results['social_analysis'] = {
            'events': social_events,
            'player_interactions': self._analyze_player_interactions(social_events)
        }
    
    def _analyze_item_intelligence(self, items: List[ItemInfo], results: Dict[str, Any]) -> None:
        """Analyze item information for intelligence"""
        valuable_items = []
        total_value = 0
        
        for item in items:
            if item.is_valuable and item.ge_price:
                valuable_items.append(item)
                total_value += item.ge_price * item.quantity
        
        # Update market analysis
        market_analysis = results.setdefault('market_analysis', {})
        market_analysis['inventory_value'] = total_value
        market_analysis['valuable_items'] = valuable_items
        
        # Generate alerts for high-value items
        for item in valuable_items:
            if item.ge_price and item.ge_price > 50000:  # High-value threshold
                results['alerts'].append({
                    'type': 'valuable_item',
                    'item': item.name,
                    'value': item.ge_price,
                    'message': f"High-value item detected: {item.name} ({item.ge_price:,} gp)",
                    'priority': TextPriority.MEDIUM,
                    'timestamp': time.time()
                })
    
    def _analyze_stats_intelligence(self, stats: PlayerStats, results: Dict[str, Any]) -> None:
        """Analyze player stats for intelligence"""
        # Health analysis
        if stats.health_current and stats.health_max:
            health_percent = (stats.health_current / stats.health_max) * 100
            
            if health_percent < 20:
                results['alerts'].append({
                    'type': 'low_health',
                    'health_percent': health_percent,
                    'message': f"Critical health: {health_percent:.1f}%",
                    'priority': TextPriority.CRITICAL,
                    'timestamp': time.time()
                })
            elif health_percent < 50:
                results['alerts'].append({
                    'type': 'moderate_health',
                    'health_percent': health_percent,
                    'message': f"Low health: {health_percent:.1f}%",
                    'priority': TextPriority.HIGH,
                    'timestamp': time.time()
                })
        
        # Prayer analysis
        if stats.prayer_current and stats.prayer_max:
            prayer_percent = (stats.prayer_current / stats.prayer_max) * 100
            
            if prayer_percent < 10:
                results['alerts'].append({
                    'type': 'low_prayer',
                    'prayer_percent': prayer_percent,
                    'message': f"Prayer almost depleted: {prayer_percent:.1f}%",
                    'priority': TextPriority.HIGH,
                    'timestamp': time.time()
                })
    
    def _determine_xp_source(self, message: str) -> str:
        """Determine the source of XP gain"""
        combat_keywords = ['hit', 'damage', 'attack', 'fight', 'kill']
        skilling_keywords = ['mine', 'fish', 'cook', 'craft', 'cut', 'make']
        
        message_lower = message.lower()
        
        if any(keyword in message_lower for keyword in combat_keywords):
            return 'combat'
        elif any(keyword in message_lower for keyword in skilling_keywords):
            return 'skilling'
        else:
            return 'unknown'
    
    def _calculate_xp_rates(self) -> Dict[str, float]:
        """Calculate XP rates per hour for each skill"""
        current_time = time.time()
        hour_ago = current_time - 3600  # 1 hour ago
        
        skill_xp = {}
        
        # Sum XP gained in the last hour for each skill
        for event in self.xp_events:
            if event.timestamp >= hour_ago:
                if event.skill not in skill_xp:
                    skill_xp[event.skill] = 0
                skill_xp[event.skill] += event.xp_gained
        
        # Calculate rates (XP per hour)
        time_factor = 3600 / (current_time - max(hour_ago, self.session_start))
        
        rates = {}
        for skill, xp in skill_xp.items():
            rates[skill] = xp * time_factor
        
        return rates
    
    def _calculate_session_xp(self) -> Dict[str, int]:
        """Calculate total XP gained this session"""
        session_xp = {}
        
        for event in self.xp_events:
            if event.timestamp >= self.session_start:
                if event.skill not in session_xp:
                    session_xp[event.skill] = 0
                session_xp[event.skill] += event.xp_gained
        
        return session_xp
    
    def _analyze_combat_performance(self, combat_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze combat performance from events"""
        damage_dealt = 0
        damage_taken = 0
        heals_used = 0
        
        for event in combat_events:
            if event['type'] == 'damage_dealt':
                damage_dealt += int(event['data'][0])
            elif event['type'] == 'damage_taken':
                damage_taken += int(event['data'][0])
            elif event['type'] == 'heal':
                heals_used += int(event['data'][0])
        
        # Calculate combat efficiency
        if damage_taken > 0:
            efficiency = damage_dealt / damage_taken
        else:
            efficiency = float('inf') if damage_dealt > 0 else 0
        
        return {
            'damage_dealt': damage_dealt,
            'damage_taken': damage_taken,
            'heals_used': heals_used,
            'efficiency': efficiency,
            'net_damage': damage_dealt - damage_taken
        }
    
    def _identify_market_opportunities(self, trade_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify market trading opportunities"""
        opportunities = []
        
        for event in trade_events:
            if event['type'] == 'price_check':
                item_name, price_str = event['data']
                price = int(price_str.replace(',', ''))
                
                # Check if this is a good deal compared to known prices
                if item_name in self.market_data:
                    market_price = self.market_data[item_name].current_price
                    if price < market_price * 0.8:  # 20% below market
                        opportunities.append({
                            'type': 'buy_opportunity',
                            'item': item_name,
                            'offered_price': price,
                            'market_price': market_price,
                            'potential_profit': market_price - price,
                            'confidence': 0.8
                        })
        
        return opportunities
    
    def _update_quest_context(self, quest_events: List[Dict[str, Any]]) -> QuestContext:
        """Update quest context from events"""
        for event in quest_events:
            if event['type'] == 'quest_start':
                self.quest_context.quest_name = event['data'][0]
                self.quest_context.dialogue_stage = 0
            elif event['type'] == 'npc_dialogue':
                self.quest_context.npc_name = event['data'][0]
                self.quest_context.dialogue_stage += 1
        
        return self.quest_context
    
    def _analyze_player_interactions(self, social_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze player social interactions"""
        player_stats = {}
        
        for event in social_events:
            player = event['player']
            if player not in player_stats:
                player_stats[player] = {
                    'message_count': 0,
                    'last_seen': 0,
                    'interaction_type': 'neutral'
                }
            
            player_stats[player]['message_count'] += 1
            player_stats[player]['last_seen'] = event['timestamp']
            
            # Analyze interaction type
            message = event['message'].lower()
            if any(word in message for word in ['trade', 'buy', 'sell']):
                player_stats[player]['interaction_type'] = 'trading'
            elif any(word in message for word in ['help', 'assist', 'guide']):
                player_stats[player]['interaction_type'] = 'helpful'
            elif any(word in message for word in ['scam', 'spam', 'bot']):
                player_stats[player]['interaction_type'] = 'suspicious'
        
        return player_stats
    
    def _assess_message_importance(self, message: str) -> str:
        """Assess the importance of a chat message"""
        message_lower = message.lower()
        
        critical_keywords = ['die', 'death', 'attack', 'help', 'emergency']
        high_keywords = ['trade', 'buy', 'sell', 'rare', 'drop', 'level']
        medium_keywords = ['question', 'where', 'how', 'what']
        
        if any(keyword in message_lower for keyword in critical_keywords):
            return 'critical'
        elif any(keyword in message_lower for keyword in high_keywords):
            return 'high'
        elif any(keyword in message_lower for keyword in medium_keywords):
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> None:
        """Generate intelligent recommendations based on analysis"""
        recommendations = []
        
        # XP recommendations
        if 'xp_analysis' in results:
            rates = results['xp_analysis'].get('skill_rates', {})
            for skill, rate in rates.items():
                if rate < 10000:  # Low XP rate
                    recommendations.append({
                        'type': 'xp_efficiency',
                        'skill': skill,
                        'message': f"Consider more efficient {skill} training methods",
                        'priority': TextPriority.MEDIUM
                    })
        
        # Health recommendations
        alerts = results.get('alerts', [])
        for alert in alerts:
            if alert['type'] == 'low_health':
                recommendations.append({
                    'type': 'healing',
                    'message': "Consider eating food or using healing items",
                    'priority': TextPriority.HIGH
                })
            elif alert['type'] == 'low_prayer':
                recommendations.append({
                    'type': 'prayer',
                    'message': "Consider drinking prayer potions",
                    'priority': TextPriority.HIGH
                })
        
        # Market recommendations
        if 'market_analysis' in results:
            opportunities = results['market_analysis'].get('market_opportunities', [])
            for opp in opportunities:
                if opp['type'] == 'buy_opportunity':
                    recommendations.append({
                        'type': 'trading',
                        'message': f"Good buying opportunity for {opp['item']}",
                        'priority': TextPriority.MEDIUM
                    })
        
        results['recommendations'] = recommendations
    
    def _update_context_memory(self, results: Dict[str, Any]) -> None:
        """Update context memory with new information"""
        current_time = time.time()
        
        # Update XP tracking
        if 'xp_analysis' in results:
            self.context_memory['last_xp_update'] = current_time
            self.context_memory['recent_skills'] = list(results['xp_analysis'].get('skill_rates', {}).keys())
        
        # Update combat context
        if 'combat_analysis' in results:
            self.context_memory['in_combat'] = len(results['combat_analysis'].get('events', [])) > 0
            self.context_memory['last_combat_update'] = current_time
    
    def _load_intelligence_data(self) -> None:
        """Load historical intelligence data"""
        try:
            # Load XP history
            xp_file = self.data_dir / "xp_history.json"
            if xp_file.exists():
                with open(xp_file, 'r') as f:
                    xp_data = json.load(f)
                    # Convert to XPEvent objects
                    for event_data in xp_data:
                        event = XPEvent(**event_data)
                        self.xp_events.append(event)
            
            # Load market data
            market_file = self.data_dir / "market_data.json"
            if market_file.exists():
                with open(market_file, 'r') as f:
                    market_data = json.load(f)
                    # Convert to MarketAnalysis objects
                    for item_name, data in market_data.items():
                        analysis = MarketAnalysis(**data)
                        self.market_data[item_name] = analysis
                        
        except Exception as e:
            logger.error(f"Failed to load intelligence data: {e}")
    
    def save_intelligence_data(self) -> None:
        """Save intelligence data to disk"""
        try:
            # Save XP history
            xp_data = []
            for event in self.xp_events[-1000:]:  # Keep last 1000 events
                xp_data.append(asdict(event))
            
            xp_file = self.data_dir / "xp_history.json"
            with open(xp_file, 'w') as f:
                json.dump(xp_data, f, indent=2)
            
            # Save market data
            market_data = {}
            for item_name, analysis in self.market_data.items():
                market_data[item_name] = asdict(analysis)
            
            market_file = self.data_dir / "market_data.json"
            with open(market_file, 'w') as f:
                json.dump(market_data, f, indent=2)
                
            logger.info("Intelligence data saved")
            
        except Exception as e:
            logger.error(f"Failed to save intelligence data: {e}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        current_time = time.time()
        session_duration = current_time - self.session_start
        
        # Calculate session XP
        session_xp = self._calculate_session_xp()
        total_session_xp = sum(session_xp.values())
        
        # Calculate XP rates
        xp_rates = self._calculate_xp_rates()
        
        return {
            'session_duration': session_duration,
            'session_xp': session_xp,
            'total_session_xp': total_session_xp,
            'xp_rates': xp_rates,
            'avg_xp_per_hour': total_session_xp / (session_duration / 3600) if session_duration > 0 else 0,
            'skills_trained': len(session_xp),
            'most_trained_skill': max(session_xp, key=lambda x: session_xp[x]) if session_xp else None
        }
    
    def cleanup(self) -> None:
        """Clean up and save data"""
        self.save_intelligence_data()
        logger.info("Text Intelligence Core cleaned up")


# Initialize global instance
text_intelligence = OSRSTextIntelligenceCore()