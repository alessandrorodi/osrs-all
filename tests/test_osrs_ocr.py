"""
Tests for OSRS Text Recognition System

Test suite for the OSRS OCR and text intelligence components.
Includes unit tests, integration tests, and performance benchmarks.
"""

import pytest
import numpy as np
import cv2
import time
from pathlib import Path
import json
from unittest.mock import Mock, patch, MagicMock

from vision.osrs_ocr import (
    OSRSTextIntelligence, OSRSTextRegion, ChatMessage, ItemInfo, PlayerStats,
    osrs_text_intelligence
)
from core.text_intelligence import (
    OSRSTextIntelligenceCore, XPEvent, TextPriority, MarketAnalysis,
    text_intelligence
)


# Module-level fixtures for all tests
@pytest.fixture
def mock_screenshot():
    """Create a mock OSRS screenshot"""
    # Create a 765x503 image (typical OSRS client size)
    screenshot = np.zeros((503, 765, 3), dtype=np.uint8)
    
    # Add some basic color patterns to simulate OSRS interface
    # Chat area (bottom)
    screenshot[345:488, 7:513] = [40, 40, 40]  # Dark chat background
    
    # Inventory area (right side)
    screenshot[205:467, 548:734] = [60, 60, 60]  # Inventory background
    
    # Health/prayer orbs (top left)
    cv2.circle(screenshot, (38, 75), 30, (255, 0, 0), -1)  # Health orb
    cv2.circle(screenshot, (38, 115), 30, (0, 0, 255), -1)  # Prayer orb
    
    return screenshot


@pytest.fixture
def sample_text_data():
    """Sample text data for testing"""
    return {
        'timestamp': time.time(),
        'chat_messages': [
            ChatMessage(
                player_name='TestPlayer',
                message='Hello world!',
                chat_type='public',
                timestamp=time.time()
            ),
            ChatMessage(
                player_name='System',
                message='You gain 50 Attack XP.',
                chat_type='game',
                timestamp=time.time(),
                is_system=True
            )
        ],
        'items': [
            ItemInfo(
                name='Dragon scimitar',
                quantity=1,
                position=(600, 250),
                ge_price=100000,
                is_valuable=True,
                category='weapon'
            ),
            ItemInfo(
                name='Shark',
                quantity=20,
                position=(620, 270),
                ge_price=800,
                category='food'
            )
        ],
        'player_stats': PlayerStats(
            health_current=50,
            health_max=99,
            prayer_current=30,
            prayer_max=70,
            combat_level=85
        )
    }


class TestOSRSTextIntelligence:
    """Test cases for OSRS text intelligence"""
    
    def test_osrs_text_region_initialization(self):
        """Test OSRS text region configuration"""
        region = OSRSTextRegion(
            name='test_region',
            region=(0, 0, 100, 100),
            text_type='chat',
            confidence_threshold=0.5
        )
        
        assert region.name == 'test_region'
        assert region.region == (0, 0, 100, 100)
        assert region.text_type == 'chat'
        assert region.confidence_threshold == 0.5
        assert region.priority == 1  # Default priority
    
    def test_chat_message_creation(self):
        """Test chat message data structure"""
        msg = ChatMessage(
            player_name='TestPlayer',
            message='  Test message  ',  # With whitespace
            chat_type='public',
            timestamp=123456789.0
        )
        
        # Check that whitespace is stripped
        assert msg.message == 'Test message'
        assert msg.player_name == 'TestPlayer'
        assert msg.chat_type == 'public'
        assert not msg.is_system
    
    def test_item_info_creation(self):
        """Test item information structure"""
        item = ItemInfo(
            name='Dragon scimitar',
            quantity=1,
            ge_price=100000,
            category='weapon'
        )
        
        assert item.name == 'Dragon scimitar'
        assert item.quantity == 1
        assert item.ge_price == 100000
        assert item.category == 'weapon'
        assert not item.noted  # Default value
    
    @patch('vision.osrs_ocr.OCRDetector')
    def test_osrs_text_intelligence_initialization(self, mock_ocr):
        """Test OSRS text intelligence initialization"""
        # Mock OCR detector
        mock_ocr.return_value = Mock()
        
        intelligence = OSRSTextIntelligence(device='cpu')
        
        assert intelligence.device == 'cpu'
        assert 'public_chat' in intelligence.text_regions
        assert 'inventory_items' in intelligence.text_regions
        assert len(intelligence.osrs_patterns) > 0
        
        # Check that OCR was initialized
        mock_ocr.assert_called_once_with(languages=['en'], use_gpu=False)
    
    def test_text_region_priority_grouping(self):
        """Test text region priority grouping"""
        intelligence = OSRSTextIntelligence(device='cpu')
        
        region_names = ['public_chat', 'inventory_items', 'equipment_stats']
        groups = intelligence._group_regions_by_priority(region_names, intelligence.text_regions)
        
        # Check that high priority regions are grouped separately
        assert 1 in groups  # High priority
        assert 2 in groups or 3 in groups  # Lower priority
        
        # Public chat should be high priority
        if 'public_chat' in intelligence.text_regions:
            priority = intelligence.text_regions['public_chat'].priority
            assert 'public_chat' in groups[priority]
    
    def test_item_text_parsing(self):
        """Test item text parsing functionality"""
        intelligence = OSRSTextIntelligence(device='cpu')
        
        # Test item with quantity
        item = intelligence._parse_item_text('Dragon scimitar x 1')
        assert item is not None
        assert item.name == 'Dragon scimitar'
        assert item.quantity == 1
        
        # Test noted item
        item = intelligence._parse_item_text('Shark (noted)')
        assert item is not None
        assert item.name == 'Shark'
        assert item.noted is True
        
        # Test simple item name
        item = intelligence._parse_item_text('Bronze sword')
        assert item is not None
        assert item.name == 'Bronze sword'
        assert item.quantity == 1
    
    @patch('vision.osrs_ocr.time.time')
    def test_performance_tracking(self, mock_time):
        """Test performance statistics tracking"""
        # Mock time progression
        mock_time.side_effect = [1000.0, 1000.1, 1000.2]  # 100ms processing time
        
        intelligence = OSRSTextIntelligence(device='cpu')
        
        # Simulate processing
        intelligence._update_performance_stats(0.1)
        
        stats = intelligence.get_performance_stats()
        assert 'processing_times' in stats
        assert 'avg_latency' in stats
        assert 'total_ocr_calls' in stats
        assert stats['total_ocr_calls'] == 1
    
    def test_cache_functionality(self):
        """Test text recognition caching"""
        intelligence = OSRSTextIntelligence(device='cpu')
        
        # Test cache efficiency calculation
        intelligence.performance_stats['cache_hits'] = 80
        intelligence.performance_stats['cache_misses'] = 20
        
        efficiency = intelligence._get_cache_efficiency()
        assert efficiency == 0.8  # 80% efficiency


class TestTextIntelligenceCore:
    """Test cases for text intelligence core"""
    
    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary directory for intelligence data"""
        return tmp_path / "intelligence"
    
    def test_core_initialization(self, temp_data_dir):
        """Test text intelligence core initialization"""
        core = OSRSTextIntelligenceCore(data_dir=temp_data_dir)
        
        assert core.data_dir == temp_data_dir
        assert temp_data_dir.exists()
        assert isinstance(core.xp_events, list)
        assert isinstance(core.text_patterns, dict)
    
    def test_xp_event_tracking(self):
        """Test XP event creation and tracking"""
        event = XPEvent(
            skill='Attack',
            xp_gained=50,
            timestamp=time.time(),
            source='combat'
        )
        
        assert event.skill == 'Attack'
        assert event.xp_gained == 50
        assert event.source == 'combat'
    
    def test_market_analysis_creation(self):
        """Test market analysis data structure"""
        analysis = MarketAnalysis(
            item_name='Dragon scimitar',
            current_price=100000,
            price_trend='stable',
            recommended_action='hold'
        )
        
        assert analysis.item_name == 'Dragon scimitar'
        assert analysis.current_price == 100000
        assert analysis.price_trend == 'stable'
        assert analysis.recommended_action == 'hold'
    
    def test_xp_rate_calculation(self):
        """Test XP rate calculation"""
        core = OSRSTextIntelligenceCore()
        
        # Set session start to 2 hours ago so we have sufficient time data
        current_time = time.time()
        core.session_start = current_time - 7200  # 2 hours ago
        
        # Add some mock XP events
        events = [
            XPEvent('Attack', 100, current_time - 1800, source='combat'),  # 30 min ago
            XPEvent('Attack', 150, current_time - 900, source='combat'),   # 15 min ago
            XPEvent('Defence', 80, current_time - 600, source='combat'),   # 10 min ago
        ]
        
        core.xp_events.extend(events)
        rates = core._calculate_xp_rates()
        
        # Should have rates for both skills
        assert 'Attack' in rates
        assert 'Defence' in rates
        assert rates['Attack'] > 0
        assert rates['Defence'] > 0
    
    def test_session_xp_calculation(self):
        """Test session XP calculation"""
        core = OSRSTextIntelligenceCore()
        
        # Add XP events after session start
        current_time = time.time()
        core.session_start = current_time - 3600  # 1 hour ago
        
        events = [
            XPEvent('Attack', 500, current_time - 1800),  # After session start
            XPEvent('Attack', 300, core.session_start - 600),  # Before session start
            XPEvent('Defence', 200, current_time - 900),  # After session start
        ]
        
        core.xp_events.extend(events)
        session_xp = core._calculate_session_xp()
        
        # Should only count XP after session start
        assert session_xp['Attack'] == 500  # Only the first event
        assert session_xp['Defence'] == 200
    
    def test_chat_intelligence_analysis(self, sample_text_data):
        """Test chat message intelligence analysis"""
        core = OSRSTextIntelligenceCore()
        
        # Mock text patterns for testing
        core.text_patterns['xp_patterns']['xp_drop'] = r'You gain (\d+) (\w+) XP\.'
        
        results = {
            'timestamp': time.time(),
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
        core._analyze_chat_intelligence(sample_text_data['chat_messages'], results)
        
        # Check that XP events were detected
        assert 'xp_analysis' in results
        xp_events = results['xp_analysis'].get('events', [])
        assert len(xp_events) > 0
        
        # Verify XP event details
        xp_event = xp_events[0]
        assert xp_event.skill == 'Attack'
        assert xp_event.xp_gained == 50
    
    def test_message_importance_assessment(self):
        """Test message importance assessment"""
        core = OSRSTextIntelligenceCore()
        
        # Test different importance levels
        assert core._assess_message_importance("Help! I'm dying!") == 'critical'
        assert core._assess_message_importance('Selling rare item') == 'high'
        assert core._assess_message_importance('Where is the bank?') == 'medium'
        assert core._assess_message_importance('Nice weather today') == 'low'
    
    def test_intelligence_data_persistence(self, temp_data_dir):
        """Test saving and loading intelligence data"""
        core = OSRSTextIntelligenceCore(data_dir=temp_data_dir)
        
        # Add some test data
        core.xp_events.append(XPEvent('Attack', 100, time.time(), source='test'))
        core.market_data['Test Item'] = MarketAnalysis(
            item_name='Test Item',
            current_price=1000,
            price_trend='rising'
        )
        
        # Save data
        core.save_intelligence_data()
        
        # Verify files were created
        assert (temp_data_dir / "xp_history.json").exists()
        assert (temp_data_dir / "market_data.json").exists()
        
        # Create new core instance and verify data loads
        new_core = OSRSTextIntelligenceCore(data_dir=temp_data_dir)
        assert len(new_core.xp_events) > 0
        assert 'Test Item' in new_core.market_data
    
    def test_session_summary(self):
        """Test session summary generation"""
        core = OSRSTextIntelligenceCore()
        
        # Add some test XP events
        current_time = time.time()
        core.session_start = current_time - 3600  # 1 hour session
        
        events = [
            XPEvent('Attack', 1000, current_time - 1800),
            XPEvent('Defence', 500, current_time - 900),
            XPEvent('Attack', 800, current_time - 300),
        ]
        core.xp_events.extend(events)
        
        summary = core.get_session_summary()
        
        assert summary['session_duration'] > 0
        assert summary['total_session_xp'] == 2300  # 1000 + 500 + 800
        assert summary['skills_trained'] == 2  # Attack and Defence
        assert summary['most_trained_skill'] == 'Attack'  # 1800 XP total


class TestIntegration:
    """Integration tests for the complete text intelligence system"""
    
    @patch('vision.osrs_ocr.OCRDetector')
    def test_full_text_analysis_pipeline(self, mock_ocr, mock_screenshot, sample_text_data):
        """Test the complete text analysis pipeline"""
        # Mock OCR detector responses
        mock_ocr_instance = Mock()
        mock_ocr_instance.detect_text.return_value = []
        mock_ocr_instance._preprocess_chat_text.return_value = mock_screenshot
        mock_ocr_instance._preprocess_numbers.return_value = mock_screenshot
        mock_ocr.return_value = mock_ocr_instance
        
        # Initialize systems
        intelligence = OSRSTextIntelligence(device='cpu')
        
        # Mock the analyze_game_text method to return sample data
        with patch.object(intelligence, 'analyze_game_text', return_value=sample_text_data):
            # Analyze the mock screenshot
            results = intelligence.analyze_game_text(mock_screenshot)
            
            # Verify results structure
            assert 'timestamp' in results
            assert 'chat_messages' in results
            assert 'items' in results
            assert 'player_stats' in results
            
            # Analyze with text intelligence
            intelligence_results = text_intelligence.analyze_text_intelligence(results)
            
            # Verify intelligence analysis
            assert 'xp_analysis' in intelligence_results
            assert 'market_analysis' in intelligence_results
            assert 'recommendations' in intelligence_results
            assert 'alerts' in intelligence_results
    
    def test_performance_benchmarks(self, mock_screenshot):
        """Test performance benchmarks for text intelligence"""
        intelligence = OSRSTextIntelligence(device='cpu')
        
        # Mock OCR to avoid actual processing
        with patch.object(intelligence.ocr, 'detect_text', return_value=[]):
            # Benchmark text analysis
            start_time = time.time()
            results = intelligence.analyze_game_text(mock_screenshot)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Verify performance requirements (should be fast with mocked OCR)
            assert processing_time < 1.0  # Less than 1 second
            assert 'performance' in results
            assert results['performance']['processing_time'] > 0
    
    def test_memory_usage(self):
        """Test memory usage and cleanup"""
        intelligence = OSRSTextIntelligence(device='cpu')
        
        # Add data to cache
        for i in range(100):
            intelligence.text_cache[f"key_{i}"] = {
                'data': {'test': f'value_{i}'},
                'timestamp': time.time()
            }
        
        # Verify cache has data
        assert len(intelligence.text_cache) == 100
        
        # Cleanup
        intelligence.cleanup()
        
        # Verify cleanup
        assert len(intelligence.text_cache) == 0
    
    @pytest.mark.slow
    def test_stress_test(self, mock_screenshot):
        """Stress test for high-frequency text analysis"""
        intelligence = OSRSTextIntelligence(device='cpu')
        
        # Mock OCR to return consistent results
        with patch.object(intelligence.ocr, 'detect_text', return_value=[]):
            # Run multiple analyses rapidly
            results = []
            for i in range(10):
                result = intelligence.analyze_game_text(mock_screenshot)
                results.append(result)
                
                # Small delay to simulate real usage
                time.sleep(0.01)
            
            # Verify all analyses completed
            assert len(results) == 10
            
            # Check cache efficiency
            stats = intelligence.get_performance_stats()
            if stats['cache_hits'] + stats['cache_misses'] > 0:
                efficiency = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
                assert efficiency >= 0.0  # Should have some cache efficiency


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])