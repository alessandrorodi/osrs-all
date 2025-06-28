#!/usr/bin/env python3
"""
Example Bot Implementation

This demonstrates how to create a bot using the OSRS Bot Framework.
This example bot performs basic combat actions.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.bot_base import BotBase, BotState
from core.screen_capture import screen_capture
from core.automation import mouse, keyboard, random_delay
from core.computer_vision import cv_system
from utils.logging import BotLogger


class ExampleCombatBot(BotBase):
    """
    Example combat bot that demonstrates framework usage
    
    This bot will:
    1. Look for monsters to attack
    2. Click on them to initiate combat
    3. Monitor health and eat food when needed
    4. Loot items after combat
    """
    
    def __init__(self, config=None):
        super().__init__("ExampleCombatBot", config)
        
        self.bot_logger = BotLogger("example_combat")
        
        # Bot-specific configuration
        self.target_monster = self.config.get("target_monster", "goblin")
        self.food_item = self.config.get("food_item", "bread")
        self.health_threshold = self.config.get("health_threshold", 50)
        
        # State tracking
        self.current_target = None
        self.last_health_check = 0
        self.last_food_time = 0
        self.combat_timeout = 10.0  # seconds
        
        # Performance tracking
        self.monsters_killed = 0
        self.food_consumed = 0
        
        self.bot_logger.log_action(f"Bot configured", 
                                 f"Target: {self.target_monster}, Food: {self.food_item}")
    
    def initialize(self) -> bool:
        """Initialize bot-specific components"""
        try:
            self.bot_logger.log_action("Initializing bot")
            
            # Check if required templates exist
            required_templates = [
                self.target_monster,
                self.food_item,
                "health_bar"
            ]
            
            missing_templates = []
            for template in required_templates:
                if not cv_system.template_manager.get_template(template):
                    missing_templates.append(template)
            
            if missing_templates:
                self.bot_logger.log_error(f"Missing templates: {', '.join(missing_templates)}")
                print(f"\nMissing required templates: {', '.join(missing_templates)}")
                print("Please use the template creator tool to create these templates:")
                print("python tools/template_creator.py")
                return False
            
            # Verify client is ready
            if not screen_capture.is_client_active():
                self.bot_logger.log_error("OSRS client not active")
                return False
            
            self.bot_logger.log_action("Bot initialization successful")
            return True
            
        except Exception as e:
            self.bot_logger.log_error("Initialization failed", e)
            return False
    
    def execute_cycle(self) -> bool:
        """Execute one cycle of bot logic"""
        try:
            # Capture and process current game state
            game_state = self.capture_and_process()
            if not game_state:
                return False
            
            # Check health and eat food if needed
            if self._should_eat_food():
                if self._eat_food():
                    self.food_consumed += 1
                    random_delay(1.0, 2.0)  # Wait after eating
            
            # Combat logic
            if not self._is_in_combat():
                # Look for monsters to attack
                if self._find_and_attack_monster():
                    self.bot_logger.log_action(f"Initiated combat with {self.target_monster}")
                    self.current_target = time.time()
            else:
                # Monitor ongoing combat
                self._handle_combat()
            
            return True
            
        except Exception as e:
            self.bot_logger.log_error("Cycle execution failed", e)
            return False
    
    def _should_eat_food(self) -> bool:
        """Check if bot should eat food"""
        # Simple time-based eating (in real implementation, check actual health)
        current_time = time.time()
        
        # Don't eat too frequently
        if current_time - self.last_food_time < 5.0:
            return False
        
        # Simulate health check (in real bot, detect health bar level)
        # For demo purposes, eat food every 30 seconds
        return current_time - self.last_food_time > 30.0
    
    def _eat_food(self) -> bool:
        """Attempt to eat food"""
        try:
            self.bot_logger.log_action(f"Attempting to eat {self.food_item}")
            
            # Find food in inventory (simplified - click in inventory area)
            # In real implementation, find food template in inventory region
            success = self.find_and_click(self.food_item, confidence=0.7, timeout=2.0)
            
            if success:
                self.last_food_time = time.time()
                self.bot_logger.log_action("Food consumed")
                return True
            else:
                self.bot_logger.log_error("No food found")
                return False
                
        except Exception as e:
            self.bot_logger.log_error("Failed to eat food", e)
            return False
    
    def _find_and_attack_monster(self) -> bool:
        """Find and attack a monster"""
        try:
            self.bot_logger.log_action(f"Looking for {self.target_monster}")
            
            # Find monster template
            success = self.find_and_click(self.target_monster, confidence=0.8, timeout=3.0)
            
            if success:
                self.current_target = time.time()
                self.bot_logger.log_action(f"Clicked on {self.target_monster}")
                return True
            else:
                self.bot_logger.log_action(f"No {self.target_monster} found")
                return False
                
        except Exception as e:
            self.bot_logger.log_error("Monster targeting failed", e)
            return False
    
    def _is_in_combat(self) -> bool:
        """Check if currently in combat"""
        if not self.current_target:
            return False
        
        # Simple timeout-based combat detection
        combat_time = time.time() - self.current_target
        return combat_time < self.combat_timeout
    
    def _handle_combat(self) -> bool:
        """Handle ongoing combat"""
        try:
            combat_time = time.time() - self.current_target
            
            # Check if combat is taking too long
            if combat_time > self.combat_timeout:
                self.bot_logger.log_action("Combat timeout, looking for new target")
                self.current_target = None
                self.monsters_killed += 1  # Assume monster was killed
                return True
            
            # Monitor for loot or combat end
            # In real implementation, check for combat indicators
            self.bot_logger.log_action(f"In combat ({combat_time:.1f}s)")
            
            return True
            
        except Exception as e:
            self.bot_logger.log_error("Combat handling failed", e)
            return False
    
    def get_status(self) -> dict:
        """Get extended bot status"""
        status = super().get_status()
        status.update({
            "monsters_killed": self.monsters_killed,
            "food_consumed": self.food_consumed,
            "in_combat": self._is_in_combat(),
            "target_monster": self.target_monster
        })
        return status
    
    def cleanup(self):
        """Clean up when bot stops"""
        super().cleanup()
        
        # Log final statistics
        stats = self.get_status()
        self.bot_logger.log_performance(stats)
        
        print(f"\n{self.name} Final Statistics:")
        print(f"  Runtime: {stats['runtime']:.1f} seconds")
        print(f"  Actions: {stats['actions']}")
        print(f"  Monsters killed: {self.monsters_killed}")
        print(f"  Food consumed: {self.food_consumed}")
        print(f"  Success rate: {stats['performance']['success_rate']:.1f}%")


def main():
    """Main function to run the example bot"""
    print("OSRS Example Combat Bot")
    print("=" * 30)
    
    # Configuration
    config = {
        "target_monster": "goblin",  # Template name for target monster
        "food_item": "bread",        # Template name for food
        "health_threshold": 50,      # Health percentage to eat at
        "require_focus": True        # Require client to be focused
    }
    
    # Create and configure bot
    bot = ExampleCombatBot(config)
    
    try:
        print(f"\nStarting {bot.name}...")
        print("Press Ctrl+C to stop the bot")
        
        if bot.start(threaded=False):  # Run in main thread for demo
            print(f"Bot stopped normally")
        else:
            print(f"Bot failed to start")
            
    except KeyboardInterrupt:
        print(f"\nStopping bot...")
        bot.stop()
        print(f"Bot stopped by user")
    
    except Exception as e:
        print(f"Bot error: {e}")
        bot.emergency_stop()
    
    finally:
        # Show final status
        status = bot.get_status()
        print(f"\nFinal Status: {status}")


if __name__ == "__main__":
    main() 