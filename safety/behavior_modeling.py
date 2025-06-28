#!/usr/bin/env python3
"""
OSRS Bot Framework - AI Behavior Modeling System

Sophisticated behavior modeling using AI to simulate human-like patterns and avoid detection.
Leverages RTX 4090 for real-time behavioral analysis and pattern recognition.
"""

import numpy as np
import time
import random
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque
import logging

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import scipy.stats as stats
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False
    torch = None
    nn = None

from utils.logging import get_logger
from config.settings import SAFETY, PROJECT_ROOT

logger = get_logger(__name__)


@dataclass
class MouseMovement:
    """Represents a mouse movement event"""
    start_pos: Tuple[int, int]
    end_pos: Tuple[int, int]
    duration: float
    timestamp: float
    acceleration: float = 0.0
    jitter: float = 0.0
    bezier_points: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class ActionEvent:
    """Represents a user action event"""
    action_type: str
    timestamp: float
    duration: float
    position: Tuple[int, int]
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorProfile:
    """Represents a human behavior profile"""
    user_id: str
    mouse_speed_avg: float
    mouse_speed_std: float
    reaction_time_avg: float
    reaction_time_std: float
    break_frequency: float
    attention_span: float
    multitasking_tendency: float
    consistency_score: float
    risk_tolerance: float
    created_at: datetime = field(default_factory=datetime.now)


class HumanBehaviorModel:
    """AI-powered human behavior modeling system"""
    
    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)
        self.models_dir = PROJECT_ROOT / "data" / "behavior_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Behavior data storage
        self.mouse_movements: deque = deque(maxlen=10000)
        self.action_events: deque = deque(maxlen=5000)
        self.break_patterns: deque = deque(maxlen=1000)
        self.attention_events: deque = deque(maxlen=2000)
        
        # AI Models
        self.movement_model = None
        self.timing_model = None
        self.attention_model = None
        self.risk_model = None
        
        # Behavior analytics
        self.behavior_analyzer = BehaviorAnalyzer()
        self.pattern_detector = PatternDetector()
        self.anomaly_detector = AnomalyDetector()
        
        # Current behavior profile
        self.current_profile = None
        self.baseline_profile = None
        
        # Initialize models
        self._initialize_models()
        
        logger.info(f"Behavior modeling initialized on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Get the appropriate device for computation"""
        if device == "auto":
            if GPU_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _initialize_models(self):
        """Initialize AI models for behavior prediction"""
        if not GPU_AVAILABLE:
            logger.warning("GPU not available, using simplified behavior modeling")
            return
        
        try:
            # Mouse movement prediction model
            self.movement_model = MouseMovementPredictor().to(self.device)
            
            # Timing prediction model
            self.timing_model = TimingPredictor().to(self.device)
            
            # Attention modeling
            self.attention_model = AttentionModel().to(self.device)
            
            # Risk assessment model
            self.risk_model = RiskAssessmentModel().to(self.device)
            
            logger.info("AI behavior models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
            GPU_AVAILABLE = False
    
    def record_mouse_movement(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int], 
                            duration: float) -> None:
        """Record a mouse movement for behavior analysis"""
        movement = MouseMovement(
            start_pos=start_pos,
            end_pos=end_pos,
            duration=duration,
            timestamp=time.time()
        )
        
        # Calculate movement metrics
        distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        movement.acceleration = distance / (duration ** 2) if duration > 0 else 0
        movement.jitter = self._calculate_jitter(start_pos, end_pos)
        
        self.mouse_movements.append(movement)
        
        # Analyze in real-time if GPU available
        if GPU_AVAILABLE and len(self.mouse_movements) % 10 == 0:
            self._analyze_movement_patterns()
    
    def record_action(self, action_type: str, position: Tuple[int, int], 
                     duration: float, context: Optional[Dict[str, Any]] = None) -> None:
        """Record an action event for behavior analysis"""
        action = ActionEvent(
            action_type=action_type,
            timestamp=time.time(),
            duration=duration,
            position=position,
            context=context or {}
        )
        
        self.action_events.append(action)
        
        # Update attention tracking
        self._update_attention_tracking(action)
    
    def generate_human_movement(self, start_pos: Tuple[int, int], 
                              end_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Generate human-like mouse movement path"""
        if GPU_AVAILABLE and self.movement_model:
            return self._ai_generate_movement(start_pos, end_pos)
        else:
            return self._fallback_generate_movement(start_pos, end_pos)
    
    def get_next_action_timing(self, action_type: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Get human-like timing for next action"""
        if GPU_AVAILABLE and self.timing_model:
            return self._ai_predict_timing(action_type, context)
        else:
            return self._fallback_predict_timing(action_type)
    
    def should_take_break(self) -> Tuple[bool, float]:
        """Determine if a break should be taken and for how long"""
        current_time = time.time()
        
        # Check attention span
        attention_score = self._calculate_attention_score()
        
        # Check time since last break
        time_since_break = self._time_since_last_break()
        
        # AI-powered break prediction
        if GPU_AVAILABLE and self.attention_model:
            break_probability = self._ai_predict_break_need(attention_score, time_since_break)
        else:
            break_probability = self._fallback_break_prediction(attention_score, time_since_break)
        
        # Decision threshold
        if break_probability > 0.7:
            break_duration = self._calculate_break_duration(attention_score)
            return True, break_duration
        
        return False, 0.0
    
    def get_risk_assessment(self) -> Dict[str, float]:
        """Get current behavioral risk assessment"""
        if GPU_AVAILABLE and self.risk_model:
            return self._ai_risk_assessment()
        else:
            return self._fallback_risk_assessment()
    
    def update_behavior_profile(self) -> BehaviorProfile:
        """Update and return current behavior profile"""
        if len(self.mouse_movements) < 10:
            return self.current_profile
        
        # Calculate behavior metrics
        movements = list(self.mouse_movements)[-100:]  # Last 100 movements
        
        speeds = [self._calculate_speed(m) for m in movements]
        mouse_speed_avg = np.mean(speeds)
        mouse_speed_std = np.std(speeds)
        
        # Reaction time analysis
        reaction_times = self._calculate_reaction_times()
        reaction_time_avg = np.mean(reaction_times) if reaction_times else 0.5
        reaction_time_std = np.std(reaction_times) if reaction_times else 0.1
        
        # Behavioral characteristics
        break_frequency = self._calculate_break_frequency()
        attention_span = self._calculate_attention_span()
        multitasking_tendency = self._calculate_multitasking_tendency()
        consistency_score = self._calculate_consistency_score()
        risk_tolerance = self._calculate_risk_tolerance()
        
        self.current_profile = BehaviorProfile(
            user_id="current_session",
            mouse_speed_avg=mouse_speed_avg,
            mouse_speed_std=mouse_speed_std,
            reaction_time_avg=reaction_time_avg,
            reaction_time_std=reaction_time_std,
            break_frequency=break_frequency,
            attention_span=attention_span,
            multitasking_tendency=multitasking_tendency,
            consistency_score=consistency_score,
            risk_tolerance=risk_tolerance
        )
        
        return self.current_profile
    
    def _calculate_jitter(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> float:
        """Calculate movement jitter/smoothness"""
        distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        return random.uniform(0.1, 0.3) * distance / 100
    
    def _calculate_speed(self, movement: MouseMovement) -> float:
        """Calculate movement speed"""
        distance = np.sqrt(
            (movement.end_pos[0] - movement.start_pos[0])**2 + 
            (movement.end_pos[1] - movement.start_pos[1])**2
        )
        return distance / movement.duration if movement.duration > 0 else 0
    
    def _ai_generate_movement(self, start_pos: Tuple[int, int], 
                            end_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """AI-powered movement generation"""
        try:
            # Prepare input features
            features = torch.tensor([
                start_pos[0], start_pos[1], end_pos[0], end_pos[1],
                self.current_profile.mouse_speed_avg if self.current_profile else 200,
                self.current_profile.consistency_score if self.current_profile else 0.5
            ], dtype=torch.float32).to(self.device)
            
            # Generate movement path
            with torch.no_grad():
                path_points = self.movement_model(features.unsqueeze(0))
                path_points = path_points.cpu().numpy()[0]
            
            # Convert to coordinate list
            path = []
            for i in range(0, len(path_points), 2):
                if i + 1 < len(path_points):
                    path.append((int(path_points[i]), int(path_points[i + 1])))
            
            return path
            
        except Exception as e:
            logger.error(f"AI movement generation failed: {e}")
            return self._fallback_generate_movement(start_pos, end_pos)
    
    def _fallback_generate_movement(self, start_pos: Tuple[int, int], 
                                  end_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Fallback movement generation using bezier curves"""
        # Generate bezier curve with human-like characteristics
        num_points = max(10, int(np.sqrt(
            (end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2
        ) / 20))
        
        # Control points for bezier curve
        mid_x = (start_pos[0] + end_pos[0]) / 2
        mid_y = (start_pos[1] + end_pos[1]) / 2
        
        # Add some randomness to make it more human-like
        offset_x = random.uniform(-50, 50)
        offset_y = random.uniform(-50, 50)
        
        control1 = (mid_x + offset_x, mid_y + offset_y)
        
        path = []
        for i in range(num_points + 1):
            t = i / num_points
            # Cubic bezier curve
            x = (1-t)**3 * start_pos[0] + 3*(1-t)**2*t * control1[0] + 3*(1-t)*t**2 * control1[0] + t**3 * end_pos[0]
            y = (1-t)**3 * start_pos[1] + 3*(1-t)**2*t * control1[1] + 3*(1-t)*t**2 * control1[1] + t**3 * end_pos[1]
            
            # Add micro-jitter
            jitter_x = random.uniform(-2, 2)
            jitter_y = random.uniform(-2, 2)
            
            path.append((int(x + jitter_x), int(y + jitter_y)))
        
        return path
    
    def _ai_predict_timing(self, action_type: str, context: Optional[Dict[str, Any]] = None) -> float:
        """AI-powered timing prediction"""
        try:
            # Feature engineering
            features = [
                hash(action_type) % 1000 / 1000,  # Action type encoding
                len(self.action_events) % 100 / 100,  # Recent activity
                self.current_profile.reaction_time_avg if self.current_profile else 0.5,
                self._calculate_attention_score(),
                time.time() % 3600 / 3600  # Time of day factor
            ]
            
            if context:
                features.append(len(context) / 10)  # Context complexity
            else:
                features.append(0.0)
            
            features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                predicted_time = self.timing_model(features_tensor.unsqueeze(0))
                return float(predicted_time.cpu().numpy()[0])
                
        except Exception as e:
            logger.error(f"AI timing prediction failed: {e}")
            return self._fallback_predict_timing(action_type)
    
    def _fallback_predict_timing(self, action_type: str) -> float:
        """Fallback timing prediction"""
        base_timings = {
            "click": 0.1,
            "double_click": 0.2,
            "drag": 0.3,
            "key_press": 0.05,
            "type": 0.1,
            "scroll": 0.15
        }
        
        base_time = base_timings.get(action_type, 0.1)
        
        # Add human-like variation
        variation = random.gauss(1.0, 0.3)
        variation = max(0.5, min(2.0, variation))  # Clamp to reasonable range
        
        return base_time * variation
    
    def _analyze_movement_patterns(self):
        """Analyze movement patterns for behavioral insights"""
        if len(self.mouse_movements) < 10:
            return
        
        movements = list(self.mouse_movements)[-50:]  # Last 50 movements
        
        # Calculate pattern metrics
        speeds = [self._calculate_speed(m) for m in movements]
        accelerations = [m.acceleration for m in movements]
        
        # Update behavior analytics
        self.behavior_analyzer.update_movement_analysis(speeds, accelerations)
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_movement_anomalies(movements)
        if anomalies:
            logger.warning(f"Detected {len(anomalies)} movement anomalies")
    
    def _update_attention_tracking(self, action: ActionEvent):
        """Update attention tracking based on action"""
        self.attention_events.append({
            'timestamp': action.timestamp,
            'action_type': action.action_type,
            'duration': action.duration
        })
    
    def _calculate_attention_score(self) -> float:
        """Calculate current attention score (0-1)"""
        if len(self.attention_events) < 5:
            return 0.8  # Default high attention
        
        recent_events = list(self.attention_events)[-20:]
        
        # Calculate attention based on action consistency and timing
        time_intervals = []
        for i in range(1, len(recent_events)):
            interval = recent_events[i]['timestamp'] - recent_events[i-1]['timestamp']
            time_intervals.append(interval)
        
        if not time_intervals:
            return 0.8
        
        # More consistent intervals = higher attention
        interval_std = np.std(time_intervals)
        consistency_score = 1.0 / (1.0 + interval_std)
        
        # Factor in recent activity level
        recent_activity = len(recent_events) / 20.0
        
        attention_score = (consistency_score * 0.7 + recent_activity * 0.3)
        return max(0.1, min(1.0, attention_score))
    
    def _time_since_last_break(self) -> float:
        """Get time since last break in seconds"""
        if not self.break_patterns:
            return 3600  # 1 hour default
        
        last_break = self.break_patterns[-1]
        return time.time() - last_break['timestamp']
    
    def _calculate_break_duration(self, attention_score: float) -> float:
        """Calculate appropriate break duration"""
        base_duration = 30  # 30 seconds base
        
        # Longer breaks for lower attention
        attention_factor = (1.0 - attention_score) * 2.0
        
        # Random variation
        random_factor = random.uniform(0.8, 1.5)
        
        duration = base_duration * (1.0 + attention_factor) * random_factor
        
        # Clamp to reasonable range
        return max(15, min(300, duration))  # 15 seconds to 5 minutes
    
    def _calculate_break_frequency(self) -> float:
        """Calculate break frequency (breaks per hour)"""
        if len(self.break_patterns) < 2:
            return 6.0  # Default 6 breaks per hour
        
        breaks = list(self.break_patterns)
        if len(breaks) < 2:
            return 6.0
        
        time_span = breaks[-1]['timestamp'] - breaks[0]['timestamp']
        if time_span <= 0:
            return 6.0
        
        return len(breaks) / (time_span / 3600)  # Breaks per hour
    
    def _calculate_attention_span(self) -> float:
        """Calculate average attention span in minutes"""
        if len(self.attention_events) < 10:
            return 15.0  # Default 15 minutes
        
        # Implementation would analyze attention patterns
        return 15.0  # Simplified for now
    
    def _calculate_multitasking_tendency(self) -> float:
        """Calculate multitasking tendency (0-1)"""
        # Implementation would analyze action switching patterns
        return 0.3  # Simplified for now
    
    def _calculate_consistency_score(self) -> float:
        """Calculate behavioral consistency score (0-1)"""
        if len(self.mouse_movements) < 10:
            return 0.5
        
        movements = list(self.mouse_movements)[-50:]
        speeds = [self._calculate_speed(m) for m in movements]
        
        # Lower standard deviation = higher consistency
        speed_std = np.std(speeds)
        speed_mean = np.mean(speeds)
        
        if speed_mean == 0:
            return 0.5
        
        consistency = 1.0 / (1.0 + speed_std / speed_mean)
        return max(0.1, min(1.0, consistency))
    
    def _calculate_risk_tolerance(self) -> float:
        """Calculate risk tolerance (0-1)"""
        # Implementation would analyze risky behavior patterns
        return 0.4  # Simplified for now
    
    def _calculate_reaction_times(self) -> List[float]:
        """Calculate reaction times from recent actions"""
        if len(self.action_events) < 2:
            return [0.5]  # Default reaction time
        
        reactions = []
        events = list(self.action_events)[-20:]
        
        for i in range(1, len(events)):
            interval = events[i].timestamp - events[i-1].timestamp
            if 0.1 <= interval <= 2.0:  # Reasonable reaction time range
                reactions.append(interval)
        
        return reactions if reactions else [0.5]
    
    def _ai_predict_break_need(self, attention_score: float, time_since_break: float) -> float:
        """AI-powered break need prediction"""
        try:
            features = torch.tensor([
                attention_score,
                time_since_break / 3600,  # Normalize to hours
                len(self.action_events) % 100 / 100,  # Recent activity
                self._calculate_consistency_score(),
                time.time() % 86400 / 86400  # Time of day
            ], dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                break_probability = self.attention_model(features.unsqueeze(0))
                return float(torch.sigmoid(break_probability).cpu().numpy()[0])
                
        except Exception as e:
            logger.error(f"AI break prediction failed: {e}")
            return self._fallback_break_prediction(attention_score, time_since_break)
    
    def _fallback_break_prediction(self, attention_score: float, time_since_break: float) -> float:
        """Fallback break prediction"""
        # Time-based factor
        time_factor = min(1.0, time_since_break / 1800)  # 30 minutes = 1.0
        
        # Attention-based factor
        attention_factor = 1.0 - attention_score
        
        # Random element
        random_factor = random.uniform(0.8, 1.2)
        
        probability = (time_factor * 0.6 + attention_factor * 0.4) * random_factor
        return max(0.0, min(1.0, probability))
    
    def _ai_risk_assessment(self) -> Dict[str, float]:
        """AI-powered risk assessment"""
        try:
            # Prepare features for risk model
            features = [
                self._calculate_consistency_score(),
                self._calculate_attention_score(),
                len(self.mouse_movements) / 1000,  # Activity level
                self._time_since_last_break() / 3600,  # Hours since break
                self.current_profile.mouse_speed_avg / 500 if self.current_profile else 0.4
            ]
            
            features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                risk_scores = self.risk_model(features_tensor.unsqueeze(0))
                risk_scores = torch.sigmoid(risk_scores).cpu().numpy()[0]
            
            return {
                'detection_risk': float(risk_scores[0]),
                'pattern_risk': float(risk_scores[1]),
                'timing_risk': float(risk_scores[2]),
                'overall_risk': float(np.mean(risk_scores))
            }
            
        except Exception as e:
            logger.error(f"AI risk assessment failed: {e}")
            return self._fallback_risk_assessment()
    
    def _fallback_risk_assessment(self) -> Dict[str, float]:
        """Fallback risk assessment"""
        consistency = self._calculate_consistency_score()
        attention = self._calculate_attention_score()
        
        # Higher consistency and attention = lower risk
        detection_risk = 1.0 - (consistency * 0.7 + attention * 0.3)
        pattern_risk = 1.0 - consistency
        timing_risk = 1.0 - attention
        overall_risk = (detection_risk + pattern_risk + timing_risk) / 3.0
        
        return {
            'detection_risk': max(0.0, min(1.0, detection_risk)),
            'pattern_risk': max(0.0, min(1.0, pattern_risk)),
            'timing_risk': max(0.0, min(1.0, timing_risk)),
            'overall_risk': max(0.0, min(1.0, overall_risk))
        }


class MouseMovementPredictor(nn.Module):
    """Neural network for predicting human-like mouse movements"""
    
    def __init__(self, input_size=6, hidden_size=128, output_size=20):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


class TimingPredictor(nn.Module):
    """Neural network for predicting human-like timing"""
    
    def __init__(self, input_size=6, hidden_size=64, output_size=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


class AttentionModel(nn.Module):
    """Neural network for modeling attention patterns"""
    
    def __init__(self, input_size=5, hidden_size=64, output_size=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


class RiskAssessmentModel(nn.Module):
    """Neural network for risk assessment"""
    
    def __init__(self, input_size=5, hidden_size=64, output_size=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


class BehaviorAnalyzer:
    """Analyzes behavioral patterns and trends"""
    
    def __init__(self):
        self.movement_history = deque(maxlen=1000)
        self.timing_history = deque(maxlen=1000)
        self.patterns = {}
    
    def update_movement_analysis(self, speeds: List[float], accelerations: List[float]):
        """Update movement analysis with new data"""
        self.movement_history.extend(speeds)
        
        # Calculate movement statistics
        if len(self.movement_history) >= 10:
            self.patterns['speed_mean'] = np.mean(self.movement_history)
            self.patterns['speed_std'] = np.std(self.movement_history)
            self.patterns['speed_trend'] = self._calculate_trend(list(self.movement_history))
    
    def _calculate_trend(self, data: List[float]) -> float:
        """Calculate trend in data (positive = increasing, negative = decreasing)"""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        slope, _ = np.polyfit(x, data, 1)
        return slope


class PatternDetector:
    """Detects behavioral patterns and anomalies"""
    
    def __init__(self):
        self.known_patterns = {}
        self.anomaly_threshold = 2.0  # Standard deviations
    
    def detect_patterns(self, data: List[float]) -> List[Dict]:
        """Detect patterns in behavioral data"""
        patterns = []
        
        if len(data) < 10:
            return patterns
        
        # Simple pattern detection (can be enhanced with ML)
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        # Detect outliers
        outliers = [i for i, val in enumerate(data) if abs(val - mean_val) > self.anomaly_threshold * std_val]
        
        if outliers:
            patterns.append({
                'type': 'outliers',
                'indices': outliers,
                'severity': len(outliers) / len(data)
            })
        
        return patterns


class AnomalyDetector:
    """Detects anomalous behavior patterns"""
    
    def __init__(self):
        self.baseline_established = False
        self.baseline_stats = {}
    
    def detect_movement_anomalies(self, movements: List[MouseMovement]) -> List[Dict]:
        """Detect anomalous movement patterns"""
        anomalies = []
        
        if len(movements) < 5:
            return anomalies
        
        speeds = [self._calculate_speed(m) for m in movements]
        
        # Statistical anomaly detection
        mean_speed = np.mean(speeds)
        std_speed = np.std(speeds)
        
        for i, speed in enumerate(speeds):
            if abs(speed - mean_speed) > 2.5 * std_speed:
                anomalies.append({
                    'type': 'speed_anomaly',
                    'index': i,
                    'value': speed,
                    'severity': abs(speed - mean_speed) / std_speed
                })
        
        return anomalies
    
    def _calculate_speed(self, movement: MouseMovement) -> float:
        """Calculate movement speed"""
        distance = np.sqrt(
            (movement.end_pos[0] - movement.start_pos[0])**2 + 
            (movement.end_pos[1] - movement.start_pos[1])**2
        )
        return distance / movement.duration if movement.duration > 0 else 0


# Global behavior model instance
behavior_model = None


def get_behavior_model() -> HumanBehaviorModel:
    """Get the global behavior model instance"""
    global behavior_model
    if behavior_model is None:
        behavior_model = HumanBehaviorModel()
    return behavior_model


def initialize_behavior_modeling():
    """Initialize the behavior modeling system"""
    global behavior_model
    behavior_model = HumanBehaviorModel()
    logger.info("Behavior modeling system initialized")
    return behavior_model


if __name__ == "__main__":
    # Test the behavior modeling system
    model = HumanBehaviorModel()
    
    # Simulate some movements
    for i in range(10):
        start = (random.randint(0, 800), random.randint(0, 600))
        end = (random.randint(0, 800), random.randint(0, 600))
        duration = random.uniform(0.1, 0.5)
        
        model.record_mouse_movement(start, end, duration)
        
        # Generate human-like movement
        path = model.generate_human_movement(start, end)
        print(f"Generated path with {len(path)} points")
        
        # Get timing
        timing = model.get_next_action_timing("click")
        print(f"Next action timing: {timing:.3f}s")
    
    # Check break recommendation
    should_break, duration = model.should_take_break()
    print(f"Should take break: {should_break}, Duration: {duration:.1f}s")
    
    # Get risk assessment
    risk = model.get_risk_assessment()
    print(f"Risk assessment: {risk}")
    
    # Update behavior profile
    profile = model.update_behavior_profile()
    print(f"Behavior profile updated: {profile}")