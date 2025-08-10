"""Intelligent auto-scaling system with resource prediction and adaptive scaling."""

import asyncio
import time
import logging
import statistics
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Auto-scaling directions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


class ResourceMetric(Enum):
    """Resource metrics for scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_LENGTH = "queue_length"
    ERROR_RATE = "error_rate"


@dataclass
class ScalingEvent:
    """Represents a scaling event."""
    timestamp: datetime
    direction: ScalingDirection
    old_capacity: int
    new_capacity: int
    trigger_metric: ResourceMetric
    trigger_value: float
    reason: str


@dataclass
class ResourcePrediction:
    """Resource usage prediction."""
    metric: ResourceMetric
    predicted_value: float
    confidence: float
    time_horizon_minutes: int
    prediction_accuracy: float = 0.0


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    name: str
    metric: ResourceMetric
    scale_up_threshold: float
    scale_down_threshold: float
    scale_up_amount: int = 2
    scale_down_amount: int = 1
    min_capacity: int = 2
    max_capacity: int = 50
    cooldown_period_seconds: int = 300  # 5 minutes
    evaluation_period_seconds: int = 60  # 1 minute
    enabled: bool = True


class ResourcePredictor:
    """Intelligent resource usage prediction system."""
    
    def __init__(self, history_size: int = 1000):
        """Initialize resource predictor."""
        self.history_size = history_size
        self.metric_history: Dict[ResourceMetric, List[Tuple[float, float]]] = {
            metric: [] for metric in ResourceMetric
        }
        self.prediction_accuracy: Dict[ResourceMetric, List[float]] = {
            metric: [] for metric in ResourceMetric
        }
        logger.info("Resource predictor initialized")
    
    def record_metric(self, metric: ResourceMetric, value: float, timestamp: float = None) -> None:
        """Record a metric value with timestamp."""
        if timestamp is None:
            timestamp = time.time()
        
        history = self.metric_history[metric]
        history.append((timestamp, value))
        
        # Trim history to prevent memory growth
        if len(history) > self.history_size:
            self.metric_history[metric] = history[-self.history_size:]
    
    def predict_metric(self, metric: ResourceMetric, minutes_ahead: int = 5) -> ResourcePrediction:
        """Predict future metric value using trend analysis."""
        history = self.metric_history[metric]
        
        if len(history) < 10:
            # Not enough data for prediction
            return ResourcePrediction(
                metric=metric,
                predicted_value=history[-1][1] if history else 0.0,
                confidence=0.1,
                time_horizon_minutes=minutes_ahead,
                prediction_accuracy=0.0
            )
        
        # Extract recent values for trend analysis
        recent_history = history[-60:]  # Last 60 data points
        timestamps = [t for t, v in recent_history]
        values = [v for t, v in recent_history]
        
        # Simple linear trend prediction
        try:
            # Calculate slope (rate of change)
            time_diffs = [timestamps[i] - timestamps[0] for i in range(len(timestamps))]
            slope = self._calculate_slope(time_diffs, values)
            
            # Predict value at future time
            current_value = values[-1]
            future_seconds = minutes_ahead * 60
            predicted_value = current_value + (slope * future_seconds)
            
            # Calculate confidence based on recent trend consistency
            confidence = self._calculate_prediction_confidence(values, slope)
            
            # Get historical accuracy
            avg_accuracy = statistics.mean(self.prediction_accuracy[metric]) if self.prediction_accuracy[metric] else 0.0
            
            return ResourcePrediction(
                metric=metric,
                predicted_value=max(0.0, predicted_value),
                confidence=confidence,
                time_horizon_minutes=minutes_ahead,
                prediction_accuracy=avg_accuracy
            )
            
        except Exception as e:
            logger.warning(f"Prediction failed for {metric.value}: {e}")
            return ResourcePrediction(
                metric=metric,
                predicted_value=values[-1],
                confidence=0.1,
                time_horizon_minutes=minutes_ahead
            )
    
    def _calculate_slope(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate linear regression slope."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _calculate_prediction_confidence(self, values: List[float], slope: float) -> float:
        """Calculate prediction confidence based on trend consistency."""
        if len(values) < 5:
            return 0.1
        
        # Calculate variance in recent values
        recent_values = values[-10:]
        variance = statistics.variance(recent_values) if len(recent_values) > 1 else 0.0
        mean_value = statistics.mean(recent_values)
        
        # Normalize variance
        normalized_variance = variance / max(mean_value, 1.0)
        
        # Lower variance = higher confidence
        base_confidence = max(0.1, 1.0 - min(normalized_variance, 1.0))
        
        # Adjust for trend strength
        abs_slope = abs(slope)
        if abs_slope > mean_value * 0.1:  # Significant trend
            base_confidence *= 1.2
        
        return min(1.0, base_confidence)
    
    def validate_prediction(self, metric: ResourceMetric, predicted_value: float, actual_value: float) -> float:
        """Validate a previous prediction and update accuracy metrics."""
        if predicted_value == 0:
            accuracy = 0.0
        else:
            accuracy = 1.0 - abs(predicted_value - actual_value) / max(predicted_value, actual_value, 1.0)
        
        accuracy = max(0.0, accuracy)
        self.prediction_accuracy[metric].append(accuracy)
        
        # Keep only recent accuracy scores
        if len(self.prediction_accuracy[metric]) > 100:
            self.prediction_accuracy[metric] = self.prediction_accuracy[metric][-100:]
        
        return accuracy


class AutoScaler:
    """Intelligent auto-scaling system with predictive capabilities."""
    
    def __init__(self):
        """Initialize auto-scaler."""
        self.predictor = ResourcePredictor()
        self.current_capacity = 10  # Starting capacity
        self.scaling_history: List[ScalingEvent] = []
        self.last_scaling_time = 0.0
        
        # Default scaling policies
        self.policies: Dict[str, ScalingPolicy] = {
            "cpu_policy": ScalingPolicy(
                name="CPU Utilization",
                metric=ResourceMetric.CPU_UTILIZATION,
                scale_up_threshold=75.0,
                scale_down_threshold=30.0,
                scale_up_amount=3,
                scale_down_amount=1,
                min_capacity=5,
                max_capacity=100
            ),
            "response_time_policy": ScalingPolicy(
                name="Response Time",
                metric=ResourceMetric.RESPONSE_TIME,
                scale_up_threshold=2.0,  # 2 seconds
                scale_down_threshold=0.5,  # 0.5 seconds
                scale_up_amount=2,
                scale_down_amount=1,
                min_capacity=5,
                max_capacity=50
            ),
            "queue_length_policy": ScalingPolicy(
                name="Queue Length",
                metric=ResourceMetric.QUEUE_LENGTH,
                scale_up_threshold=10.0,
                scale_down_threshold=2.0,
                scale_up_amount=5,  # Aggressive scaling for queue buildup
                scale_down_amount=1,
                min_capacity=5,
                max_capacity=75
            ),
            "error_rate_policy": ScalingPolicy(
                name="Error Rate",
                metric=ResourceMetric.ERROR_RATE,
                scale_up_threshold=5.0,  # 5% error rate
                scale_down_threshold=1.0,  # 1% error rate
                scale_up_amount=4,  # Quick response to errors
                scale_down_amount=1,
                min_capacity=5,
                max_capacity=100
            )
        }
        
        logger.info(f"Auto-scaler initialized with capacity {self.current_capacity}")
    
    async def evaluate_scaling_decision(self, metrics: Dict[ResourceMetric, float]) -> Tuple[ScalingDirection, int, str]:
        """Evaluate scaling decision based on current metrics and predictions."""
        
        # Record current metrics
        current_time = time.time()
        for metric, value in metrics.items():
            self.predictor.record_metric(metric, value, current_time)
        
        # Check cooldown period
        if current_time - self.last_scaling_time < 300:  # 5 minute minimum cooldown
            return ScalingDirection.MAINTAIN, self.current_capacity, "Cooldown period active"
        
        # Evaluate each policy
        scaling_recommendations = []
        
        for policy_name, policy in self.policies.items():
            if not policy.enabled:
                continue
            
            current_value = metrics.get(policy.metric, 0.0)
            if current_value == 0.0:
                continue  # No data for this metric
            
            # Get prediction for next 5 minutes
            prediction = self.predictor.predict_metric(policy.metric, minutes_ahead=5)
            
            # Make scaling decision
            decision = self._evaluate_policy(policy, current_value, prediction)
            if decision[0] != ScalingDirection.MAINTAIN:
                scaling_recommendations.append((decision, policy_name, current_value, prediction.predicted_value))
        
        # Aggregate recommendations
        if not scaling_recommendations:
            return ScalingDirection.MAINTAIN, self.current_capacity, "All metrics within acceptable ranges"
        
        # Prioritize scale-up over scale-down
        scale_up_recs = [r for r in scaling_recommendations if r[0][0] == ScalingDirection.SCALE_UP]
        if scale_up_recs:
            # Choose most aggressive scale-up
            chosen = max(scale_up_recs, key=lambda x: x[0][1])
            direction, amount = chosen[0]
            policy_name = chosen[1]
            current_val = chosen[2]
            predicted_val = chosen[3]
            
            new_capacity = min(self.current_capacity + amount, 200)  # Global max capacity
            reason = f"{policy_name}: current={current_val:.2f}, predicted={predicted_val:.2f}"
            
            return direction, new_capacity, reason
        
        # Only scale-down recommendations
        scale_down_recs = [r for r in scaling_recommendations if r[0][0] == ScalingDirection.SCALE_DOWN]
        if scale_down_recs:
            # Choose most conservative scale-down
            chosen = min(scale_down_recs, key=lambda x: x[0][1])
            direction, amount = chosen[0]
            policy_name = chosen[1]
            current_val = chosen[2]
            predicted_val = chosen[3]
            
            new_capacity = max(self.current_capacity - amount, 2)  # Global min capacity
            reason = f"{policy_name}: current={current_val:.2f}, predicted={predicted_val:.2f}"
            
            return direction, new_capacity, reason
        
        return ScalingDirection.MAINTAIN, self.current_capacity, "No clear scaling recommendation"
    
    def _evaluate_policy(self, policy: ScalingPolicy, current_value: float, prediction: ResourcePrediction) -> Tuple[ScalingDirection, int]:
        """Evaluate scaling decision for a specific policy."""
        
        # Weight current and predicted values
        confidence_weight = min(prediction.confidence, 0.7)  # Max 70% weight on prediction
        weighted_value = (current_value * (1 - confidence_weight)) + (prediction.predicted_value * confidence_weight)
        
        # Scale up decision
        if weighted_value > policy.scale_up_threshold:
            # More aggressive scaling if prediction is much higher
            multiplier = 1
            if prediction.predicted_value > current_value * 1.2:
                multiplier = 2
            
            amount = policy.scale_up_amount * multiplier
            return ScalingDirection.SCALE_UP, amount
        
        # Scale down decision (be more conservative)
        elif weighted_value < policy.scale_down_threshold and current_value < policy.scale_down_threshold:
            # Only scale down if both current and predicted are low
            if prediction.predicted_value < policy.scale_down_threshold:
                return ScalingDirection.SCALE_DOWN, policy.scale_down_amount
        
        return ScalingDirection.MAINTAIN, 0
    
    async def execute_scaling(self, direction: ScalingDirection, new_capacity: int, reason: str) -> bool:
        """Execute the scaling decision."""
        if direction == ScalingDirection.MAINTAIN:
            return True
        
        old_capacity = self.current_capacity
        
        try:
            # Record scaling event
            event = ScalingEvent(
                timestamp=datetime.utcnow(),
                direction=direction,
                old_capacity=old_capacity,
                new_capacity=new_capacity,
                trigger_metric=ResourceMetric.CPU_UTILIZATION,  # TODO: Pass actual trigger
                trigger_value=0.0,  # TODO: Pass actual value
                reason=reason
            )
            
            # Execute scaling (this would interface with actual infrastructure)
            success = await self._perform_scaling(old_capacity, new_capacity)
            
            if success:
                self.current_capacity = new_capacity
                self.last_scaling_time = time.time()
                self.scaling_history.append(event)
                
                # Trim scaling history
                if len(self.scaling_history) > 100:
                    self.scaling_history = self.scaling_history[-100:]
                
                logger.info(f"Scaled {direction.value} from {old_capacity} to {new_capacity}: {reason}")
                return True
            else:
                logger.error(f"Scaling failed: {direction.value} to {new_capacity}")
                return False
                
        except Exception as e:
            logger.error(f"Scaling execution failed: {e}")
            return False
    
    async def _perform_scaling(self, old_capacity: int, new_capacity: int) -> bool:
        """Perform the actual scaling operation."""
        # This is a placeholder for actual infrastructure scaling
        # In a real implementation, this would:
        # 1. Update load balancer configuration
        # 2. Start/stop worker processes/containers
        # 3. Update service discovery
        # 4. Verify new capacity is operational
        
        await asyncio.sleep(0.1)  # Simulate scaling time
        logger.info(f"Scaling simulation: {old_capacity} -> {new_capacity} workers")
        return True
    
    def get_scaling_recommendations(self, metrics: Dict[ResourceMetric, float]) -> List[Dict[str, Any]]:
        """Get scaling recommendations without executing them."""
        recommendations = []
        
        for policy_name, policy in self.policies.items():
            if not policy.enabled:
                continue
                
            current_value = metrics.get(policy.metric, 0.0)
            if current_value == 0.0:
                continue
            
            prediction = self.predictor.predict_metric(policy.metric, minutes_ahead=5)
            decision = self._evaluate_policy(policy, current_value, prediction)
            
            recommendations.append({
                "policy": policy_name,
                "metric": policy.metric.value,
                "current_value": current_value,
                "predicted_value": prediction.predicted_value,
                "prediction_confidence": prediction.confidence,
                "recommendation": decision[0].value,
                "scale_amount": decision[1],
                "threshold_up": policy.scale_up_threshold,
                "threshold_down": policy.scale_down_threshold
            })
        
        return recommendations
    
    def get_auto_scaler_stats(self) -> Dict[str, Any]:
        """Get comprehensive auto-scaler statistics."""
        recent_events = [e for e in self.scaling_history if e.timestamp > datetime.utcnow() - timedelta(hours=24)]
        
        scale_up_count = len([e for e in recent_events if e.direction == ScalingDirection.SCALE_UP])
        scale_down_count = len([e for e in recent_events if e.direction == ScalingDirection.SCALE_DOWN])
        
        # Prediction accuracy stats
        accuracy_stats = {}
        for metric in ResourceMetric:
            accuracies = self.predictor.prediction_accuracy[metric]
            if accuracies:
                accuracy_stats[metric.value] = {
                    "average_accuracy": statistics.mean(accuracies),
                    "samples": len(accuracies),
                    "min_accuracy": min(accuracies),
                    "max_accuracy": max(accuracies)
                }
        
        return {
            "current_capacity": self.current_capacity,
            "scaling_policies": len([p for p in self.policies.values() if p.enabled]),
            "total_scaling_events": len(self.scaling_history),
            "recent_24h_events": {
                "total": len(recent_events),
                "scale_up": scale_up_count,
                "scale_down": scale_down_count
            },
            "last_scaling": {
                "timestamp": self.scaling_history[-1].timestamp.isoformat() if self.scaling_history else None,
                "direction": self.scaling_history[-1].direction.value if self.scaling_history else None,
                "capacity_change": f"{self.scaling_history[-1].old_capacity} -> {self.scaling_history[-1].new_capacity}" if self.scaling_history else None
            },
            "prediction_accuracy": accuracy_stats,
            "policy_status": {
                name: {
                    "enabled": policy.enabled,
                    "metric": policy.metric.value,
                    "thresholds": f"{policy.scale_down_threshold} < x < {policy.scale_up_threshold}",
                    "capacity_range": f"{policy.min_capacity} - {policy.max_capacity}"
                }
                for name, policy in self.policies.items()
            }
        }
    
    def update_policy(self, policy_name: str, **kwargs) -> bool:
        """Update scaling policy parameters."""
        if policy_name not in self.policies:
            return False
        
        policy = self.policies[policy_name]
        for key, value in kwargs.items():
            if hasattr(policy, key):
                setattr(policy, key, value)
                logger.info(f"Updated policy {policy_name}: {key} = {value}")
        
        return True


# Global auto-scaler instance
auto_scaler = AutoScaler()