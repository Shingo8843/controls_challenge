# controllers/adaptive_pid.py

from . import BaseController
import numpy as np

class Controller(BaseController):
    """
    Adaptive PID controller:
    - Starts with optimized base values
    - Adjusts PID gains based on error magnitude and system behavior
    - Uses error magnitude to adjust P gain
    - Uses error rate to adjust D gain
    - Uses integral accumulation to adjust I gain
    """

    def __init__(self):
        # Base PID gains (from optimization)
        self.p = 0.259
        self.i = 0.094
        self.d = 0.012

        # Adaptive parameters - more conservative learning rates
        self.p_lr = 0.0001  # Reduced from 0.001
        self.i_lr = 0.00001  # Reduced from 0.0001
        self.d_lr = 0.00005  # Reduced from 0.0005

        # Tighter bounds for PID gains
        self.p_bounds = (0.2, 0.3)  # Tighter around optimal P
        self.i_bounds = (0.08, 0.12)  # Tighter around optimal I
        self.d_bounds = (0.005, 0.02)  # Tighter around optimal D

        # State variables
        self.error_integral = 0
        self.prev_error = 0
        self.prev_lataccel = 0
        self.error_history = []  # Store recent errors for trend analysis
        self.history_size = 10  # Increased from 5 for better trend analysis

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Calculate current error
        error = target_lataccel - current_lataccel
        
        # Update error history
        self.error_history.append(error)
        if len(self.error_history) > self.history_size:
            self.error_history.pop(0)
        
        # Calculate error metrics
        error_magnitude = abs(error)
        error_rate = error - self.prev_error
        error_trend = np.mean(np.diff(self.error_history)) if len(self.error_history) > 1 else 0
        
        # Update integral term with anti-windup
        self.error_integral += error
        if abs(self.error_integral) > 5:  # Reduced from 10 for tighter control
            self.error_integral = np.sign(self.error_integral) * 5
        
        # Adaptive P gain: increase when error is large, but more conservatively
        p_adjustment = self.p_lr * error_magnitude * (1 if error_magnitude > 0.5 else 0.5)
        self.p += p_adjustment
        
        # Adaptive I gain: more conservative adjustment
        if abs(self.error_integral) > 3:  # Reduced from 5
            self.i *= (1 - self.i_lr)
        elif error_magnitude > 0.5:  # Only adjust when error is significant
            self.i += self.i_lr * error_magnitude
        
        # Adaptive D gain: more conservative adjustment
        d_adjustment = self.d_lr * abs(error_rate) * (1 if abs(error_rate) > 0.1 else 0.5)
        if error_trend > 0.1:  # Only adjust for significant trends
            self.d += d_adjustment
        elif error_trend < -0.1:
            self.d -= d_adjustment
        
        # Clamp PID values within bounds
        self.p = np.clip(self.p, *self.p_bounds)
        self.i = np.clip(self.i, *self.i_bounds)
        self.d = np.clip(self.d, *self.d_bounds)
        
        # Calculate control output
        control = (self.p * error + 
                  self.i * self.error_integral + 
                  self.d * error_rate)
        
        # Update previous values
        self.prev_error = error
        self.prev_lataccel = current_lataccel
        
        return control
