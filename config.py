"""
ANETA Configuration Management
Centralized configuration with environment variable support
"""
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path

# Environment detection
class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class NeuroevolutionConfig:
    """Configuration for neuroevolution algorithms"""
    
    # NEAT Algorithm Parameters
    pop_size: int = 200
    max_generations: int = 500
    fitness_threshold: float = 0.95
    activation_mutation_rate: float = 0.3
    connection_mutation_rate: float = 0.5
    node_mutation_rate: float = 0.2
    weight_mutation_rate: float = 0.8
    weight_mutation_strength: float = 0.5
    crossover_rate: float = 0.75
    
    # Neural Network Architecture
    input_nodes: int = 30  # Technical indicators + market features
    hidden_layers: List[int] = field(default_factory=lambda: [20, 15, 10])
    output_nodes: int = 3  # BUY, SELL, HOLD
    activation_functions: List[str] = field(default_factory=lambda: [
        "relu", "sigmoid", "tanh"
    ])
    
    # Speciation
    compatibility_threshold: float = 3.0
    species_elitism: int = 2
    stagnation_threshold: int = 15
    
    # Novelty Search (Optional)
    use_novelty_search: bool = True
    novelty_k_nearest: int = 15
    novelty_threshold: float = 0.3

@dataclass
class TradingConfig:
    """Configuration for trading environment"""
    
    # Market Data
    symbols: List[str] = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "SOL/USDT"
    ])
    timeframe: str = "1h"
    lookback_periods: int = 200
    
    # Risk Management
    initial_capital: float = 10000.0
    max_position_size: float = 0.2  # 20% of capital
    stop_loss_pct: float = 0.05  # 5%
    take_profit_pct: float = 0.10  # 10%
    max_daily_loss: float = 0.02  # 2%
    
    # Transaction Costs
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    
    # Technical Indicators
    indicators: Dict[str, Dict] = field(default_factory=lambda: {
        "sma": {"periods": [20, 50, 200]},
        "ema": {"periods": [12, 26]},
        "rsi": {"period": 14},
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "bollinger": {"period": 20, "std": 2},
        "atr": {"period": 14},
        "volume_profile": {"period": 20}
    })

@dataclass
class FirebaseConfig:
    """Firebase configuration for state management"""
    
    project_id: str = "aneta-trading-system"
    collection_name: str = "aneta_experiments"
    cache_ttl: int = 300  # 5 minutes
    batch_size: int = 500
    
    # Collections
    collections: Dict[str, str] = field(default_factory=lambda: {
        "experiments": "experiments",
        "strategies": "evolved_strategies",
        "performances": "strategy_performances",
        "market_data": "market_data_cache",
        "system_metrics": "system_metrics"
    })

class ANETAConfig:
    """Main configuration class with environment-based settings"""
    
    def __init__(self, env: str = "development"):
        self.environment = Environment(env.lower())
        self.version = "1.0.0"
        
        # Initialize sub-configurations
        self.neuroevolution = NeuroevolutionConfig()
        self.trading = TradingConfig()
        self.firebase = FirebaseConfig()
        
        # Environment-specific overrides
        self._apply_environment_overrides()
        
        # Logging configuration
        self.logging_config = self._setup_logging()
        
        # Paths
        self.base_path = Path(__file__).parent
        self.data_path = self.base_path / "data"
        self.models_path = self.base_path / "models"
        self.logs_path = self.base_path / "logs"
        
        # Create directories
        self._