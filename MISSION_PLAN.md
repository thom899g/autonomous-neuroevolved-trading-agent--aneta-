# Autonomous NeuroEvolved Trading Agent (ANETA)

## Objective
ANETA is an AI system that autonomously evolves trading strategies using neuroevolution and reinforcement learning. It generates hypotheses for new models, tests them in simulations, and refines its approach based on performance feedback, all without human intervention.

## Strategy
Research and implement using available tools.

## Execution Output
SUMMARY: I've created a comprehensive foundation for ANETA (Autonomous NeuroEvolved Trading Agent) with 8 core Python modules implementing neuroevolution for trading strategies. The system features a complete architecture with data collection, neuroevolution algorithms, strategy testing, autonomous orchestration, and Firebase integration for state management.

OUTPUT:

### FILE: requirements.txt
```
# Core dependencies
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
ta-lib>=0.4.0
ccxt>=4.0.0
gym>=0.26.0

# Firebase for state management
firebase-admin>=6.3.0
google-cloud-firestore>=2.11.0

# Neuroevolution & ML
neat-python>=0.92
deap>=1.4.0
tensorflow>=2.13.0

# Data processing & utilities
python-dotenv>=1.0.0
redis>=4.5.0
psutil>=5.9.0
matplotlib>=3.7.0

# Trading & financial
backtrader>=1.9.76.123
yfinance>=0.2.28

# Logging & monitoring
structlog>=23.1.0
prometheus-client>=0.17.0
```

### FILE: config.py
```python
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