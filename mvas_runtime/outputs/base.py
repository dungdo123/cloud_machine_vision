"""
MVAS Output Handler Base

Abstract base class for output handlers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

from ..models import InspectionResult

logger = logging.getLogger(__name__)


class OutputHandler(ABC):
    """
    Abstract base class for output handlers.
    
    Output handlers receive inspection results and route them
    to various destinations: APIs, PLCs, databases, etc.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.enabled = True
    
    @abstractmethod
    def send(self, result: InspectionResult) -> bool:
        """
        Send inspection result to destination.
        
        Args:
            result: Inspection result to send
            
        Returns:
            True if sent successfully
        """
        pass
    
    def enable(self):
        """Enable this handler"""
        self.enabled = True
    
    def disable(self):
        """Disable this handler"""
        self.enabled = False


class DatabaseLogger(OutputHandler):
    """Log results to database"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.connection = None
        self._connect()
    
    def _connect(self):
        """Connect to database"""
        db_url = self.config.get("database_url", "sqlite:///mvas_results.db")
        # TODO: Implement database connection
        logger.info(f"Database logger configured for: {db_url}")
    
    def send(self, result: InspectionResult) -> bool:
        if not self.enabled:
            return True
        
        # TODO: Insert result into database
        logger.debug(f"Logging result {result.request_id} to database")
        return True


class FileLogger(OutputHandler):
    """Log results to file"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.log_path = self.config.get("log_path", "results.jsonl")
    
    def send(self, result: InspectionResult) -> bool:
        if not self.enabled:
            return True
        
        try:
            import json
            with open(self.log_path, "a") as f:
                f.write(result.model_dump_json() + "\n")
            return True
        except Exception as e:
            logger.error(f"Failed to log result: {e}")
            return False


class MQTTPublisher(OutputHandler):
    """Publish results to MQTT broker"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.client = None
        self.topic = self.config.get("topic", "mvas/results")
    
    def send(self, result: InspectionResult) -> bool:
        if not self.enabled or not self.client:
            return True
        
        try:
            self.client.publish(self.topic, result.model_dump_json())
            return True
        except Exception as e:
            logger.error(f"Failed to publish to MQTT: {e}")
            return False


class PLCOutput(OutputHandler):
    """
    Send results to PLC via Modbus TCP.
    
    Register mapping:
    - 0: Result code (0=pass, 1=fail, 2=review, 3=error)
    - 1: Anomaly score (0-1000, scaled)
    - 2: Confidence (0-1000, scaled)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.client = None
        self._connect()
    
    def _connect(self):
        """Connect to PLC"""
        plc_ip = self.config.get("plc_ip")
        if not plc_ip:
            return
        
        try:
            from pymodbus.client import ModbusTcpClient
            port = self.config.get("port", 502)
            self.client = ModbusTcpClient(plc_ip, port=port)
            self.client.connect()
            logger.info(f"Connected to PLC at {plc_ip}:{port}")
        except ImportError:
            logger.warning("pymodbus not installed, PLC output disabled")
        except Exception as e:
            logger.error(f"Failed to connect to PLC: {e}")
    
    def send(self, result: InspectionResult) -> bool:
        if not self.enabled or not self.client:
            return True
        
        try:
            # Map decision to code
            decision_map = {
                "pass": 0,
                "fail": 1,
                "review": 2,
                "error": 3,
            }
            
            base_addr = self.config.get("base_address", 0)
            
            self.client.write_register(base_addr, decision_map.get(result.decision.value, 3))
            self.client.write_register(base_addr + 1, int(result.anomaly_score * 1000))
            self.client.write_register(base_addr + 2, int(result.confidence * 1000))
            
            return True
        except Exception as e:
            logger.error(f"Failed to write to PLC: {e}")
            return False

