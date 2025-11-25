"""
Base Agent Class
Provides common functionality for all specialized agents
"""

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime

from .communication.agent_bus import AgentBus, AgentMessage, MessageType

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all specialized agents"""
    
    def __init__(self, name: str, agent_bus: Optional[AgentBus] = None):
        """
        Initialize base agent
        
        Args:
            name: Unique name for this agent
            agent_bus: Communication bus for agent coordination
        """
        self.name = name
        self.agent_bus = agent_bus or AgentBus()
        self.state: Dict[str, Any] = {}
        self.task_history: List[Dict[str, Any]] = []
        self.is_active = False
        
        # Subscribe to bus
        self.agent_bus.subscribe(self.name, self._handle_message)
        
        logger.debug(f"Agent {self.name} initialized")
    
    def _handle_message(self, message: AgentMessage):
        """Handle incoming messages from the bus"""
        if message.receiver and message.receiver != self.name:
            return  # Not for this agent
        
        try:
            if message.message_type == MessageType.TASK_REQUEST:
                self._handle_task_request(message)
            elif message.message_type == MessageType.COORDINATION:
                self._handle_coordination(message)
            elif message.message_type == MessageType.STATE_UPDATE:
                self._handle_state_update(message)
        except Exception as e:
            logger.error(f"Error handling message in {self.name}: {e}")
            self._send_error_response(message, str(e))
    
    def _handle_task_request(self, message: AgentMessage):
        """Handle task request - to be overridden by subclasses"""
        logger.debug(f"{self.name} received task request: {message.payload.get('task')}")
    
    def _handle_coordination(self, message: AgentMessage):
        """Handle coordination message - to be overridden by subclasses"""
        logger.debug(f"{self.name} received coordination message")
    
    def _handle_state_update(self, message: AgentMessage):
        """Handle state update message"""
        state_updates = message.payload.get('state', {})
        self.state.update(state_updates)
        logger.debug(f"{self.name} state updated: {list(state_updates.keys())}")
    
    def _send_message(self, receiver: Optional[str], message_type: MessageType, 
                     payload: Dict[str, Any], correlation_id: Optional[str] = None):
        """Send a message via the bus"""
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            sender=self.name,
            receiver=receiver,
            message_type=message_type,
            payload=payload,
            timestamp=datetime.now(),
            correlation_id=correlation_id
        )
        self.agent_bus.publish(message)
        return message.message_id
    
    def _send_task_result(self, receiver: str, task_id: str, result: Dict[str, Any], 
                         success: bool = True, error: Optional[str] = None):
        """Send task result to receiver"""
        payload = {
            "task_id": task_id,
            "success": success,
            "result": result if success else None,
            "error": error
        }
        self._send_message(receiver, MessageType.TASK_RESULT, payload, correlation_id=task_id)
    
    def _send_error_response(self, original_message: AgentMessage, error: str):
        """Send error response for a message"""
        payload = {
            "original_message_id": original_message.message_id,
            "error": error
        }
        self._send_message(
            original_message.sender,
            MessageType.ERROR,
            payload,
            correlation_id=original_message.message_id
        )
    
    def _send_state_update(self, state: Dict[str, Any], receiver: Optional[str] = None):
        """Broadcast state update"""
        payload = {"state": state}
        self._send_message(receiver, MessageType.STATE_UPDATE, payload)
    
    @abstractmethod
    def execute_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task - must be implemented by subclasses
        
        Args:
            task: Task identifier
            context: Current simulation context
            
        Returns:
            Task result dictionary
        """
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        return {
            "name": self.name,
            "is_active": self.is_active,
            "state": self.state.copy(),
            "task_count": len(self.task_history)
        }
    
    def activate(self):
        """Activate the agent"""
        self.is_active = True
        logger.debug(f"Agent {self.name} activated")
    
    def deactivate(self):
        """Deactivate the agent"""
        self.is_active = False
        logger.debug(f"Agent {self.name} deactivated")
    
    def reset(self):
        """Reset agent state"""
        self.state.clear()
        self.task_history.clear()
        self.is_active = False
        logger.info(f"Agent {self.name} reset")

