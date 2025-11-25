"""
Agent Communication Bus
Handles message passing and event coordination between agents
"""

import logging
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the agent bus"""
    TASK_REQUEST = "task_request"
    TASK_RESULT = "task_result"
    STATE_UPDATE = "state_update"
    ERROR = "error"
    COORDINATION = "coordination"


@dataclass
class AgentMessage:
    """Message structure for agent communication"""
    message_id: str
    sender: str
    receiver: Optional[str]  # None for broadcast
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None  # For tracking related messages

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id
        }


class AgentBus:
    """Event bus for agent communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: List[AgentMessage] = []
        self.max_history = 1000
    
    def subscribe(self, agent_name: str, callback: Callable[[AgentMessage], None]):
        """Subscribe an agent to receive messages"""
        if agent_name not in self.subscribers:
            self.subscribers[agent_name] = []
        self.subscribers[agent_name].append(callback)
        logger.debug(f"Agent {agent_name} subscribed to bus")
    
    def unsubscribe(self, agent_name: str, callback: Callable[[AgentMessage], None]):
        """Unsubscribe an agent from receiving messages"""
        if agent_name in self.subscribers:
            try:
                self.subscribers[agent_name].remove(callback)
            except ValueError:
                pass
    
    def publish(self, message: AgentMessage):
        """Publish a message to the bus"""
        # Store in history
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)
        
        # Deliver to specific receiver
        if message.receiver:
            if message.receiver in self.subscribers:
                for callback in self.subscribers[message.receiver]:
                    try:
                        callback(message)
                    except Exception as e:
                        logger.error(f"Error delivering message to {message.receiver}: {e}")
        else:
            # Broadcast to all subscribers
            for agent_name, callbacks in self.subscribers.items():
                if agent_name != message.sender:  # Don't send to sender
                    for callback in callbacks:
                        try:
                            callback(message)
                        except Exception as e:
                            logger.error(f"Error broadcasting message to {agent_name}: {e}")
        
        logger.debug(f"Message {message.message_id} published from {message.sender} to {message.receiver or 'all'}")
    
    def get_messages(self, agent_name: Optional[str] = None, message_type: Optional[MessageType] = None) -> List[AgentMessage]:
        """Get message history filtered by agent or type"""
        messages = self.message_history
        
        if agent_name:
            messages = [m for m in messages if m.sender == agent_name or m.receiver == agent_name]
        
        if message_type:
            messages = [m for m in messages if m.message_type == message_type]
        
        return messages
    
    def clear_history(self):
        """Clear message history"""
        self.message_history.clear()
        logger.debug("Message history cleared")

