import logging
from logger import setup_logger

logger = setup_logger()

def needs_hitl(confidence, parsed):
    if confidence < 0.5 or parsed.get("needs_clarification", False):
        logger.warning("HITL triggered (confidence=%.2f)", confidence)
        return True
    return False
