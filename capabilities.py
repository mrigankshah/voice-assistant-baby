"""Supported workflow routes, independent of classifier weights.

This registry is used by the experimental router. It does not execute tools.
Argument validation and authorization remain the workflow's responsibility.
"""

CAPABILITIES = {
    "weather": {"description": "Get current weather or a forecast", "enabled": True},
    "clock": {"description": "Read the current time or date", "enabled": True},
    "read_setting": {"description": "Read saved weather preferences", "enabled": True},
    "write_setting": {"description": "Change saved city or temperature units", "enabled": True},
    "timer": {"description": "Create, list, or cancel local timers", "enabled": True},
    "alarm": {"description": "Create, list, or cancel local alarms", "enabled": True},
}
NON_ACTIONS = {"chat", "unsupported_action", "uncertain"}


def resolve_route(intent, registry=None):
    """Return a decision only; never execute an action based on a model label."""
    registry = CAPABILITIES if registry is None else registry
    if intent in NON_ACTIONS:
        return intent
    capability = registry.get(intent)
    if capability is None:
        return "uncertain"
    return intent if capability["enabled"] else "unsupported_action"
