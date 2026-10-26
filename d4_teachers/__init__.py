"""Independent, single-node domain teachers on the accepted D4 runtime."""

VERSION = "d4-domain-teachers-v1"
DOMAINS = ("math", "if", "conversational_pivot", "logic_algorithmic")
VALIDATION_COUNTS = dict(zip(DOMAINS, (238, 256, 256, 256)))
