"""0390 independent training owners and transactional topology qualification."""
VERSION = "2026-10-21.continuation.1"
BASE = "qwen30b_d4_10k_uniform_ctx64k_seed42_v1"
QUEUE = "R9920261000"
PHASES = (("A", 8, 16), ("B", 16, 16), ("C", 16, 8), ("D", 8, 8), ("E", 8, 16))
TOTAL_STEPS = 468
