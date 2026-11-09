"""Plain, synchronous four-domain OPD on a pinned upstream verl."""

VERL_REVISION = "482f86fd09959912c99c0acf2b90329f9e061da7"
DOMAINS = ("math", "if", "conversational_pivot", "logic_algorithmic")
TEACHERS = {
    "math": ("math_step50", 50),
    "if": ("if_step70", 70),
    "conversational_pivot": ("cp_step80", 80),
    "logic_algorithmic": ("logic_algorithmic", 30),
}
ARMS = ("domain_mix", "domain_only")
