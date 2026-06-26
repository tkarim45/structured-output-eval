"""soeval — benchmark LLM structured-output reliability.

Measures valid-JSON rate, JSON-Schema adherence, and field-level accuracy across prompting
strategies (plain → schema-in-prompt → native structured outputs → strict tool use) under
clean and adversarial inputs.
"""

__version__ = "0.1.0"
