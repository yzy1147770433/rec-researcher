"""Project-specific exception hierarchy."""


class RecResearcherError(Exception):
    """Base class for expected RecResearcher failures."""


class ConfigurationError(RecResearcherError):
    """Raised when runtime configuration is invalid or incomplete."""


class ProviderError(RecResearcherError):
    """Raised when an external provider operation fails."""


class RetrievalError(RecResearcherError):
    """Raised when document retrieval cannot complete."""


class ReportValidationError(RecResearcherError):
    """Raised when claims or citations violate the report contract."""


class BudgetExceededError(RecResearcherError):
    """Raised when a configured workflow budget is exhausted."""
