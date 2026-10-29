"""Context-local, fail-closed read-only database access for new audit paths."""

from contextlib import contextmanager
from contextvars import ContextVar

DATABASE_READS_ONLY = ContextVar("sciquest_database_reads_only", default=False)


@contextmanager
def database_reads_only():
    token = DATABASE_READS_ONLY.set(True)
    try:
        yield
    finally:
        DATABASE_READS_ONLY.reset(token)
