"""Local persistence: SQLite event repository and evidence snapshots.

Events are persisted on the node before any optional upstream sync (future phases), and
neither component needs network access.
"""
