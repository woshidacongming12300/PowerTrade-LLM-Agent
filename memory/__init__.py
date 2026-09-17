"""Long-term memory package: SQLite-backed persistent memory for agents.

Three stores:
- user_preferences: key-value preferences extracted from conversation
  (e.g. lang=zh, region=ERCOT, currency=USD) — injected into agent prompts.
- prediction_history: every forecast the agent produced, for audit/tracing.
- conversation_memory: compact summaries of past sessions per user.

Thread-safe via a single connection with check_same_thread=False guarded by
a lock (agents run on multiple threads under uvicorn).
"""
