# Checkpointing

This note is part of the local search corpus for the portfolio demo.

LangGraph checkpoints store the thread state after each step. A thread id points at that history. SQLite checkpointing is enough for this local demo. Resuming a thread loads the saved state, including a run that is waiting on human approval.

The checkpoint does not make the demo a production deployment.
