"""Legacy module retained to document the removed ReAct retrieval design.

Retrieval now runs directly in the graph so each request uses the current Qdrant
collection and session filter. A single mandatory retrieval tool did not benefit
from an agent loop and was a source of stale-retriever bugs.
"""

agent_executor = None
