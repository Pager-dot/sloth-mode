"""The agents. Each one removes a chore you keep putting off.

* :mod:`sloth.agents.tidy`   — sorts and renames a junk-drawer folder.
* :mod:`sloth.agents.brief`  — turns an overnight pile of chat logs into two lines.
* :mod:`sloth.agents.fridge` — turns a fridge photo into a grocery list.

They all share one rule: propose first, change nothing until told, and write
every change to the journal so it can be undone.
"""

from . import brief, fridge, tidy

__all__ = ["brief", "fridge", "tidy"]
