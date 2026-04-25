"""
Indicator phrases shown during active tasks.

Each tuple holds 5 human-readable variations of the same concept.
They cycle every 5-7 seconds while the task is running.

Edit freely — change wording, reorder, remove entries you dislike,
or add new ones. The indicator picks them in order and loops.
"""

MSG_PHRASES = (
    "thinking",
    "pulling context",
    "working through it",
    "on it",
    "chewing on that",
)

FORAGE_PHRASES = (
    "foraging",
    "on the trail",
    "digging in",
    "casting the net",
    "running it down",
)

WEB_PHRASES = (
    "crawling sources",
    "fetching pages",
    "reading the web",
    "checking sources",
    "scraping content",
)

PLAN_PHRASES = (
    "planning",
    "laying it out",
    "mapping the build",
    "sketching it up",
    "drawing up a plan",
)

BUILD_PHRASES = (
    "building",
    "writing code",
    "putting it together",
    "generating files",
    "assembling the pieces",
)

RESEARCH_AGENT_PHRASES = (
    "running agents",
    "agents working",
    "researchers on it",
    "digging deeper",
    "agents in parallel",
)

# Maps task key → phrase tuple.
# Add new task types here if you add new commands.
TASK_PHRASES: dict[str, tuple[str, ...]] = {
    "msg":      MSG_PHRASES,
    "forage":   FORAGE_PHRASES,
    "web":      WEB_PHRASES,
    "plan":     PLAN_PHRASES,
    "build":    BUILD_PHRASES,
    "research": RESEARCH_AGENT_PHRASES,
}

# Seconds between phrase changes (chosen randomly in this range each time).
PHRASE_INTERVAL_MIN = 5
PHRASE_INTERVAL_MAX = 7
