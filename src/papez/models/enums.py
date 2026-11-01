from enum import Enum


class MemoryStatus(str, Enum):
    TAGGED = "tagged"
    ACTIVE = "active"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    CORE = "core"
    DORMANT = "dormant"
    PRUNED = "pruned"


class EdgeType(str, Enum):
    CAUSED_BY = "caused_by"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    DERIVED_FROM = "derived_from"
    RELATED_TO = "related_to"
    TEMPORAL_SEQUENCE = "temporal_sequence"


CAUSAL_EDGE_TYPES = {EdgeType.CAUSED_BY, EdgeType.SUPPORTS, EdgeType.DERIVED_FROM}

SUPPORTIVE_EDGE_TYPES = {EdgeType.CAUSED_BY, EdgeType.SUPPORTS, EdgeType.DERIVED_FROM, EdgeType.RELATED_TO}

NEGATIVE_EDGE_TYPES = {EdgeType.CONTRADICTS, EdgeType.SUPERSEDES}


class Visibility(str, Enum):
    PRIVATE = "private"
    ORG = "org"


class ReactivationPattern(str, Enum):
    BURST = "burst"
    STEADY = "steady"
    SINGLE = "single"


class SpaceRole(str, Enum):
    OWNER = "owner"     # manage members and settings, archive and erase, plus everything a writer can do
    WRITER = "writer"   # store, amend, pin, delete memories in the space, plus everything a reader can do
    READER = "reader"   # recall, search, traverse, explain


# Role ordering: owner > writer > reader. Compare with
# SPACE_ROLE_RANK[SpaceRole(have)] >= SPACE_ROLE_RANK[need].
SPACE_ROLE_RANK: dict[SpaceRole, int] = {
    SpaceRole.READER: 0,
    SpaceRole.WRITER: 1,
    SpaceRole.OWNER: 2,
}


class ForgettingPolicy(str, Enum):
    RETAIN_ALL = "retain_all"   # nothing in the space is ever pruned (default for spaces)
    STANDARD = "standard"       # the normal forgetting rules apply, same as personal memory


class PrincipalType(str, Enum):
    USER = "user"
    SERVICE = "service"
