"""
DAG4 — the best-scoring causal structure from Assignment 3.

Selected by BIC comparison against three alternatives:
    DAG4  24,299  (best)      DAG3  24,946  (+647)
    DAG2  25,569  (+1,270)    DAG1  25,615  (+1,316)

Caveat carried over from Assignment 3: DAG4 is also the densest candidate
(9 edges vs 4-5), and BIC rewards edges whenever they improve fit by more than
the complexity penalty. Part of its advantage is structural richness rather
than demonstrated causal correctness.
"""

import networkx as nx

OUTCOME = "stress_level"

# Treatments whose causal effect we estimate (Assignment 4, Q5)
TREATMENTS = [
    "study_load",
    "noise_level",
    "living_conditions",
    "academic_performance",
    "future_career_concerns",
]

# DAG4 structure.
#   - living_conditions drives the physical environment (noise) and academic performance
#   - study_load reaches stress directly AND via performance -> career concerns
#   - safety, mental_health_history, noise_level, study_load act directly
EDGES = [
    ("safety",                 "stress_level"),
    ("mental_health_history",  "stress_level"),
    ("study_load",             "stress_level"),
    ("noise_level",            "stress_level"),
    ("future_career_concerns", "stress_level"),
    ("living_conditions",      "noise_level"),
    ("living_conditions",      "academic_performance"),
    ("study_load",             "academic_performance"),
    ("academic_performance",   "future_career_concerns"),
]

# Variables appearing in DAG4 (8 of the dataset's 21 columns)
NODES = sorted({n for edge in EDGES for n in edge})


def build_graph() -> nx.DiGraph:
    """Return DAG4 as a networkx DiGraph, validated as acyclic."""
    g = nx.DiGraph(EDGES)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("DAG4 edge list is not acyclic")
    return g


def covariates_for(treatment: str) -> list[str]:
    """
    Candidate covariates for a given treatment: every DAG node that is neither
    the treatment, the outcome, nor a descendant of the treatment.

    Descendants are excluded because conditioning on a variable caused by the
    treatment either blocks part of the effect (mediator) or opens a spurious
    path (collider). DoWhy selects the actual adjustment set itself; this is a
    safe superset to hand to the nuisance learners as X.
    """
    g = build_graph()
    if treatment not in g:
        raise ValueError(f"{treatment!r} is not in DAG4")
    excluded = {treatment, OUTCOME} | nx.descendants(g, treatment)
    return sorted(n for n in g.nodes if n not in excluded)


# Reference: minimal backdoor adjustment sets implied by DAG4.
# NOTE: these are not always unique. For academic_performance the backdoor path
# is  academic_performance <- living_conditions -> noise_level -> stress_level,
# which can be blocked at EITHER living_conditions OR noise_level. DoWhy may
# return a different (equally valid) set than the one listed here, and the two
# will give slightly different finite-sample estimates.
MINIMAL_ADJUSTMENT_SETS = {
    "study_load":             [],
    "living_conditions":      [],
    "noise_level":            ["living_conditions"],
    "academic_performance":   ["living_conditions", "study_load"],   # or {noise_level, study_load}
    "future_career_concerns": ["academic_performance"],
}


if __name__ == "__main__":
    g = build_graph()
    print(f"DAG4: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, acyclic OK")
    print(f"Outcome: {OUTCOME}")
    print()
    for t in TREATMENTS:
        print(f"  {t:<24} covariates: {covariates_for(t)}")
