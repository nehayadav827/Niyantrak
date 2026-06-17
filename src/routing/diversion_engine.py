import networkx as nx


def build_corridor_graph():

    graph = nx.Graph()

    edges = [
        ("ORR East 1", "ORR East 2"),
        ("ORR East 1", "Old Airport Road"),
        ("ORR East 1", "Varthur Road"),
        ("ORR East 2", "Old Madras Road"),
        ("ORR East 2", "Hosur Road"),

        ("ORR North 1", "ORR North 2"),
        ("ORR North 1", "Bellary Road 1"),
        ("ORR North 2", "Hennur Main Road"),
        ("ORR North 2", "IRR(Thanisandra road)"),

        ("Bellary Road 1", "Bellary Road 2"),
        ("Bellary Road 2", "Airport New South Road"),
        ("Bellary Road 1", "CBD 1"),
        ("Bellary Road 2", "ORR North 1"),

        ("Tumkur Road", "West of Chord Road"),
        ("Tumkur Road", "ORR West 1"),
        ("West of Chord Road", "Magadi Road"),
        ("ORR West 1", "Mysore Road"),

        ("Mysore Road", "Magadi Road"),
        ("Mysore Road", "CBD 2"),
        ("Mysore Road", "Bannerghata Road"),

        ("CBD 1", "CBD 2"),
        ("CBD 1", "Old Airport Road"),
        ("CBD 2", "Hosur Road"),
        ("CBD 2", "Bannerghata Road"),

        ("Hosur Road", "Bannerghata Road"),
        ("Hosur Road", "ORR East 2"),
        ("Bannerghata Road", "ORR West 1"),

        ("Old Madras Road", "Varthur Road"),
        ("Old Madras Road", "ORR East 2"),
        ("Varthur Road", "Old Airport Road"),

        ("Non-corridor", "CBD 1"),
        ("Non-corridor", "CBD 2"),
        ("Non-corridor", "ORR East 1"),
        ("Non-corridor", "Mysore Road"),
    ]

    for u, v in edges:
        graph.add_edge(
            u,
            v,
            weight=1
        )

    return graph


def resolve_corridor_name(
    graph,
    corridor
):

    corridor_clean = (
        str(corridor)
        .strip()
        .lower()
    )

    for node in graph.nodes:

        if (
            str(node)
            .strip()
            .lower()
            ==
            corridor_clean
        ):
            return node

    return str(corridor).strip()


def get_unique_list(items):

    seen = set()
    result = []

    for item in items:

        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def rank_candidates(
    graph,
    candidates,
    affected_corridor
):

    def score(node):

        degree_score = graph.degree(node)

        fallback_penalty = 0

        if (
            node == "Non-corridor"
            and affected_corridor != "Non-corridor"
        ):
            fallback_penalty = 100

        return (
            fallback_penalty,
            -degree_score,
            node
        )

    return sorted(
        candidates,
        key=score
    )


def recommend_diversions(
    affected_corridor,
    final_risk_level,
    road_closure=False
):

    graph = build_corridor_graph()

    affected_corridor = resolve_corridor_name(
        graph,
        affected_corridor
    )

    if affected_corridor not in graph.nodes:

        return {
            "status": "NO_GRAPH_MATCH",
            "message": "No corridor graph match found. Use manual local diversion planning.",
            "primary_detour": "Manual diversion required",
            "secondary_detour": "Nearest available service road",
            "support_corridors": []
        }

    neighbors = list(
        graph.neighbors(
            affected_corridor
        )
    )

    if not neighbors:

        return {
            "status": "NO_ALTERNATE",
            "message": "No alternate corridor found.",
            "primary_detour": "Manual diversion required",
            "secondary_detour": "Nearest available service road",
            "support_corridors": []
        }

    two_hop = []

    for neighbor in neighbors:

        for candidate in graph.neighbors(neighbor):

            if (
                candidate != affected_corridor
                and candidate not in neighbors
            ):
                two_hop.append(candidate)

    two_hop = get_unique_list(
        two_hop
    )

    ranked_neighbors = rank_candidates(
        graph,
        neighbors,
        affected_corridor
    )

    ranked_two_hop = rank_candidates(
        graph,
        two_hop,
        affected_corridor
    )

    primary = ranked_neighbors[0]

    if len(ranked_neighbors) > 1:
        secondary = ranked_neighbors[1]

    elif ranked_two_hop:
        secondary = ranked_two_hop[0]

    else:
        secondary = "Local service road"

    support_corridors = []

    support_corridors.extend(
        ranked_neighbors[2:]
    )

    support_corridors.extend(
        ranked_two_hop
    )

    support_corridors = [
        corridor
        for corridor in get_unique_list(support_corridors)
        if (
            corridor not in [
                affected_corridor,
                primary,
                secondary
            ]
            and
            (
                corridor != "Non-corridor"
                or affected_corridor == "Non-corridor"
            )
        )
    ]

    support_corridors = support_corridors[:5]

    if final_risk_level == "CRITICAL" or road_closure:

        action = (
            "Activate diversion immediately and place officers at major approach junctions."
        )

    elif final_risk_level == "HIGH":

        action = (
            "Prepare diversion support and deploy officers at entry/exit points."
        )

    elif final_risk_level == "MODERATE":

        action = (
            "Keep diversion route on standby; monitor corridor conditions."
        )

    else:

        action = (
            "No full diversion needed. Continue normal monitoring."
        )

    return {
        "status": "OK",
        "message": action,
        "primary_detour": primary,
        "secondary_detour": secondary,
        "support_corridors": support_corridors
    }