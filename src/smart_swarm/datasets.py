"""Source-group audit for Smart Swarm episode manifests."""


def audit_lineage(manifests):
    indexed = {row["world_episode_id"]: row for row in manifests}
    if len(indexed) != len(manifests):
        raise ValueError("Duplicate world episode ID")
    groups = {}
    for world, row in indexed.items():
        group = row["scene_group_id"]
        split = row["lineage"]["split"]
        if groups.setdefault(group, split) != split:
            raise ValueError(f"Scene group {group} straddles splits")
        audit_sources(indexed, world, split, ())
    return {"worlds": len(indexed), "scene_groups": len(groups),
            "worlds_by_split": {split: sum(row["lineage"]["split"] == split for row in manifests)
                                for split in sorted(set(groups.values()))}}


def audit_sources(indexed, world, split, ancestors):
    if world in ancestors or world not in indexed:
        raise ValueError(f"Cyclic or unresolved world ancestry: {world}")
    entry = indexed[world]["lineage"]
    if entry["split"] != split:
        raise ValueError(f"Source world {world} straddles splits")
    for source in entry["source_world_ids"]:
        audit_sources(indexed, source, split, (*ancestors, world))
