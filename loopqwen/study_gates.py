"""Small fail-closed decisions shared by staged research studies."""


def decide_stage(required, measured):
    failed = [name for name in required if measured.get(name) is False]
    pending = [
        name
        for name in required
        if measured.get(name) is not True and name not in failed
    ]
    # A failed prerequisite is sufficient to close the branch even with other scores pending.
    status = "closed" if failed else "pending" if pending else "eligible"
    return {"status": status, "failed": failed, "pending": pending}
