"""Project the decoded state using the active model and its causal cache."""

def project_output(model, hidden, cache):
    contextual = getattr(model, "project_cached", None)
    return (
        contextual(hidden, cache) if contextual is not None else model.project(hidden)
    )
