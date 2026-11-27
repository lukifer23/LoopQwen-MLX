"""Count dependent evidence steps only after the model has received the previous observation."""


def progress(turns, required, accepts):
    position = 0
    for events in turns:
        if position == required:
            break
        # Parallel calls in one assistant message share one prefix. The model first
        # sees their returns together in the next generated assistant turn.
        if any(accepts(position, event) for event in events):
            position += 1
    return position
