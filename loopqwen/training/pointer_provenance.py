"""Restore the frozen pre-migration categorical study and fully qualify new source keys."""

from ..io import digest

# The migration changes only source identity and its validation on restore.
# All original optimizer, objective and cursor code is byte-for-byte unchanged.
MIGRATED_TRAINER = "cdd5dc09c6bae2de6bfded734120fbf9554809b7ad0a2736198f37c59172df2b"


def source_keys(config, files):
    hashed = {name: digest(path) for name, path in files.items()}
    if config.get("pointer_identity_schema") == 2:
        return hashed
    if (
        "pointer_identity_schema" in config
        or hashed["loopqwen.training.pointer5"] != MIGRATED_TRAINER
    ):
        raise ValueError(
            "Legacy categorical checkpoints require the verified compatibility implementation"
        )
    # Reproduce the v1 key collision only for this exact reviewed implementation.
    return {files[name].name: value for name, value in hashed.items()}
