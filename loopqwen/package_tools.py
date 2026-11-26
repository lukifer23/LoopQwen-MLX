"""Read installed Python distributions and their active runtime dependencies."""

from importlib.metadata import PackageNotFoundError, distribution, distributions

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from .tool_session import Tool


def dependencies(installed):
    names = set()
    for entry in installed.requires or []:
        requirement = Requirement(entry)
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
            names.add(canonicalize_name(requirement.name))
    return sorted(names)


def snapshot():
    return {
        canonicalize_name(d.metadata["Name"]): {
            "version": d.version,
            "dependencies": dependencies(d),
        }
        for d in distributions()
        if d.metadata.get("Name")
    }


def tools(record=None):
    schema = {
        "type": "object",
        "properties": {
            "distribution": {
                "type": "string",
                "description": "Python distribution name",
            }
        },
        "required": ["distribution"],
        "additionalProperties": False,
    }

    def get(distribution_name, field):
        if record is not None:
            package = record.get(canonicalize_name(distribution_name))
            return (
                {field: package[field]}
                if package is not None
                else {"error": "package is not installed"}
            )
        try:
            installed = distribution(distribution_name)
        except PackageNotFoundError:
            return {"error": "package is not installed"}
        return {
            field: installed.version if field == "version" else dependencies(installed)
        }

    return [
        Tool(
            "package_dependencies",
            "List the active direct dependencies of an installed Python distribution in alphabetical order.",
            schema,
            lambda distribution: get(distribution, "dependencies"),
        ),
        Tool(
            "package_version",
            "Read the version of an installed Python distribution.",
            schema,
            lambda distribution: get(distribution, "version"),
        ),
    ]
