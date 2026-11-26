"""Callable read-only properties of dates, time zones, Unicode and installed software."""

import sysconfig
import unicodedata
from datetime import date, datetime
from importlib.metadata import PackageNotFoundError, distributions
from zoneinfo import ZoneInfo

from .tool_session import Tool

# Every operation in a pair accepts the same arguments. The caller supplies the
# exposed names and order. Values come from the live provider during inference.
OPERATIONS = {
    "calendar": (
        ("weekday_number", "ISO weekday number from Monday (1) to Sunday (7)."),
        ("day_of_year", "One-based ordinal day within the calendar year."),
    ),
    "zone": (
        (
            "utc_offset_minutes",
            "Time zone's total offset east of UTC in minutes at local noon.",
        ),
        ("daylight_minutes", "Minutes contributed by daylight saving at local noon."),
    ),
    "unicode": (
        ("character_name", "Official Unicode name of the character."),
        ("general_category", "Two-letter Unicode general category of the character."),
    ),
    "distribution": (
        ("installed_version", "Installed software distribution version string."),
        (
            "number_of_files",
            "Number of entries in the installed distribution's file inventory.",
        ),
    ),
}
ARGUMENTS = {
    "calendar": {"date": "Local date in ISO YYYY-MM-DD format."},
    "zone": {
        "zone": "IANA time zone identifier.",
        "date": "Local date in ISO YYYY-MM-DD format.",
    },
    "unicode": {"character": "A single Unicode character."},
    "distribution": {"distribution": "Installed Python distribution name."},
}


def site_packages():
    return list(dict.fromkeys(sysconfig.get_path(k) for k in ("purelib", "platlib")))


def installed_distribution(name):
    # An adjacent source checkout may contain an egg-info directory. Only the
    # environment's installed files belong to this provider.
    package = next(distributions(name=name, path=site_packages()), None)
    if package is None:
        raise PackageNotFoundError(name)
    return package


def read_value(family, index, kwargs):
    if family == "calendar":
        day = date.fromisoformat(kwargs["date"])
        return day.isoweekday() if index == 0 else day.timetuple().tm_yday
    if family == "zone":
        day = date.fromisoformat(kwargs["date"])
        value = datetime(
            day.year, day.month, day.day, 12, tzinfo=ZoneInfo(kwargs["zone"])
        )
        delta = value.utcoffset() if index == 0 else value.dst()
        return int(delta.total_seconds() / 60)
    if family == "unicode":
        character = kwargs["character"]
        return (
            unicodedata.name(character)
            if index == 0
            else unicodedata.category(character)
        )
    if family == "distribution":
        package = installed_distribution(kwargs["distribution"])
        return package.version if index == 0 else len(package.files or ())
    raise ValueError("Unknown resource family")


def tools(family, *, names=None, order=(0, 1), lookup=read_value):
    """Build the two real tools for a family, optionally with caller-supplied names."""
    if family not in OPERATIONS:
        raise ValueError("Unknown local resource family")
    functions = OPERATIONS[family]
    names = [name for name, _ in functions] if names is None else list(names)
    if len(names) != 2 or len(set(names)) != 2 or sorted(order) != [0, 1]:
        raise ValueError("Each resource family exposes two distinct operations")
    properties = {
        k: {"type": "string", "description": v} for k, v in ARGUMENTS[family].items()
    }
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }

    def callback(index):
        def call(**kwargs):
            return {"value": lookup(family, index, kwargs)}

        return call

    return [Tool(names[i], functions[i][1], schema, callback(i)) for i in order]
