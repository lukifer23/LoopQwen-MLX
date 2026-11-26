from importlib.metadata import version

from loopqwen.resource_tools import tools


def test_live_properties_preserve_units_and_caller_selected_names():
    zone, summer = tools("zone")
    args = {"zone": "America/St_Johns", "date": "2025-07-17"}
    assert zone.callback(**args) == {"value": -150}
    assert summer.callback(**args) == {"value": 60}
    category, name = tools("unicode", names=["inspect_a", "inspect_b"], order=(1, 0))
    assert (category.name, category.callback(character="€")) == (
        "inspect_b",
        {"value": "Sc"},
    )
    assert (name.name, name.callback(character="€")) == (
        "inspect_a",
        {"value": "EURO SIGN"},
    )
    weekday, ordinal = tools("calendar")
    assert weekday.callback(date="2024-02-29") == {"value": 4}
    assert ordinal.callback(date="2024-02-29") == {"value": 60}
    installed, inventory = tools("distribution")
    assert installed.callback(distribution="mlx") == {"value": version("mlx")}
    assert inventory.callback(distribution="mlx")["value"] > 0
