from loopqwen.evaluation.service_answers import visible_reference


def test_echoed_file_and_punctuation_do_not_change_the_quantity():
    assert visible_reference(
        "The file path-01.json has **1,855 bytes**.",
        "data/path-01.json",
        "1855",
        file=True,
    )
    assert not visible_reference(
        "Maybe 1,855 or 1,856", "path-01.json", "1855", file=True
    )
    assert not visible_reference(
        "The file path-01.json", "path-01.json", "1", file=True
    )


def test_version_claim_requires_actual_version_or_observed_absence():
    assert visible_reference("La versión de mlx-lm es 0.32.2.", "mlx-lm", "0.32.2")
    assert visible_reference(
        "It appears requests is not installed.", "requests", "package is not installed"
    )
    assert not visible_reference(
        "It is not installed, but the version is 2.1",
        "requests",
        "package is not installed",
    )
