"""Record independently interpretable human wordings and competing denotations before model testing."""

from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from loopqwen.evaluation.mechanism import unpack
from loopqwen.io import digest, read
from loopqwen.relational_tools import select
from scripts.dialogue_database_sources import CACHE
from scripts.dialogue_database_suite import SUITE
from scripts.provider_recovery_field import PANEL
from scripts.table_policy_material import persist

READINGS = {
    "how many students are enrolled in a bachelor degree program?": (
        "entity_count",
        "The reference counts enrollment records. The request names students, who may have several enrollments.",
        "SELECT COUNT(DISTINCT e.student_id) FROM Student_Enrolment e JOIN Degree_Programs d ON e.degree_program_id=d.degree_program_id WHERE d.degree_summary_name='Bachelor'",
    ),
    "How many graduates of the school are there?": (
        "unsupported_premise",
        "The reference counts all students. The schema records dates of departure and transcripts but supplies no explicit graduation indicator.",
        None,
    ),
    "how many customers live in germany and have invoice": (
        "entity_count",
        "The reference counts invoices belonging to customers. The request counts the customers who have at least one invoice.",
        "SELECT COUNT(DISTINCT Customer.CustomerId) FROM Customer JOIN Invoice ON Customer.CustomerId=Invoice.CustomerId WHERE Country='Germany'",
    ),
    "How many exhibitions have an attendance of more than 100?": (
        "entity_count",
        "The reference counts dated attendance rows. An exhibition can appear on multiple dates.",
        "SELECT COUNT(DISTINCT exhibition.Exhibition_ID) FROM exhibition JOIN exhibition_record USING(Exhibition_ID) WHERE Attendance>100",
    ),
    "Hi, what journalist had worked the most amount of years?": (
        "identification",
        "The expert identifies one journalist by ID. A reader can identify the same journalist by name and may identify several tied journalists.",
        "SELECT Name FROM journalist WHERE Years_working=(SELECT MAX(Years_working) FROM journalist) ORDER BY Name",
    ),
    "What is the birth date of the poker player with the lowest earnings?": (
        "extremum_ties",
        "The reference takes the first row at the minimum. A tied minimum would have more than one valid player.",
        "SELECT DISTINCT Birth_Date FROM people JOIN poker_player USING(People_ID) WHERE Earnings=(SELECT MIN(Earnings) FROM poker_player) ORDER BY Birth_Date",
    ),
    "Hi there!  Can you show me the name of the architect who built the most mills?": (
        "extremum_ties",
        "The reference takes one architect. A tied count of mills can have several legitimate names.",
        "WITH counts AS (SELECT architect.id AS id, architect.name AS name, count(*) AS n FROM architect JOIN mill ON architect.id=mill.architect_id GROUP BY architect.id) SELECT name FROM counts WHERE n=(SELECT MAX(n) FROM counts) ORDER BY name",
    ),
}


def run():
    oracle = PANEL.with_name("field-denotations.json")
    by_split = read(oracle)["panels"]
    records = []
    with (
        ZipFile(CACHE) as original,
        ZipFile(SUITE) as suite,
        TemporaryDirectory() as tmp,
    ):
        for split, panel in (
            ("dev", PANEL),
            ("train", PANEL.with_name("withheld-training-split-requests.json.gz")),
        ):
            grouped = defaultdict(list)
            for r in unpack(panel)["rows"]:
                if r["condition"] == "unassisted":
                    grouped[r["source"]].append(r)
            for db, pair in grouped.items():
                src = by_split[split]["by_database"][db]
                path = Path(tmp) / (db + ".sqlite")
                for row in pair:
                    item = {
                        "id": row["id"],
                        "question": row["question"],
                        "reference": row["expert_sql"],
                        "reference_answer": row["gold"],
                    }
                    if row["question"] in READINGS:
                        tag, why, query = READINGS[row["question"]]
                        item.update(
                            tag=tag, reason=why, alternate_sql=query, executions=[]
                        )
                        if query:
                            for archive, entry in [(original, src["original"])] + [
                                (suite, e)
                                for e in src["distilled"]
                                if not e["named_original"]
                            ]:
                                path.write_bytes(archive.read(entry["member"]))
                                assert digest(path) == entry["sha256"]
                                gold = select(path, row["expert_sql"])["rows"]
                                alternative = select(path, query)
                                assert not alternative["truncated"]
                                item["executions"].append(
                                    {
                                        "member": entry["member"],
                                        "reference": gold,
                                        "alternate": alternative["rows"],
                                    }
                                )
                    else:
                        item["tag"] = "no_specific_ambiguity_flagged"
                    records.append(item)
    assert len(records) == 24 and sum(
        r["tag"] != "no_specific_ambiguity_flagged" for r in records
    ) == len(READINGS)
    output = {
        "source": digest(oracle),
        "procedure": "One project reviewer read all 24 original requests and expert SQL before either model answered. Keep all in the primary source-reproduction score. Disclose alternate entity or identifier readings and show the unflagged subset separately. These alternatives are project interpretations, not human annotations by the CoSQL authors.",
        "items": sorted(records, key=lambda r: r["id"]),
    }
    persist(PANEL.with_name("field-wording.json"), output)
    for r in records:
        if r.get("executions"):
            xs = r["executions"]
            print(
                r["question"],
                xs[0]["reference"],
                xs[0]["alternate"],
                sum(x["reference"] != x["alternate"] for x in xs),
                "/",
                len(xs),
            )


if __name__ == "__main__":
    run()
