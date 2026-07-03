import contextlib
from xml.etree import ElementTree as ET
import rsi_common

def parse_output_xml(path):
    root = ET.parse(path).getroot()
    rows = []
    for suite in root.findall(".//suite"):
        suite_name = suite.get("name", "unknown")
        for test in suite.findall("test"):           # direct children only
            st = test.find("status")
            status = st.get("status", "UNKNOWN") if st is not None else "UNKNOWN"
            rationale = None
            for msg in test.findall(".//msg"):
                txt = (msg.text or "")
                if "RFC_DATA:grading_reason:" in txt:
                    rationale = txt.split("RFC_DATA:grading_reason:", 1)[1].strip()
            if rationale is None and st is not None and (st.text or "").strip():
                rationale = st.text.strip()
            rows.append({"suite_id": suite_name, "test_id": test.get("name", "unknown"),
                         "status": status, "grader_rationale": rationale})
    return rows

def import_results(path, experiment_id, pool, repeat_idx=0):
    rows = parse_output_xml(path)
    inserted = 0
    with contextlib.closing(rsi_common.connect()) as c, c.cursor() as cur:
        for r in rows:
            cur.execute(
                """insert into rsi.test_results
                   (experiment_id, suite_id, test_id, pool, status, grader_rationale, repeat_idx)
                   values (%s,%s,%s,%s,%s,%s,%s)
                   on conflict (experiment_id, suite_id, test_id, pool, repeat_idx) do nothing""",
                (experiment_id, r["suite_id"], r["test_id"], pool, r["status"], r["grader_rationale"], repeat_idx))
            inserted += cur.rowcount
        c.commit()
    return inserted
