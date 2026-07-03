import import_results as ir

MINIMAL = """<?xml version="1.0" encoding="UTF-8"?>
<robot generator="Robot 7.0">
  <suite name="Math Tests" id="s1">
    <test name="Basic Addition" id="s1-t1">
      <status status="PASS" start="1" end="2"/>
    </test>
    <test name="Hard One" id="s1-t2">
      <kw name="Ask LLM"><msg>RFC_DATA:grading_reason:wrong number</msg></kw>
      <status status="FAIL">Expected 5 got 4</status>
    </test>
    <status status="FAIL"/>
  </suite>
</robot>"""

def test_parse_per_test(tmp_path):
    p = tmp_path / "output.xml"; p.write_text(MINIMAL)
    rows = ir.parse_output_xml(str(p))
    by = {r["test_id"]: r for r in rows}
    assert by["Basic Addition"]["status"] == "PASS"
    assert by["Hard One"]["status"] == "FAIL"
    assert "wrong number" in (by["Hard One"]["grader_rationale"] or "")
