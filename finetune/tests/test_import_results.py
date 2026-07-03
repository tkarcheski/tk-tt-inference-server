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

NESTED = """<?xml version="1.0" encoding="UTF-8"?>
<robot generator="Robot 7.0">
  <suite name="Root" id="s1">
    <suite name="Child A" id="s1-s1">
      <test name="a1" id="s1-s1-t1"><status status="PASS"/></test>
    </suite>
    <suite name="Child B" id="s1-s2">
      <test name="b1" id="s1-s2-t1"><status status="FAIL">boom</status></test>
    </suite>
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

def test_nested_suites_attributed_to_direct_parent(tmp_path):
    p = tmp_path / "output.xml"; p.write_text(NESTED)
    rows = ir.parse_output_xml(str(p))
    by = {r["test_id"]: r for r in rows}
    assert set(by) == {"a1", "b1"}
    assert by["a1"]["suite_id"] == "Child A" and by["a1"]["status"] == "PASS"
    assert by["b1"]["suite_id"] == "Child B" and by["b1"]["status"] == "FAIL"

def test_missing_msg_yields_none_rationale(tmp_path):
    p = tmp_path / "output.xml"; p.write_text(MINIMAL)  # reuse the existing fixture
    by = {r["test_id"]: r for r in ir.parse_output_xml(str(p))}
    assert by["Basic Addition"]["grader_rationale"] is None
