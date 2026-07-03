import subprocess, pathlib
import rsi_common

def test_schema_applies_and_tables_exist():
    sql = pathlib.Path(__file__).parent.parent / "sql" / "rsi_bootstrap.sql"
    subprocess.run(
        ["docker", "exec", "-i", "rfc-monorepo-postgres-1", "psql", "-U", "rfc", "-d", "rfc"],
        stdin=open(sql), check=True,
    )
    with rsi_common.connect() as c, c.cursor() as cur:
        cur.execute("select to_regclass('rsi.experiments'), to_regclass('rsi.test_results')")
        a, b = cur.fetchone()
    assert a == "rsi.experiments" and b == "rsi.test_results"

def test_token_len_and_constants():
    assert rsi_common.SALT == "rfc-split-v816"
    assert len(rsi_common.GOLD_SUITES) == 10 and "variables" not in rsi_common.GOLD_SUITES
    assert rsi_common.token_len("hello world") >= 2
