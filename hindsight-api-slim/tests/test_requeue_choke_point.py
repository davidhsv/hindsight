"""Only the requeue helper may put a memory back in the consolidation queue.

The requeue invariant: a superseded or refuted raw fact never has ``consolidated_at IS NULL``.
Requeuing one would hand a retired belief back to consolidation, whose pool no longer shows it
as retired, so the model could rebuild a current observation from it. The invariant is enforced
where NULL is written, so there must be exactly one place that writes it:
``engine/memories/pg/adaptive_claims.py`` (``requeue_units`` / ``requeue_bank``), whose UPDATEs
carry the ``NOT EXISTS (superseded|refuted claim)`` guard. Everything else calls
``MemoriesExtension.requeue_units``.

This test fails on:

* SQL text (a string or f-string) that sets ``consolidated_at = NULL`` anywhere else;
* a ``mark_consolidated(..., when=None)`` call anywhere but the interface's own
  ``requeue_units`` default (for stores that keep no claims).

Migrations are exempt: they run before any claim exists.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "hindsight_api"

#: Files allowed to clear the marker in SQL text: the helper, and nothing else.
SQL_ALLOWED = ("engine/memories/pg/adaptive_claims.py",)
#: ``(file, enclosing function)`` allowed to call ``mark_consolidated(when=None)``.
CALL_ALLOWED = {("engine/memories/base.py", "requeue_units")}
EXEMPT_DIRS = ("alembic/",)

# `\b` keeps `last_consolidated_at = NULL` (the banks table) out of it.
_CLEAR_SQL = re.compile(r"\bconsolidated_at\s*=\s*NULL\b", re.IGNORECASE)


def _strings(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return ["".join(v.value if isinstance(v, ast.Constant) else "{x}" for v in node.values)]
    return []


def _is_clearing_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
    if name != "mark_consolidated":
        return False
    return any(
        kw.arg == "when" and isinstance(kw.value, ast.Constant) and kw.value.value is None for kw in node.keywords
    )


def violations(rel: str, source: str) -> list[str]:
    """Every requeue that bypasses the helper in ``source`` (the file at ``rel`` in the package)."""
    tree = ast.parse(source, filename=rel)
    out: list[str] = []

    def visit(node: ast.AST, function: str | None) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node.name
        if rel not in SQL_ALLOWED:
            for text in _strings(node):
                if _CLEAR_SQL.search(text):
                    out.append(f"{rel}:{node.lineno}: SQL clears consolidated_at (use requeue_units)")
        if _is_clearing_call(node) and (rel, function) not in CALL_ALLOWED:
            out.append(f"{rel}:{node.lineno}: mark_consolidated(when=None) (use requeue_units)")
        # JoinedStr children are its Constant parts: scanning them again would double-report.
        if isinstance(node, ast.JoinedStr):
            return
        for child in ast.iter_child_nodes(node):
            visit(child, function)

    visit(tree, None)
    return out


def test_only_the_requeue_helper_clears_consolidated_at() -> None:
    found: list[str] = []
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(PKG).as_posix()
        if rel.startswith(EXEMPT_DIRS):
            continue
        found.extend(violations(rel, path.read_text()))
    assert not found, "consolidated_at cleared outside requeue_units:\n" + "\n".join(found)


def test_the_helper_is_where_the_allowlist_says() -> None:
    """The allowlist must name the file that really holds the guarded UPDATE, or it rots silently."""
    source = (PKG / SQL_ALLOWED[0]).read_text()
    assert _CLEAR_SQL.search(source)
    assert "NOT EXISTS" in source and "RETIRED_STATUSES_SQL" in source
    base = (PKG / "engine/memories/base.py").read_text()
    assert "async def requeue_units" in base


def test_the_scan_sees_every_form() -> None:
    source = (
        "async def a(conn, t):\n"
        "    await conn.execute(f'UPDATE {t} SET consolidated_at = NULL WHERE id = $1')\n"
        "async def b(conn):\n"
        "    await conn.execute('UPDATE mu SET x = 1, consolidated_at=null')\n"
        "async def c(store):\n"
        "    await store.mark_consolidated(conn=None, fq_table=None, bank_id='b', unit_ids=[], when=None)\n"
        "async def requeue_units(self):\n"
        "    await self.mark_consolidated(conn=None, fq_table=None, bank_id='b', unit_ids=[], when=None)\n"
        "async def d(conn):\n"
        "    await conn.execute('UPDATE banks SET last_consolidated_at = NULL')\n"
        "async def e(store, now):\n"
        "    await store.mark_consolidated(conn=None, fq_table=None, bank_id='b', unit_ids=[], when=now)\n"
    )
    found = violations("engine/somewhere.py", source)
    assert [line.split(":")[1] for line in found] == ["2", "4", "6", "8"]
    # The interface default is the one sanctioned caller of the clearing form.
    assert len(violations("engine/memories/base.py", source)) == 3
