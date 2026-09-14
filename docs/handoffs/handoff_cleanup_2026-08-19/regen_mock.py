#!/usr/bin/env python
"""Rebuild mock_live.html = current harmonia_min/app_shell.html + the /api mock prelude.

The prelude (a MODEL for Autumn Leaves + a fetch() shim) is kept verbatim in
mock_prelude.html so the mock can be regenerated after every pull without
re-deriving it from a stale snapshot.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHELL = HERE.parent / "harmonia_min" / "app_shell.html"
PRELUDE = HERE / "mock_prelude.html"
OUT = HERE / "mock_live.html"

shell = SHELL.read_text()
anchor = '<div id="app"></div>\n'
assert anchor in shell, "anchor not found in app_shell.html"
out = shell.replace(anchor, anchor + PRELUDE.read_text(), 1)

# A test hook, in the MOCK ONLY — app_shell.html never gets it. Everything the
# acceptance criteria need to measure lives inside the IIFE's closure.
hook = "\n  API._t = { get S(){ return S; }, formRuns, formGroups, paintFormRail,\n"\
       "             paintFormChip, setPlayhead, buildFormRail };\n"
assert out.count("  return API;\n") == 1
out = out.replace("  return API;\n", hook + "  return API;\n", 1)
OUT.write_text(out)
print(f"wrote {OUT} ({len(out.splitlines())} lines) from {SHELL}")
