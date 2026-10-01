"""Tests for code_compressor v1.1 (run: pytest)."""
import json

import pytest

import code_compressor as cc
from code_compressor import (
    available_languages, compress, compress_with_stats, detect_language, get_profile, token_scores,
)

PY = '''\
# Copyright (c) 2024 Example Corp. Internal use only.
# Licensed under the Example Public License.

def total(items, threshold=10):
    """Return the order total.

    A line with exactly threshold units gets the discount.
    """
    out = 0
    for price, qty in items:
        if qty >= threshold:
            out += price * qty * 85 // 100
        else:
            out += price * qty
    return out


if __name__ == "__main__":
    print(total([(500, 10), (1200, 2)]))
'''

JS = '''\
// cart.js
// Copyright 2024 Example Corp. Internal use only.
const LIMIT = 5000;

/* Sum the cart. Lines with enough units get the discount. */
function total(items, threshold = 10) {
  let out = 0;
  for (const [price, qty] of items) {
    if (qty >= threshold) {
      out += price * qty * 85 / 100;
    } else {
      out += price * qty;
    }
  }
  return out;
}
'''

C = '''\
// util.c
// Copyright 2024 Example Corp. Internal use only.
#include <stdio.h>
#define LIMIT 5000

int total(int *items, int n) {
    int out = 0;
    for (int i = 0; i < n; i++) {
        if (items[i] >= 10) {
            out += items[i] * 85 / 100;
        } else {
            out += items[i];
        }
    }
    return out;
}
'''

SQL = '''\
-- report.sql
-- Monthly totals for active customers only.
SELECT c.region, SUM(o.amount) AS total
FROM orders o
JOIN customers c ON c.id = o.customer_id
WHERE o.amount >= 100 AND c.active = 1
GROUP BY c.region
HAVING SUM(o.amount) > 5000
ORDER BY total DESC;
'''

SHELL = '''\
#!/usr/bin/env bash
# Retry the upload a few times before giving up.
MAX=5
for i in $(seq 1 $MAX); do
  if curl -fs --max-time 30 "$URL" > /dev/null; then
    echo "ok after $i tries"
    exit 0
  fi
  sleep 2
done
exit 1
'''


# ------------------------------------------------------------------ python (benchmarked profile)
def test_rate_zero_is_identity_up_to_whitespace():
    assert compress(PY, 0.0, lang="python") == PY.strip()


def test_exact_token_count_removed():
    out, stats = compress_with_stats(PY, 0.4, lang="python")
    assert stats["tokens_dropped"] == round(0.4 * stats["tokens_in"])
    assert len(out.split()) == stats["tokens_out"]
    assert stats["language"] == "python"


def test_deterministic():
    assert compress(PY, 0.5, lang="python") == compress(PY, 0.5, lang="python")


def test_keeps_numbers_and_comparisons():
    out = compress(PY, 0.5, lang="python")
    for must in ("85", "100", ">=", "//"):
        assert must in out


def test_drops_licence_comment_before_code():
    out = compress(PY, 0.5, lang="python")
    assert "Copyright" not in out and "Internal" not in out


def test_docstring_spec_is_kept_longer_than_line_comments():
    items, scores = token_scores(PY, lang="python")
    by_tok = {tok: s for (_, tok), s in zip(items, scores)}
    assert by_tok["exactly"] > by_tok["Copyright"]


def test_monotonic_in_rate():
    sizes = [len(compress(PY, r, lang="python")) for r in (0.1, 0.3, 0.5, 0.7)]
    assert sizes == sorted(sizes, reverse=True)


def test_bad_rate_rejected():
    for r in (-0.1, 1.0, 2):
        with pytest.raises(ValueError):
            compress(PY, r, lang="python")


def test_empty_input():
    assert compress("", 0.5, lang="python") == ""


def test_newlines_not_glued():
    out = compress("a = 1\n# note\nb = 2\n", 0.34, lang="python")
    assert "\n" in out


def test_nltk_backend_runs_if_installed():
    if not cc._nlp_available():
        pytest.skip("nltk/wordfreq not installed")
    assert compress(PY, 0.4, prose="nltk", lang="python")


def test_bad_prose_rejected():
    with pytest.raises(ValueError):
        compress(PY, 0.4, prose="nope", lang="python")


# ------------------------------------------------------------------ profiles
@pytest.mark.parametrize("name", available_languages())
def test_profiles_are_sane(name):
    p = get_profile(name)
    assert p.name == name
    assert p.keywords and p.operators
    assert all(0 <= w <= 5 for w in p.keywords.values())
    assert p.line_comments or p.block_comments


def test_expected_languages_present():
    expected = {"generic", "python", "javascript", "typescript", "java", "c", "cpp", "csharp", "go",
                "rust", "ruby", "php", "swift", "kotlin", "shell", "sql", "lua", "powershell"}
    assert expected <= set(available_languages())


def test_python_profile_matches_v1_weights():
    p = get_profile("python")
    assert p.keywords["def"] == 1.0 and p.keywords["not"] == 3.0 and p.keywords["self"] == 0.3
    assert "//=" in p.operators and "&&" not in p.operators


def test_extends_inherits_and_overrides():
    ts, js = get_profile("typescript"), get_profile("javascript")
    assert ts.keywords["function"] == js.keywords["function"]      # inherited
    assert ts.keywords["interface"] == 2.0 and "interface" not in js.keywords  # added
    assert ts.line_comments == js.line_comments
    cpp, c = get_profile("cpp"), get_profile("c")
    assert "::" in cpp.operators and "::" not in c.operators and "->" in cpp.operators


def test_case_insensitive_profile():
    assert get_profile("sql").keywords["select"] == 1.5
    low = compress("SELECT a FROM t WHERE a > 100", 0.0, lang="sql")
    assert low == "SELECT a FROM t WHERE a > 100"
    sc = dict(zip(*[[t for _, t in token_scores("SELECT x", lang="sql")[0]],
                    token_scores("SELECT x", lang="sql")[1]]))
    assert sc["SELECT"] < sc["x"]


# ------------------------------------------------------------------ detection
def test_detect_by_extension_and_filename():
    assert detect_language(path="a/b/main.go") == "go"
    assert detect_language(path="x.TSX") == "typescript"
    assert detect_language(path="x.hpp") == "cpp"
    assert detect_language(path="x.h") == "c"
    assert detect_language(path="home/.bashrc") == "shell"
    assert detect_language(path="Gemfile") == "ruby"


def test_every_declared_extension_resolves_to_its_language():
    for ext, name in cc._index()["ext"].items():
        assert detect_language(path="file" + ext) == name


def test_detect_by_shebang():
    assert detect_language(text="#!/usr/bin/env python3\nprint(1)") == "python"
    assert detect_language(text="#!/bin/bash\necho hi") == "shell"
    assert detect_language(text="#!/usr/bin/env -S node --flag\n") == "javascript"
    assert detect_language(text="#!/usr/bin/python3.11\n") == "python"


def test_detect_falls_back_to_generic():
    assert detect_language(path="notes.txt") == "generic"
    assert detect_language(text="plain text") == "generic"
    assert detect_language() == "generic"


def test_lang_names_aliases_and_extensions():
    assert get_profile("py").name == "python"
    assert get_profile("c++").name == "cpp"
    assert get_profile("C#").name == "csharp"
    assert get_profile(".rs").name == "rust"
    assert get_profile("JS").name == "javascript"


def test_unknown_explicit_language_raises():
    with pytest.raises(ValueError, match="available"):
        compress("x = 1", 0.2, lang="brainfuck")


def test_path_and_shebang_drive_compress():
    assert compress_with_stats(JS, 0.3, path="cart.js")[1]["language"] == "javascript"
    assert compress_with_stats(SHELL, 0.3)[1]["language"] == "shell"
    assert compress_with_stats("a b c", 0.3)[1]["language"] == "generic"


# ------------------------------------------------------------------ custom profiles
def test_custom_dict_profile():
    prof = {
        "name": "mylang", "line_comments": ["%"], "block_comments": [],
        "keywords": {"3.0": ["unless"], "0.1": ["begin", "end"]},
        "operators": ["<>"],
    }
    p = get_profile(prof)
    assert p.keywords["unless"] == 3.0 and "<>" in p.operators
    src = "begin\n% remove this long explanatory remark entirely\nx <> 10\nend\n"
    out = compress(src, 0.5, lang=prof)
    assert "<>" in out and "10" in out
    assert "%" not in out.split() and "begin" not in out.split() and "end" not in out.split()


def test_custom_profile_file_and_extends(tmp_path):
    f = tmp_path / "mini.json"
    f.write_text(json.dumps({"name": "mini", "extends": "python", "keywords": {"4.0": ["quux"]}}))
    p = get_profile(str(f))
    assert p.keywords["quux"] == 4.0 and p.keywords["def"] == 1.0 and p.line_comments == ("#",)
    assert get_profile(f).name == "mini"


def test_bad_profiles_rejected(tmp_path):
    with pytest.raises(ValueError, match="not a number"):
        get_profile({"name": "x", "keywords": {"heavy": ["a"]}})
    with pytest.raises(ValueError, match="pairs"):
        get_profile({"name": "x", "block_comments": [["only-one"]]})
    with pytest.raises(ValueError, match="parent"):
        get_profile({"name": "x", "extends": "does_not_exist"})
    with pytest.raises(ValueError, match="not found"):
        get_profile(tmp_path / "nope.json")


# ------------------------------------------------------------------ other languages
def _kept(out, *words):
    return [w for w in words if w not in out]


@pytest.mark.parametrize("lang,src,keep", [
    ("javascript", JS, ["5000", "85", ">="]),
    ("c", C, ["5000", "85", ">="]),
    ("sql", SQL, ["100", "5000", ">=", ">"]),
    ("shell", SHELL, ["30", ">"]),
])
def test_numbers_and_operators_survive(lang, src, keep):
    out = compress(src, 0.5, lang=lang)
    assert not _kept(out, *keep), (lang, out)


@pytest.mark.parametrize("lang,src,drop", [
    ("javascript", JS, ["Copyright", "Internal"]),
    ("c", C, ["Copyright", "Internal"]),
    ("sql", SQL, ["Monthly"]),
    ("shell", SHELL, ["Retry"]),
])
def test_line_comment_prose_goes_first(lang, src, drop):
    out = compress(src, 0.5, lang=lang)
    assert all(w not in out for w in drop), (lang, out)


def test_block_comment_tokens_are_prose_scored():
    items, scores = token_scores("x = 1 /* the of a */ y = 2", lang="c")
    sc = {}
    for (_, tok), s in zip(items, scores):
        sc.setdefault(tok, s)
    assert sc["the"] < 0 and sc["of"] < 0 and sc["a"] < 0
    assert sc["x"] >= 3.0 and sc["y"] >= 3.0 and sc["1"] >= 4.5


def test_block_comment_spanning_lines_closes():
    src = "/* the of\n the a */\nx = 1\n"
    items, scores = token_scores(src, lang="c")
    sc = {}
    for (_, tok), s in zip(items, scores):
        sc.setdefault(tok, s)
    assert sc["x"] >= 3.0 and sc["1"] >= 4.5 and sc["the"] < 0


def test_python_floor_division_is_a_comment_only_in_generic():
    code = "n = 10\nm = n // 2\nk = m + 7\n"
    assert "2" in compress(code, 0.3, lang="python")
    items, scores = token_scores(code, lang="generic")
    sc = {tok: s for (_, tok), s in zip(items, scores)}
    assert sc["2"] < 4.5  # in generic, '//' starts a comment, so the 2 is treated as prose


# ------------------------------------------------------------------ v1.2: string literals
def _scores(src, lang, word):
    """Rounded scores of every token equal to `word`, in order of appearance."""
    items, scores = token_scores(src, lang=lang)
    return [round(s, 2) for (_, t), s in zip(items, scores) if t == word]


def test_keyword_inside_multi_token_string_is_data():
    src = 'throw new Error("too heavy for intl");\nfor (;;) {}\n'
    assert _scores(src, "typescript", "for") == [3.5, 2.5]   # string word, then the real loop


def test_string_closing_token_is_data_too():
    src = 'x = "a if b";\n'
    assert _scores(src, "javascript", "if") == [3.5]
    assert _scores(src, "javascript", 'b";') == [3.5]


def test_template_literal_and_escapes():
    src = 'const m = `bad qty for ${sku}`;\nconst n = "say \\"hi\\" or else";\nif (x) {}\n'
    assert _scores(src, "typescript", "for") == [3.5]
    assert _scores(src, "typescript", 'else";') == [3.5]   # still inside the string after \"hi\"
    assert _scores(src, "typescript", "if") == [2.5]     # code after the strings is code again


def test_string_state_resets_at_newline():
    src = 'const a = "unterminated for\nfor (;;) {}\n'
    assert _scores(src, "javascript", "for") == [3.5, 2.5]


def test_single_token_quoted_keyword_is_data():
    assert _scores('const k = "for";\n', "javascript", '"for";') == [3.5]


def test_strings_do_not_touch_comments_or_python():
    # an apostrophe in a line comment must not open a string
    assert _scores("// don't stop\nfor (;;) {}\n", "javascript", "for") == [2.5]
    # python declares no string quotes: scoring identical to v1
    assert get_profile("python").string_quotes == ()
    assert _scores('print("a if b")\n', "python", "if") == [2.5]


def test_string_quotes_inherited_and_validated():
    assert get_profile("typescript").string_quotes == ('"', "'", "`")
    assert get_profile("cpp").string_quotes == get_profile("c").string_quotes
    with pytest.raises(ValueError, match="single characters"):
        get_profile({"name": "x", "string_quotes": ['"""']})
    p = get_profile({"name": "q", "string_quotes": ["'"], "keywords": {"2.5": ["if"]}})
    assert p.string_quotes == ("'",)


def test_version():
    assert cc.__version__ == "1.3.0"

# ------------------------------------------------------------------ v1.3: numeric calls + scoring levels
def _by_token(src, **kw):
    items, scores = token_scores(src, lang="typescript", **kw)
    return {t: round(s, 2) for (_, t), s in zip(items, scores)}


def test_numeric_calls_outrank_identifiers():
    src = "const a = Math.floor(x / 2);\nconst b = foo(x);\nconst m = minSubtotal;\nconst z = FREE_SHIPPING_MIN;\n"
    sc = _by_token(src, scoring="strings+math")
    assert sc["Math.floor(x"] == 4.0
    assert sc["foo(x);"] == 3.0                 # ordinary call: unchanged
    assert sc["minSubtotal;"] == 3.0            # `min` glued to a longer name: not a numeric call
    assert sc["FREE_SHIPPING_MIN;"] == 4.0      # underscore bonus only, no numeric-call match
    assert _by_token(src)["Math.floor(x"] == 3.0   # off by default


def test_numeric_calls_survive_high_compression():
    names = ["".join(chr(97 + (i // 5 ** k) % 5) for k in range(3)) for i in range(40)]  # digit-free
    src = "\n".join(f"const v{n} = helper{n}(a{n}, b{n});" for n in names)
    src += "\nconst r = Math.ceil(total * 8 / 100);\n"
    assert "Math.ceil(total" in compress(src, 0.65, lang="typescript", scoring="strings+math")


def test_python_profile_has_no_numeric_calls_or_strings():
    p = get_profile("python")
    assert p.numeric_calls == () and p.string_quotes == ()
    items, scores = token_scores("x = round(y)\n", lang="python")
    assert {t: round(s, 2) for (_, t), s in zip(items, scores)}["round(y)"] == 3.0


def test_scoring_levels_differ_only_where_expected():
    src = 'throw new Error("too heavy for intl");\nconst a = Math.floor(x / 2);\n'
    default = _by_token(src)
    assert default == _by_token(src, scoring="strings")   # "strings" is the default
    plus, basic = _by_token(src, scoring="strings+math"), _by_token(src, scoring="basic")
    assert plus["Math.floor(x"] == 4.0 and default["Math.floor(x"] == 3.0 and basic["Math.floor(x"] == 3.0
    assert default["for"] == 3.5 and plus["for"] == 3.5     # string handling on
    assert basic["for"] == 2.5                              # basic: `for` is just a keyword
    assert {k: v for k, v in plus.items() if k != "Math.floor(x"} == {
        k: v for k, v in default.items() if k != "Math.floor(x"}   # nothing else differs


def test_scoring_rejects_unknown_value():
    with pytest.raises(ValueError, match="scoring"):
        compress("x = 1", 0.2, lang="python", scoring="0.9")


def test_cli_scoring_option(tmp_path, capsys):
    f = tmp_path / "a.ts"
    f.write_text('const a = Math.floor(x / 2);\nconst b = "for if";\n', encoding="utf-8")
    outs = {}
    for level in ("strings", "strings+math", "basic"):
        assert cc._main([str(f), "--rate", "0.5", "--scoring", level]) == 0
        outs[level] = capsys.readouterr().out
    assert cc._main([str(f), "--rate", "0.5"]) == 0
    assert capsys.readouterr().out == outs["strings"]     # default level
    assert outs["basic"] != outs["strings"]               # string words are scored differently


def test_numeric_calls_profile_validation_and_inheritance():
    assert get_profile("typescript").numeric_calls == get_profile("javascript").numeric_calls
    assert get_profile({"name": "x", "numeric_calls": ["clamp"]}).numeric_calls == ("clamp",)
    assert get_profile({"name": "x", "numeric_calls": False}).numeric_calls == ()
    with pytest.raises(ValueError, match="numeric_calls"):
        get_profile({"name": "x", "numeric_calls": ["a b"]})
    with pytest.raises(ValueError, match="numeric_calls"):
        get_profile({"name": "x", "numeric_calls": "floor"})


def test_json_profile_with_bom_loads(tmp_path):
    f = tmp_path / "bom.json"
    f.write_bytes(b"\xef\xbb\xbf" + b'{"name": "bom", "keywords": {"2.0": ["foo"]}}')
    assert get_profile(f).keywords["foo"] == 2.0
