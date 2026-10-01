"""code_compressor -- trivial, deterministic context compressor for source code, any language.

Removes a fixed fraction of whitespace-delimited tokens from code, dropping the tokens an LLM can
most easily regenerate (syntax glue, comments, boilerplate prose) and keeping the ones it cannot
guess (numbers, operators, comparison symbols, identifiers, control-flow keywords, negations).

Languages
---------
Each language is a JSON profile in the `languages/` folder next to this file: keyword weights,
meaningful operators, comment syntax, and how to detect the language (extension, filename,
shebang). Built in: python, javascript, typescript, java, c, cpp, csharp, go, rust, ruby, php,
swift, kotlin, shell, sql, lua, powershell, plus `generic` (fallback for unknown languages).
See `languages/generic.json` ("_doc" key) for the schema and the weight scale. Add a language by
dropping a new JSON file in that folder, or pass your own with `lang=` / `--lang-file`.
Profiles may `"extends"` another one (typescript extends javascript, cpp extends c).

Scoring levels (`scoring=` argument, CLI `--scoring`):
  "strings"       (default) string-literal handling: for profiles that declare `string_quotes`,
                  words inside a string literal (error messages, format strings, keys) are scored
                  as data -- never as keywords -- and kept longer than identifiers, because a model
                  can neither guess them nor recover them from context. Tracked per line.
  "strings+math"  additionally, for profiles that set `numeric_calls`, calls such as Math.floor( /
                  .ceil( / round( / abs( / min( / max( are kept ahead of ordinary identifiers.
                  Opt-in: it helped on one test file and hurt on another (see Validation).
  "basic"         neither: only the plain keyword / operator / identifier / number weights. The
                  simplest and most predictable level; reproduces the original v1.0/v1.1 output.
The python profile declares neither feature, so Python output is identical at every level.

Validation: the benchmark (5 Python tasks, LLM solver) solved all tasks at 20-50% of tokens
removed and 3-4 of 5 at 65%; random removal at 50% solved 3 of 10. Python is the best-tested
profile. TypeScript (2 files) and Go (1 file) were each tried with 5 tasks, scored out of 5 and
averaged over 1-3 solver runs per cell (identical prompts differ by about 0.5 between runs):
                    65% removed                         50% removed
                    strings  +math  basic  random      strings  +math  basic  random
  store.ts (TS)*      3.1     4.9    3.9    3.0          5.0     5.0    4.6    4.0
  rental.ts (TS)      5.0     3.4    4.2    2.0          5.0     5.0    4.8    2.1
  tariff.go (Go)      4.9     4.1    4.4    1.2          5.0     5.0    5.0    2.4
  mean                4.3     4.1    4.2    2.1          5.0     5.0    4.8    2.8
  (* store.ts was used to design both features, so it is in-sample; the other two files were
  held out. "+math" = strings+math.)
At 50% every language-aware level beats random removal by 2-3 points. At 65% the levels are within
noise of each other on average, but "strings" has the best mean over all six cells (4.67 against
4.55 and 4.49), which is why it is the default; "strings+math" lost on two of three files because it
spends the budget on Math.* calls at the expense of the operands they apply to. Other languages
are UNTESTED against a solver.
Known gap: the first token of a string literal that is glued to code (`Error("too`) is scored as
an ordinary identifier, so the first word of a message can still be dropped.
Recommended removal range: 0.2 - 0.5.

Usage
-----
    from code_compressor import compress
    small = compress(source_code, rate=0.4, lang="javascript")
    small = compress(source_code, rate=0.4, path="app/main.go")     # language from extension

    python code_compressor.py path/to/file.go --rate 0.4            # language from extension
    python code_compressor.py - --lang rust --rate 0.4 < main.rs    # stdin needs --lang (or a shebang)
    python code_compressor.py --list-languages

If the language cannot be detected the `generic` profile is used. Pass `lang=` to avoid that: the
generic profile treats both `#` and `//` as comment starts, so e.g. Python floor division would be
misread as a comment.

The output is NOT valid code (it is meant to be read by a model, not executed). Tell the model
that some tokens were deleted. Newlines and indentation of surviving tokens are preserved.
"""
from __future__ import annotations

import argparse
import functools
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "compress", "compress_with_stats", "token_scores",
    "Profile", "get_profile", "detect_language", "available_languages",
]
__version__ = "1.3.0"

_STRING_WORD = 3.5   # score of a word inside a string literal (identifiers 3.0, operators 3.5)
_NUMERIC_CALL = 4.0  # score of a rounding/math call such as Math.floor( (numbers are 4.5)
_NUMERIC_CALL_WORDS = ("floor", "ceil", "ceiling", "round", "trunc", "truncate", "abs", "min",
                       "max", "sqrt", "pow", "mod")
_SCORING = ("strings", "strings+math", "basic")

LANG_DIR = Path(__file__).resolve().parent / "languages"

_COMPARISON = re.compile(r"[<>!=]=|[<>%]|//")

# Prose (comments / docstrings) -- small built-in lexicon instead of NLP libraries.
_STOP = frozenset(
    "a an the of to in on at by for with from and or is are was were be been being this that these "
    "those it its as into over under than then so such can will would should could may might do "
    "does did has have had you your we our they their there here also just very".split()
)
_CUE = frozenset(
    "not no never only both either neither half next directly before after until unless except all "
    "none each every more less first last least most still but because same twice per must "
    "inclusive exclusive otherwise raise raises return returns".split()
)


# --------------------------------------------------------------------------- language profiles
@dataclass(frozen=True, eq=False)
class Profile:
    name: str
    keywords: dict          # word -> weight (lowercased when case_sensitive is False)
    operators: frozenset    # punctuation-only tokens that carry meaning
    line_comments: tuple    # prefixes starting a comment that runs to end of line
    block_comments: tuple   # ((open, close), ...)
    case_sensitive: bool = True
    string_quotes: tuple = ()   # single-character string delimiters, e.g. ('"', "'")
    numeric_calls: tuple = ()   # math/rounding function names whose calls are kept (empty = off)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot load language profile {path}: {e}") from e


@functools.lru_cache(maxsize=1)
def _index() -> dict:
    """Detection tables built from the raw profile files (identity fields are never inherited)."""
    idx = {"names": {}, "aliases": {}, "ext": {}, "file": {}, "interp": {}}
    for p in sorted(LANG_DIR.glob("*.json")):
        raw = _read_json(p)
        name = p.stem.lower()
        idx["names"][name] = p
        for key, table in (("aliases", "aliases"), ("extensions", "ext"),
                           ("filenames", "file"), ("interpreters", "interp")):
            for v in raw.get(key, []):
                idx[table][str(v).lower()] = name
    return idx


def available_languages() -> list:
    """Names of all built-in language profiles."""
    return sorted(_index()["names"])


def _flatten_keywords(kw, src) -> dict:
    flat = {}
    for weight, words in (kw or {}).items():
        try:
            w = float(weight)
        except ValueError:
            raise ValueError(f"{src}: keyword weight {weight!r} is not a number") from None
        if not isinstance(words, list):
            raise ValueError(f"{src}: keywords[{weight!r}] must be a list of words")
        for word in words:
            flat[str(word)] = w
    return flat


def _raw_for(name: str) -> dict:
    p = LANG_DIR / f"{name}.json"
    if not p.is_file():
        raise ValueError(f"unknown parent profile {name!r} (looked for {p})")
    return _read_json(p)


def _resolve(raw, src, chain=()) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"{src}: profile must be a JSON object")
    base = {"keywords": {}, "operators": [], "line_comments": [], "block_comments": [],
            "case_sensitive": True, "string_quotes": [], "numeric_calls": False}
    parent = raw.get("extends")
    if parent:
        if parent in chain:
            raise ValueError(f"{src}: circular 'extends' ({' -> '.join(chain + (parent,))})")
        base = _resolve(_raw_for(parent), parent, chain + (parent,))
    out = {k: raw.get(k, base[k]) for k in ("line_comments", "block_comments", "case_sensitive",
                                            "string_quotes", "numeric_calls")}
    out["keywords"] = {**base["keywords"], **_flatten_keywords(raw.get("keywords"), src)}
    out["operators"] = sorted(set(base["operators"]) | set(raw.get("operators", [])))
    out["name"] = str(raw.get("name", src)).lower()
    return out


def _build_profile(raw, src="custom") -> Profile:
    r = _resolve(raw, src, (str(src).lower(),))
    for pair in r["block_comments"]:
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            raise ValueError(f"{src}: block_comments entries must be [open, close] pairs")
    for q in r["string_quotes"]:
        if not (isinstance(q, str) and len(q) == 1):
            raise ValueError(f"{src}: string_quotes entries must be single characters")
    nc = r["numeric_calls"]
    if nc is True:
        nc = list(_NUMERIC_CALL_WORDS)
    elif not nc:
        nc = []
    elif not (isinstance(nc, list) and all(isinstance(w, str) and re.fullmatch(r"\w+", w) for w in nc)):
        raise ValueError(f"{src}: numeric_calls must be true, false or a list of plain names")
    cs = bool(r["case_sensitive"])
    kw = r["keywords"] if cs else {k.lower(): v for k, v in r["keywords"].items()}
    return Profile(
        name=r["name"], keywords=kw, operators=frozenset(r["operators"]),
        line_comments=tuple(r["line_comments"]),
        block_comments=tuple((o, c) for o, c in r["block_comments"]),
        case_sensitive=cs,
        string_quotes=tuple(r["string_quotes"]),
        numeric_calls=tuple(nc),
    )


@functools.lru_cache(maxsize=None)
def _load_named(name: str) -> Profile:
    p = _index()["names"].get(name)
    if p is None:
        raise ValueError(f"language profile {name!r} not found in {LANG_DIR}")
    return _build_profile(_read_json(p), name)


def _canon(lang) -> str:
    s = str(lang).strip().lower()
    idx = _index()
    if s in idx["names"]:
        return s
    if s in idx["aliases"]:
        return idx["aliases"][s]
    ext = s if s.startswith(".") else "." + s
    if ext in idx["ext"]:
        return idx["ext"][ext]
    raise ValueError(f"unknown language {lang!r}; available: {', '.join(available_languages())}")


def detect_language(path=None, text=None) -> str:
    """Guess a language name from a file path (name, extension) or a shebang line.
    Returns "generic" if nothing matches."""
    idx = _index()
    if path:
        p = Path(str(path))
        if p.name.lower() in idx["file"]:
            return idx["file"][p.name.lower()]
        if p.suffix.lower() in idx["ext"]:
            return idx["ext"][p.suffix.lower()]
    if text:
        first = text.lstrip("\ufeff").split("\n", 1)[0].strip()
        if first.startswith("#!"):
            parts = first[2:].split()
            if parts:
                cmd = Path(parts[0]).name
                if cmd == "env":
                    cmd = next((Path(a).name for a in parts[1:] if not a.startswith("-") and "=" not in a), "")
                for cand in (cmd.lower(), re.sub(r"[\d.]+$", "", cmd.lower())):
                    if cand in idx["interp"]:
                        return idx["interp"][cand]
    return "generic"


def get_profile(lang=None, *, path=None, text=None) -> Profile:
    """Resolve `lang` to a Profile. `lang` may be a Profile, a dict (profile JSON), a language
    name / alias / extension, a path to a .json profile file, or None (auto-detect)."""
    if isinstance(lang, Profile):
        return lang
    if isinstance(lang, dict):
        return _build_profile(lang, str(lang.get("name", "custom")))
    if isinstance(lang, Path) or (isinstance(lang, str) and lang.lower().endswith(".json")):
        p = Path(lang)
        if not p.is_file():
            raise ValueError(f"language profile file not found: {p}")
        return _build_profile(_read_json(p), p.stem)
    if lang is None:
        return _load_named(detect_language(path, text))
    return _load_named(_canon(lang))


# --------------------------------------------------------------------------- tokens
def _clean(tok: str) -> str:
    return re.sub(r"^\W+|\W+$", "", tok)


# Optional exact backend: NLTK POS tags + wordfreq rarity (the scorer used in the benchmark).
_NLP_CUE = frozenset(
    "not no never only both either neither half next directly left right before after until unless "
    "except all none each every than more less first last at least most still but because unrelated "
    "same twice per if else return in from to".split()
)


def _nlp_available() -> bool:
    try:
        import nltk  # noqa: F401
        import wordfreq  # noqa: F401
        from nltk.corpus import stopwords
        stopwords.words("english")
        nltk.pos_tag(["test"])
        return True
    except Exception:
        return False


def _nlp_prose_scores(items):
    """Per-token prose scores for ALL tokens (the POS tagger needs full context)."""
    import nltk
    from nltk.corpus import stopwords
    from wordfreq import zipf_frequency

    stop = set(stopwords.words("english"))
    toks = [t for _, t in items]
    words = [_clean(t) for t in toks]
    tags = [p for _, p in nltk.pos_tag([w or t for w, t in zip(words, toks)])]
    seen, scores = set(), []
    for w, t, tag in zip(words, toks, tags):
        lw = w.lower()
        s = 0.0
        if any(ch.isdigit() for ch in t):
            s += 4.0
        if re.search(r"[=\[\](){}<>*+\-/%:]", t) and not w.isalpha():
            s += 2.0
        if tag in ("NNP", "NNPS"):
            s += 3.0
        elif tag.startswith("NN"):
            s += 2.0
        elif tag.startswith("VB"):
            s += 1.5
        elif tag.startswith("JJ") or tag == "CD":
            s += 1.5
        elif tag.startswith("RB"):
            s += 0.8
        if lw in _NLP_CUE:
            s += 2.5
        if lw in stop and lw not in _NLP_CUE:
            s -= 1.5
        if lw:
            s += max(0.0, 6.5 - zipf_frequency(lw, "en")) * 0.4
        if lw and lw not in seen:
            s += 0.5
            seen.add(lw)
        scores.append(s)
    return scores


def _tokenize(text: str):
    """-> list of (leading_whitespace, token)."""
    return [(m.group(1), m.group(2)) for m in re.finditer(r"(\s*)(\S+)", text)]


def _detokenize(items, keep) -> str:
    out, pending = [], ""
    for (ws, tok), k in zip(items, keep):
        if k:
            sep = pending or ws
            if out and not sep:
                sep = " "
            out.append(sep + tok)
            pending = ""
        elif "\n" in ws or "\n" in pending:
            pending = "\n"  # a dropped token must not glue two lines together
    return "".join(out).strip()


def _prose_score(word: str, raw: str, seen: set) -> float:
    lw = word.lower()
    s = 0.0
    if any(c.isdigit() for c in raw):
        s += 4.0
    if lw in _CUE:
        s += 2.5
    if lw in _STOP and lw not in _CUE:
        s -= 1.5
    if lw and lw not in _STOP:
        s += 1.5 + min(len(lw), 10) * 0.15  # longer words are rarer, hence more informative
    if word[:1].isupper() or "_" in word or "`" in raw or "(" in raw:
        s += 1.0  # names, identifiers, code mentions inside prose
    if lw and lw not in seen:
        s += 0.5  # first mention; repeats are recoverable
        seen.add(lw)
    return s


def _find_opener(tok: str, pairs):
    """Earliest block-comment opener inside `tok` -> (open, close, position) or None."""
    best = None
    for o, c in pairs:
        p = tok.find(o)
        if p >= 0 and (best is None or p < best[2] or (p == best[2] and len(o) > len(best[0]))):
            best = (o, c, p)
    return best


_ESCAPED = re.compile(r"\\.")


@functools.lru_cache(maxsize=None)
def _call_regex(words):
    """A function name not glued to a longer identifier: `.floor(`, `(max`, `Math.Round`, but not
    `minSubtotal` or `FREE_SHIPPING_MIN`."""
    return re.compile(r"(?<![A-Za-z0-9_])(?:%s)(?![A-Za-z0-9_])" % "|".join(map(re.escape, words)),
                      re.IGNORECASE)


def _first_quote(tok: str, quotes):
    """First string delimiter in `tok` -> (quote, closed_in_same_token), or None."""
    t = _ESCAPED.sub("", tok)
    best = None
    for q in quotes:
        p = t.find(q)
        if p >= 0 and (best is None or p < best[1]):
            best = (q, p)
    if best is None:
        return None
    q, p = best
    return q, q in t[p + 1:]


def token_scores(text: str, prose: str = "light", lang=None, scoring: str = "strings"):
    """Return (items, scores) for every whitespace token of `text` (higher = keep longer).

    lang: see get_profile(); None auto-detects from a shebang line, else uses "generic".
    prose: scorer for comment/docstring text. "light" = built-in lexicon, no dependencies
    (default, identical on every machine); "nltk" = exact benchmarked scorer, needs nltk+wordfreq;
    "auto" = "nltk" if installed else "light".
    scoring: "strings" (default) = string-literal handling; "strings+math" = also boost numeric/
    rounding calls; "basic" = neither (plain keyword/operator/identifier/number weights).
    """
    if scoring not in _SCORING:
        raise ValueError('scoring must be "strings", "strings+math" or "basic"')
    if prose not in ("light", "nltk", "auto"):
        raise ValueError('prose must be "light", "nltk" or "auto"')
    if prose == "auto":
        prose = "nltk" if _nlp_available() else "light"
    profile = get_profile(lang, text=text)
    items = _tokenize(text)
    nlp = _nlp_prose_scores(items) if prose == "nltk" else None
    kw, ops, cs = profile.keywords, profile.operators, profile.case_sensitive
    line_c, block_c = profile.line_comments, profile.block_comments
    quotes = profile.string_quotes if scoring != "basic" else ()
    call_re = (_call_regex(profile.numeric_calls)
               if profile.numeric_calls and scoring == "strings+math" else None)
    scores, seen = [], set()
    in_comment = False
    block_close = None
    str_close = None   # quote character of the string literal we are inside, if any
    for i, (ws, tok) in enumerate(items):
        if "\n" in ws:
            in_comment = False
            str_close = None
        if line_c and tok.startswith(line_c):
            in_comment = True
        in_block = False
        if block_close is not None:  # inside a block comment / docstring
            in_block = True
            if block_close in tok:
                block_close = None
        elif block_c:
            hit = _find_opener(tok, block_c)
            if hit:
                in_block = True
                o_s, c_s, pos = hit
                if c_s not in tok[pos + len(o_s):]:  # not closed within the same token
                    block_close = c_s
        word = _clean(tok)
        if in_comment or in_block:  # prose inside comments and docstrings
            base = nlp[i] if nlp is not None else _prose_score(word, tok, seen)
            s = base - (2.0 if in_comment else 0.0)
        elif str_close is not None:  # inside a multi-token string literal
            if str_close in _ESCAPED.sub("", tok):
                str_close = None
            if any(c.isdigit() for c in tok):
                s = 4.5
            elif any(c.isalnum() for c in tok):
                s = _STRING_WORD
            else:
                s = 0.4
        else:  # real code
            hit_q = _first_quote(tok, quotes) if quotes else None
            if hit_q and not hit_q[1]:  # this token opens a string that continues
                str_close = hit_q[0]
            key = word if cs else word.lower()
            if any(c.isdigit() for c in tok):
                s = 4.5
            elif key in kw:
                s = _STRING_WORD if hit_q else kw[key]  # a quoted word is data, not a keyword
            elif not any(c.isalnum() for c in tok):
                s = 3.5 if tok in ops else (1.0 if tok == "=" else 0.4)
            else:
                s = 3.0 + (1.0 if "_" in word else 0.0)
                if call_re is not None and not hit_q and call_re.search(tok):
                    s = max(s, _NUMERIC_CALL)  # Math.floor( vs .ceil( changes the answer
            if _COMPARISON.search(tok):
                s += 1.0
        s += 1e-6 * ((i * 2654435761) % 1000)  # deterministic tie-break, spread across the file
        scores.append(s)
    return items, scores


def compress_with_stats(code: str, rate: float = 0.4, prose: str = "light", lang=None, path=None,
                        scoring: str = "strings"):
    """Drop exactly round(rate * n_tokens) whitespace tokens. -> (compressed_text, stats dict).

    `path` is only used to detect the language when `lang` is None.
    `scoring` selects how much language-aware scoring is used (see token_scores)."""
    if not 0.0 <= rate < 1.0:
        raise ValueError("rate must be in [0.0, 1.0)")
    profile = get_profile(lang, path=path, text=code)
    items, scores = token_scores(code, prose, profile, scoring)
    n = len(items)
    n_drop = round(rate * n)
    order = sorted(range(n), key=lambda i: scores[i])
    drop = set(order[:n_drop])
    keep = [i not in drop for i in range(n)]
    out = _detokenize(items, keep)
    stats = {
        "language": profile.name,
        "tokens_in": n, "tokens_out": n - n_drop, "tokens_dropped": n_drop,
        "chars_in": len(code), "chars_out": len(out),
        "char_reduction": round(1 - len(out) / len(code), 3) if code else 0.0,
    }
    return out, stats


def compress(code: str, rate: float = 0.4, prose: str = "light", lang=None, path=None,
             scoring: str = "strings") -> str:
    """Compress source code by removing `rate` (0 <= rate < 1) of its whitespace tokens."""
    return compress_with_stats(code, rate, prose, lang, path, scoring)[0]


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compress source code for LLM context.")
    ap.add_argument("path", nargs="?", help="source file, or - for stdin")
    ap.add_argument("--rate", type=float, default=0.4, help="fraction of tokens to remove (default 0.4)")
    ap.add_argument("--lang", help="language name, alias or extension (default: detect from path/shebang)")
    ap.add_argument("--lang-file", help="path to a custom language profile .json")
    ap.add_argument("--list-languages", action="store_true", help="list built-in languages and exit")
    ap.add_argument("--prose", choices=["light", "nltk", "auto"], default="light",
                    help="scorer for comments/docstrings (default light = no dependencies)")
    ap.add_argument("-o", "--output", help="write to this file instead of stdout")
    ap.add_argument("--scoring", choices=list(_SCORING), default="strings",
                    help="strings (default): string-literal handling; strings+math: also keep "
                         "rounding/math calls; basic: plain keyword/operator/identifier weights only")
    ap.add_argument("--stats", action="store_true", help="print language and token/char statistics to stderr")
    args = ap.parse_args(argv)
    if args.list_languages:
        for name in available_languages():
            print(name)
        return 0
    if not args.path:
        ap.error("path is required (use - for stdin)")
    src = sys.stdin.read() if args.path == "-" else open(args.path, encoding="utf-8").read()
    lang = Path(args.lang_file) if args.lang_file else args.lang
    path = None if args.path == "-" else args.path
    out, stats = compress_with_stats(src, args.rate, args.prose, lang=lang, path=path,
                                     scoring=args.scoring)
    if lang is None and stats["language"] == "generic":
        print("note: language not detected, using the generic profile (use --lang)", file=sys.stderr)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out + "\n")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(out)
    if args.stats:
        print(stats, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
