# Trivial Context Compressor for Dev Agents

A single-file, dependency-free Python tool that shrinks source code before you put it into an LLM
prompt. It deletes a fixed fraction of the whitespace-separated tokens, starting with the ones a
model can most easily regenerate (syntax glue, boilerplate, comment prose) and keeping the ones it
cannot guess (numbers, operators, comparisons, identifiers, string contents, control flow, negations).

- **Deterministic.** Same input and same rate always give the same output. No model calls, no network.
- **Exact budget.** Removes exactly `round(rate * tokens)` tokens.
- **No dependencies.** Standard library only (NLTK is an optional extra for comment scoring).
- **Multi-language.** 18 built-in profiles, each a small JSON file you can edit or extend.
- **Layout-preserving.** Newlines and indentation of surviving tokens are kept.

The output is **not valid code**. It is meant to be read by a model, not compiled. Tell the model
that some tokens were deleted (see [Prompting](#prompting)).

```text
Input (TypeScript)                                  Output at --rate 0.4
// Apply a coupon to the cart subtotal.             Apply subtotal.
export function applyCoupon(cart: Cart,             applyCoupon(cart: Cart, code: string): number
    code: string): number {                           coupon COUPONS[code];
  const coupon = COUPONS[code];                       if (!coupon)
  if (!coupon) {                                        throw Error("unknown coupon + code);
    throw new Error("unknown coupon " + code);      if (subtotal(cart) < coupon.minSubtotal)
  }                                                     return 0;
  if (subtotal(cart) < coupon.minSubtotal) {        return Math.floor((subtotal(cart) * coupon.pct) / 100);
    return 0;
  }
  return Math.floor((subtotal(cart) * coupon.pct) / 100);
}
```

## Install

There is nothing to install. Copy `code_compressor.py` and the `languages/` folder next to it into
your project (they must stay side by side). Tested on Python 3.12; it uses only the standard library
and `from __future__ import annotations`, so recent Python 3 versions should work.

## Usage

### Command line

```bash
python code_compressor.py path/to/file.go --rate 0.4          # language from the file extension
python code_compressor.py - --lang rust --rate 0.4 < main.rs  # stdin needs --lang (or a shebang)
python code_compressor.py app.ts --rate 0.5 -o app.small.ts   # write to a file
python code_compressor.py app.ts --rate 0.5 --stats           # token/char statistics on stderr
python code_compressor.py --list-languages
```

| Option | Default | Meaning |
|---|---|---|
| `path` | | Source file, or `-` for stdin |
| `--rate` | `0.4` | Fraction of tokens to remove, `0.0` to just under `1.0` |
| `--lang` | auto | Language name, alias or extension. Detected from the path or shebang if omitted |
| `--lang-file` | | Path to your own language profile `.json` |
| `--scoring` | `strings` | `strings`, `strings+math` or `basic` (see [Scoring levels](#scoring-levels)) |
| `--prose` | `light` | Scorer for comments and docstrings: `light`, `nltk` or `auto` |
| `-o`, `--output` | stdout | Write the result to a file |
| `--stats` | off | Print language and token/char statistics to stderr |
| `--list-languages` | | List built-in languages and exit |

### Python

```python
from code_compressor import compress, compress_with_stats

small = compress(source_code, rate=0.4, lang="javascript")
small = compress(source_code, rate=0.4, path="app/main.go")    # language from the extension

small, stats = compress_with_stats(source_code, rate=0.4, lang="rust")
# stats: language, tokens_in, tokens_out, tokens_dropped, chars_in, chars_out, char_reduction
```

If the language cannot be detected the `generic` profile is used. Pass `lang=` (or `--lang`) to avoid
that: the generic profile treats both `#` and `//` as comment starts, so for example Python floor
division would be misread as a comment.

## Prompting

Tell the model what happened, for example:

> The code below was compressed by deleting about 40% of its whitespace-separated tokens. Some syntax
> and comments are missing. Reconstruct the meaning as best you can; the numbers, operators and names
> that remain are reliable.

## Choosing a rate

Recommended range: **0.2 to 0.5**. In the benchmarks below every language-aware setting solved
essentially every task at 50% removal. At 65% results get noisier and vary from file to file.

## Languages

Built in: `python`, `javascript`, `typescript`, `java`, `c`, `cpp`, `csharp`, `go`, `rust`, `ruby`,
`php`, `swift`, `kotlin`, `shell`, `sql`, `lua`, `powershell`, plus `generic` as the fallback.

Python is the best-tested profile. TypeScript and Go were also checked (see below). The other
profiles use the same weight scale but have **not** been tested against a model.

### Adding or changing a language

Each profile is a JSON file in `languages/`. Drop in a new file, or pass one with `lang=` /
`--lang-file`. Profiles can `"extends"` another one (`typescript` extends `javascript`, `cpp` extends
`c`). The schema and the weight scale are documented in the `_doc` field of
`languages/generic.json`. In short:

| Field | Purpose |
|---|---|
| `keywords` | `{weight: [words]}`. Higher weight is kept longer (0.3 syntax glue ... 3.0 negation) |
| `operators` | Punctuation-only tokens that carry meaning |
| `line_comments`, `block_comments` | Comment syntax; comment prose is scored low |
| `case_sensitive` | `false` lowercases keyword lookups (SQL, PHP, PowerShell) |
| `string_quotes` | String delimiters; words inside strings are scored as data |
| `numeric_calls` | `true` or a list of math function names (used by `strings+math`) |
| `extensions`, `filenames`, `interpreters`, `aliases` | Language detection |

Unlisted identifiers score 3.0 (4.0 with an underscore), numbers 4.5, operators 3.5.

## Scoring levels

`--scoring` / `scoring=` controls how much language-aware scoring is used.

| Level | What it does |
|---|---|
| `strings` (default) | Words inside string literals (error messages, format strings, keys) are scored as data, never as keywords, and kept longer than identifiers |
| `strings+math` | Also keeps calls such as `Math.floor(`, `.ceil(`, `round(`, `abs(`, `min(`, `max(` ahead of ordinary identifiers. Opt-in: it helped on one test file and hurt on others |
| `basic` | Plain keyword, operator, identifier and number weights only. The simplest and most predictable level |

The Python profile declares neither string quotes nor numeric calls, so Python output is identical at
every level.

## How well does it work?

Benchmark method: a model is given the compressed file and must answer questions about it and write
new code against it. Python tasks are graded by running tests; TypeScript tasks by running the
generated code in Node; Go tasks with `go run`. Compared with deleting the same number of tokens at
random.

**Python** (5 tasks, graded by tests): all tasks solved at 20% to 50% removal and 3 to 4 of 5 at 65%.
Random removal at 50% solved 3 of 10.

**TypeScript and Go** (5 tasks per file, scores out of 5, averaged over 1 to 3 solver runs per cell;
identical prompts differ by about 0.5 between runs):

| File | 65% `strings` | 65% `strings+math` | 65% `basic` | 65% random | 50% `strings` | 50% `strings+math` | 50% `basic` | 50% random |
|---|---|---|---|---|---|---|---|---|
| `store.ts` (in-sample) | 3.1 | 4.9 | 3.9 | 3.0 | 5.0 | 5.0 | 4.6 | 4.0 |
| `rental.ts` (held out) | 5.0 | 3.4 | 4.2 | 2.0 | 5.0 | 5.0 | 4.8 | 2.1 |
| `tariff.go` (held out) | 4.9 | 4.1 | 4.4 | 1.2 | 5.0 | 5.0 | 5.0 | 2.4 |
| **Mean** | **4.3** | 4.1 | 4.2 | 2.1 | **5.0** | 5.0 | 4.8 | 2.8 |

`store.ts` was used to design the string and math features, so it is in-sample; the other two files
were written after the features were frozen. `strings` has the best mean over all six cells (4.67
against 4.55 for `strings+math` and 4.49 for `basic`), which is why it is the default.

Limitations to keep in mind:

- The benchmark is small: three files outside Python, a handful of solver runs each. Treat the
  differences at 65% as suggestive, not proven.
- Only Python, TypeScript and Go were measured. All other language profiles are untested.
- The first word of a string literal glued to code (`Error("unknown`) is scored as an ordinary
  identifier, so it can still be dropped.
- Results depend on the model reading the prompt and on the code being compressed.

## Optional: NLTK comment scoring

By default, comment and docstring text is scored with a small built-in lexicon, so output is identical
on every machine. If you install `nltk` and `wordfreq`, `--prose nltk` (or `--prose auto`) uses part of
speech tags and word rarity instead. This is the scorer used in the original Python benchmark. The
TypeScript and Go results above used the default `light` scorer (one `store.ts` condition also
included an NLTK run).

## Tests

```bash
pip install pytest
python -m pytest -q
```

## Files

| File | Purpose |
|---|---|
| `code_compressor.py` | The compressor (library and CLI) |
| `languages/*.json` | Language profiles |
| `test_code_compressor.py` | Test suite |
