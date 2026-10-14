# CPython 3.12.15 expanded-corpus rights and provenance

**Official source:** [python/cpython](https://github.com/python/cpython), annotated tag [`v3.12.15`](https://github.com/python/cpython/releases/tag/v3.12.15), peeled commit `e848e4b09ca4bc67c0041c9ce7526dae471d4c36`. Tag metadata date: **2026-09-30 22:04:19 UTC**. Files were retrieved on **2026-10-12 UTC**. Exact source URLs can be formed as `https://github.com/python/cpython/blob/e848e4b09ca4bc67c0041c9ce7526dae471d4c36/<path>`; the paths and SHA-256 hashes are in [manifest.json](manifest.json). The original [LICENSE](raw/LICENSE) and [Doc/license.rst](raw/Doc/license.rst) are retained.

The pinned official license states that Python software **and documentation** are under **Python Software Foundation License Version 2**. Its grant includes reproduction, analysis, and derivative works. Retain the full PSF license and the pinned documentation's notice, **“Copyright © 2001-2023 Python Software Foundation; All Rights Reserved.”** When distributing a derivative, briefly summarize changes. This corpus selects technical prose, removes RST navigation and examples, normalizes inline markup, splits source files, and learns a tokenizer from training documents. The retained root license has a longer enumerated-year copyright notice and remains unmodified.

| Source document | Split | Domain | Names in pinned page header |
| --- | --- | --- | --- |
| `Doc/library/heapq.rst` | Train | Algorithms | Kevin O'Connor; Guido van Rossum; François Pinard; Raymond Hettinger |
| `Doc/library/bisect.rst` | Train | Algorithms | Fred L. Drake, Jr.; Raymond Hettinger |
| `Doc/library/array.rst` | Train | Data structures | None named in header; PSF documentation attribution retained |
| `Doc/library/itertools.rst` | Train | Iterators | Raymond Hettinger |
| `Doc/library/pathlib.rst` | Train | Filesystem | None named in header; PSF documentation attribution retained |
| `Doc/library/json.rst` | Train | Serialization | Bob Ippolito |
| `Doc/library/functools.rst` | Validation | Functional programming | Peter Harris; Raymond Hettinger; Nick Coghlan; Łukasz Langa; Pablo Galindo |
| `Doc/library/collections.rst` | Validation | Data structures | Raymond Hettinger |
| `Doc/library/decimal.rst` | Validation | Numerical computing | Eric Price; Facundo Batista; Raymond Hettinger; Aahz; Tim Peters; Stefan Krah; Raymond D. Hettinger |
| `Doc/library/logging.rst` | Sealed test | Diagnostics | Vinay Sajip |
| `Doc/library/re.rst` | Sealed test | Text processing | Fredrik Lundh; Andrew M. Kuchling |
| `Doc/library/io.rst` | Sealed test | Input/output | Guido van Rossum; Mike Verdone; Mark Russell; Antoine Pitrou; Amaury Forgeot d'Arc; Benjamin Peterson |

For train and validation, [config.json](config.json) gives explicit source-line spans. The sealed test uses a frozen structural rule selecting the first eight clean top-level paragraphs per document; no test prose is displayed in samples or supplied to training/evaluation. The selector rejects directives, examples, tables, hyperlinks, long quotations marked with double quotes, and medical keywords. All source bytes, selected-document text, split files, and transformation rules have recorded hashes. [quality_samples.json](quality_samples.json) shows train/validation examples only. [leakage_audit.json](leakage_audit.json) records cross-split checks.

**Limits:** The official PSF license is evidence for this source repository's documentation terms; it is not an independent audit of every contributor's chain of title or of excluded linked works. The selected train/validation spans received a prose review; sealed test content received only deterministic structural and keyword checks. No figures, code examples, external recipes, or quoted passages from the decimal specification were included. Other CPython files or third-party material require a separate rights audit before use.
