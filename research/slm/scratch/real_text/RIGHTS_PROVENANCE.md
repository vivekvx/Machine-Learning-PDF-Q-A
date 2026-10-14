# Python documentation pilot rights and provenance

Source: official [CPython repository](https://github.com/python/cpython), annotated tag [`v3.12.15`](https://github.com/python/cpython/releases/tag/v3.12.15), peeled commit `e848e4b09ca4bc67c0041c9ce7526dae471d4c36`. The tag metadata records **2026-09-30 22:04:19 UTC**; source bytes were retrieved **2026-10-12 04:55 UTC**. Each exact source URL and SHA-256 appears in [`v1/manifest.json`](v1/manifest.json). The raw [`LICENSE`](v1/raw/LICENSE) and [`Doc/license.rst`](v1/raw/Doc/license.rst) are retained with the corpus.

The pinned license says Python software **and documentation** are under **Python Software Foundation License Version 2**. It permits reproduction, analysis, derivative works, and other use subject to its terms. Retain the complete PSF license and its copyright notice: **“Copyright © 2001-2023 Python Software Foundation; All Rights Reserved”** as printed in the pinned documentation license. For distributed derivatives, provide a brief summary of changes. This project extracts selected prose, strips RST markup, and forms train/validation documents and tokens; that is the change summary. The root license has a longer enumerated-year copyright form, which is also retained verbatim.

| Selected document | Split | Attribution visible in pinned page header | Source-matching confidence |
| --- | --- | --- | --- |
| `Doc/library/heapq.rst` | Train | Kevin O'Connor; Guido van Rossum; François Pinard; Raymond Hettinger | Exact commit path and SHA-256 |
| `Doc/library/bisect.rst` | Train | Fred L. Drake, Jr.; Raymond Hettinger | Exact commit path and SHA-256 |
| `Doc/library/array.rst` | Train | No named author in page header; Python Software Foundation documentation | Exact commit path and SHA-256 |
| `Doc/library/functools.rst` | Validation | Peter Harris; Raymond Hettinger; Nick Coghlan; Łukasz Langa; Pablo Galindo | Exact commit path and SHA-256 |

Only the explicit line spans in [`v1/config.json`](v1/config.json) are included. They were inspected as technical prose. We excluded code examples, RST directives, tables, see-also lists, external recipe links, figures, and generated navigation. No medical material or QASPER text is included. Exact paragraph duplicates are rejected. The source files remain available for auditing every selected span.

**Limit:** The official documentation license establishes the repository's stated terms for these documentation pages. We did not independently reconstruct every contributor's copyright chain or investigate unquoted facts in the prose. This small pilot is restricted to the reviewed spans. Adding other CPython pages or third-party snippets requires a new source and rights audit. The license of excluded incorporated software is not inferred from the documentation grant.
