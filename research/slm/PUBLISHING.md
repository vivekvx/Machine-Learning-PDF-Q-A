# Research branch publication boundary

This branch publishes source code, configurations, tests, reports, license and
attribution notices, source identifiers, small synthetic datasets, metrics,
manifests, and checksum ledgers. It does not change the production PDF RAG app
or the frozen experiment results.

The following existing local files are deliberately **not** in Git:

- Virtual environments, package caches, downloaded Qwen weights, model and
  optimizer checkpoint tensors, the large `scratch.zip` archive, and temporary
  logs. They are retained in the working directory; no file was deleted.
- The official QASPER Parquet cache, copied paper sections/answers/evidence in
  curated, pilot, portable, feasibility, training-audit, and retrieval payloads,
  and the human evidence-review CSV. The QASPER dataset license does not
  independently establish redistribution or training rights for every paper.
  IDs, rights classifications, hashes, configurations, and aggregate reports
  remain available for review.
- All sealed test source partitions and raw test predictions. They remain local
  so publication cannot expose a held-out evaluation set.
- The full CPython source archive and selected raw documentation pages. Exact
  upstream URLs and hashes, transformation code, source ledgers, PSF license
  notices, attribution, and derived statistics are published. The extracted
  real-text train and validation files remain local to avoid redistributing
  source text or public author email addresses unnecessarily.
- ACL Anthology's full BibTeX export and policy-page snapshot. Their public
  source URLs and exact snapshot hashes remain in the provenance manifest.
- Local adapter settings containing absolute machine paths and untracked
  environment backup files.

**Reproduction limit:** a clean remote checkout supports source inspection and
data-independent tests. End-to-end checkpoint loading, old artifact-ledger
verification, and data-dependent training/evaluation require the omitted local
files or a new, rights-reviewed rebuild from the pinned official sources. A
checksum ledger that names a local-only file is a record of the original run;
its missing file must not be replaced with invented content or a different
checkpoint under the same name.

Keep the omitted files in a private, access-controlled local or encrypted
backup, with their existing SHA-256 manifests. Copy them back only into the
same ignored relative paths when an authorized researcher needs to reproduce a
run. Do not publish them through a release asset, Git LFS, or another host
without a separate rights and privacy review. The `research/slm/.gitignore`
and repository `.gitignore` exclude these paths from routine staging.
