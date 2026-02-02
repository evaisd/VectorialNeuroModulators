---
name: latex-docs
description: Create and edit LaTeX documents (.tex) with equations, figures/graphics and captions, and bibliographies (biblatex/biber), including working from journal templates and compiling with pdflatex. Use when writing new papers, modifying existing TeX sources, adding math/graphics, or preparing journal-ready LaTeX.
---

# LaTeX Docs

## Overview
Create and edit LaTeX documents with math, graphics, captions, and biblatex. Follow existing templates and compile with `pdflatex` + `biber`.

## Workflow
1. Triage inputs and intent. Identify whether this is a new document or an edit to existing `.tex` sources. List provided assets (`.tex`, `.bib`, figures) and any required template or target journal.
2. If editing, preserve structure. Read the preamble for `\documentclass`, custom macros, and template rules. Make targeted edits and avoid reformatting or renaming user-defined commands unless required.
3. If creating, build a minimal, template-aligned skeleton. Choose the class, add the minimal packages, and scaffold sections. Use the sample preamble and build steps in `references/latex-tooling.md`.
4. Add math and cross-references. Prefer `amsmath` environments and label equations, figures, and sections with `\label{...}` and refer with `\ref{...}` or `\eqref{...}`.
5. Add graphics and captions. Use the `figure` environment, `\includegraphics`, and place `\label` after `\caption`. Ensure figure files exist and are in the correct relative paths.
6. Add bibliography with biblatex. Ensure `\addbibresource{...}` matches the `.bib` filename and use `\cite{...}` keys consistently.
7. Build and verify. Run `pdflatex`, then `biber`, then `pdflatex` twice. If errors occur, summarize the first relevant error and propose the fix.

## Templates
When a journal template is required, prefer the official publisher site and follow its README. Use `references/templates.md` for sourcing guidance and `scripts/fetch_template.py` to download or copy template archives.

## References
- `references/latex-tooling.md` for package choices, preamble patterns, and build commands.
- `references/templates.md` for locating and using journal templates such as PRL/APS REVTeX.

## Scripts
Use `scripts/fetch_template.py` to download or copy a template file or zip archive and optionally extract it into the project.
