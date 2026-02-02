# Journal Templates

## Sourcing Templates
Prefer official publisher or society sites. Search for the journal name plus “LaTeX template” and download the latest official archive. Verify license/usage notes and keep README files intact.

## PRL / APS (REVTeX)
Physical Review Letters uses the APS REVTeX class. Search for “APS REVTeX 4.2 download” and use the APS domain for the official package. The archive typically contains:
- `revtex4-2.cls` and related `.bst` / `.sty` files
- Example `.tex` files with required options

If the template assumes BibTeX but the user requires biblatex, confirm compatibility before switching. Otherwise, follow the template defaults.

## Template Integration Checklist
- Keep the original `\documentclass` and required options.
- Keep template-provided `.cls` and `.sty` files alongside the main `.tex`.
- Move your content into the provided example file rather than rewriting the template.
- Compile the vendor’s sample file first to confirm the template works in this environment.

## Using the Fetch Script
Use `scripts/fetch_template.py` to download or copy a template archive and optionally extract it into a project folder.
