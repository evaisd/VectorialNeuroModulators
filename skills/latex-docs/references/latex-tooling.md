# LaTeX Tooling (pdflatex + biblatex)

## Defaults
Use `pdflatex` for compilation and `biblatex` with `biber` for references. Keep edits minimal and template-compatible.

## Minimal Preamble (generic article)
```tex
\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage[backend=biber,style=numeric,sorting=none]{biblatex}
\addbibresource{references.bib}
```

## Math
Use `amsmath` environments such as `equation`, `align`, and `gather`. Always label numbered equations and reference with `\eqref{...}`.

## Figures and Graphics
Prefer vector graphics (`.pdf`, `.svg`) for line art and 300 dpi for raster images (`.png`, `.jpg`). Place graphics under a `figures/` folder and include with:

```tex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/plot.pdf}
  \caption{Caption text.}
  \label{fig:plot}
\end{figure}
```

## Bibliography (biblatex)
Use `\addbibresource{...}` in the preamble and cite with `\cite{key}`. Print references with:

```tex
\printbibliography
```

## Build Sequence
Run the following commands from the project directory:

```
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

## Troubleshooting Checklist
- `! LaTeX Error: File ... not found` means a missing `.tex`, `.bib`, or figure file or a bad path.
- `Undefined control sequence` often means a missing package or a template macro typo.
- `Biber error` usually means the `.bib` file path or citation keys are wrong.
Check the `.log` and `.blg` files for the first relevant error and address that first.
