#!/bin/bash
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -output-directory=build main.tex
BIBINPUTS=. bibtex build/main
pdflatex -interaction=nonstopmode -output-directory=build main.tex
pdflatex -interaction=nonstopmode -output-directory=build main.tex
cp build/main.pdf main.pdf
echo "Build complete: main.pdf"
