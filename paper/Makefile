.PHONY: all clean watch

PAPER=main
OUTPUT=../output/pdf/n-cmapss-rul-paper-draft.pdf

all: $(PAPER).pdf
	@mkdir -p ../output/pdf
	cp $(PAPER).pdf $(OUTPUT)

$(PAPER).pdf: $(PAPER).tex sections/*.tex references.bib
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -xelatex -shell-escape -interaction=nonstopmode $(PAPER).tex; \
	else \
		tectonic $(PAPER).tex; \
	fi

watch:
	latexmk -xelatex -shell-escape -pvc -interaction=nonstopmode $(PAPER).tex

clean:
	@if command -v latexmk >/dev/null 2>&1; then latexmk -C; fi
	rm -f $(PAPER).pdf *.bbl *.blg *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz
	rm -f $(OUTPUT)
