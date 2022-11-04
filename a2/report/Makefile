TEX = $(shell echo *.tex)
PDF = $(TEX:.tex=.pdf)

all : $(PDF)
.PHONY: clean all

%.pdf: %.tex
	pdflatex $<
	pdflatex $<
	pdflatex $<
	pdflatex $<

clean:
	-rm *.out *.log *.aux 
