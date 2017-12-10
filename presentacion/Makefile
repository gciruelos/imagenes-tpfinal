
IMAGES_PDF = $(wildcard images/*.pdf)

presentation.pdf: presentation.tex $(IMAGES_PDF)
	-pdflatex -interaction=nonstopmode presentation.tex
	-pdflatex -interaction=nonstopmode presentation.tex
	-pdflatex -interaction=nonstopmode presentation.tex

view: presentation.pdf presentation.tex
	zathura presentation.pdf &

clean:
	rm -f presentation.aux
	rm -f presentation.pdf
	rm -f presentation.log
	rm -f presentation.nav
	rm -f presentation.out
	rm -f presentation.snm
	rm -f presentation.toc

