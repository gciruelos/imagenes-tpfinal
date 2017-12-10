presentation.pdf: presentation.tex
	-pdflatex -interaction=nonstopmode presentation.tex
	-pdflatex -interaction=nonstopmode presentation.tex
	-pdflatex -interaction=nonstopmode presentation.tex

view: presentation.pdf presentation.tex
	zathura presentation.pdf &

clean:
	rm -f presentacion.pdf
	rm -f presentation.log
	rm -f presentation.nav
	rm -f presentation.out
	rm -f presentation.snm
	rm -f presentation.toc

