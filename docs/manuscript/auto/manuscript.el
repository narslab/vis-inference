(TeX-add-style-hook
 "manuscript"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt" "twoside")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("fontenc" "T1") ("biblatex" "style=numeric" "backend=biber" "natbib" "sorting=none" "maxcitenames=2" "maxbibnames=99" "doi=false" "isbn=false" "url=false" "eprint=false") ("eucal" "mathscr")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art11"
    "etex"
    "soul"
    "geometry"
    "graphicx"
    "booktabs"
    "calc"
    "lmodern"
    "fontenc"
    "rotating"
    "biblatex"
    "chngcntr"
    "mathtools"
    "caption"
    "amssymb"
    "amsmath"
    "amsthm"
    "bm"
    "eucal"
    "colortbl"
    "color"
    "epstopdf"
    "subfigure"
    "hyperref"
    "enumerate"
    "polynom"
    "polynomial"
    "multirow"
    "minitoc"
    "fancybox"
    "array"
    "multicol"
    "tikz"
    "pgfplots"
    "pgfplotstable"
    "pgfgantt"
    "fancyhdr"
    "paralist"
    "listings")
   (TeX-add-symbols
    '("pdd" 2)
    '("dpd" 2)
    '("pd" 2)
    '("circled" 1)
    '("relph" 1)
    '("hlcyan" 1)
    '("pts" 1)
    "num"
    "osn"
    "dg"
    "lt"
    "rt"
    "pt"
    "tf"
    "fr"
    "dfr"
    "tn"
    "nl"
    "cm"
    "ol"
    "rd"
    "bl"
    "pl"
    "og"
    "gr"
    "nin"
    "la"
    "al"
    "G"
    "bc"
    "ec"
    "p")
   (LaTeX-add-environments
    "question")
   (LaTeX-add-bibliographies
    "../ai-trees-references")
   (LaTeX-add-color-definecolors
    "slblue"
    "deepblue"
    "deepred"
    "deepgreen"))
 :latex)

