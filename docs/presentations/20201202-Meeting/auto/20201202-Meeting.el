(TeX-add-style-hook
 "20201202-Meeting"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("beamer" "usenames" "dvipsnames" "smaller")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("biblatex" "backend=biber" "style=authoryear" "maxcitenames=2" "maxbibnames=99" "safeinputenc" "url=false" "eprint=false")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-environments-local "semiverbatim")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "beamer"
    "beamer10"
    "biblatex"
    "pgfpages"
    "appendixnumberbeamer"
    "palatino"
    "tipa"
    "graphicx"
    "multirow"
    "paralist"
    "booktabs"
    "amssymb"
    "amsmath"
    "amsthm"
    "mathtools"
    "bm"
    "color"
    "cancel"
    "comment"
    "minitoc"
    "array"
    "enumerate"
    "epstopdf"
    "listings"
    "forest"
    "tikz"
    "tikz-3dplot"
    "pgfplots"
    "pgfplotstable"
    "neuralnetwork"
    "animate")
   (TeX-add-symbols
    '("tikzmark" ["argument"] 2)
    '("nmfr" 3)
    '("pde" 3)
    '("pdd" 2)
    '("dpd" 2)
    '("pd" 2)
    '("circled" 1)
    '("relph" 1)
    "eps"
    "bX"
    "by"
    "bbe"
    "beps"
    "bY"
    "osn"
    "dg"
    "lt"
    "rt"
    "pt"
    "tf"
    "fr"
    "dfr"
    "ul"
    "tn"
    "nl"
    "cm"
    "ol"
    "rd"
    "bl"
    "pl"
    "og"
    "gr"
    "lbl"
    "dca"
    "nin"
    "bc"
    "ec"
    "p"
    "Err"
    "err"
    "la"
    "al"
    "G"
    "si"
    "Si")
   (LaTeX-add-bibliographies
    "bib/references")
   (LaTeX-add-xcolor-definecolors
    "darkcandyapplered"
    "lightcandyapplered"
    "UBCblue"
    "UBCgrey"
    "slblue"
    "deepblue"
    "deepred"
    "deepgreen")
   (LaTeX-add-mathtools-DeclarePairedDelimiters
    '("ceil" "")
    '("floor" "")))
 :latex)

