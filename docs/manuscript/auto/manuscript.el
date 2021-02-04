(TeX-add-style-hook
 "manuscript"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("ascelike-new" "Journal" "letterpaper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("caption" "figurename=Fig." "labelfont=bf" "labelsep=period") ("hyperref" "colorlinks=true" "citecolor=red" "linkcolor=black")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
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
    "ascelike-new"
    "ascelike-new10"
    "graphicx"
    "booktabs"
    "calc"
    "inputenc"
    "lmodern"
    "fontenc"
    "caption"
    "subcaption"
    "amsmath"
    "rotating"
    "epstopdf"
    "enumerate"
    "multirow"
    "minitoc"
    "fancybox"
    "array"
    "multicol"
    "newtxtext"
    "newtxmath"
    "hyperref"
    "tikz"
    "pgfplots"
    "pgfplotstable"
    "pgfgantt")
   (TeX-add-symbols
    '("pdd" 2)
    '("dpd" 2)
    '("pd" 2)
    '("circled" 1)
    '("relph" 1)
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
   (LaTeX-add-labels
    "tab:classdist"
    "tab:scenarios"
    "tab:comp")
   (LaTeX-add-bibliographies
    "../ai-trees-references"))
 :latex)

