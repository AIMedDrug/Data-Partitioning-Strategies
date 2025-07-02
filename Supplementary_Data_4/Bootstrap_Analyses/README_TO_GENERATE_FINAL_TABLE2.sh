
<<'comment'

    Script to reproduce the information in Table 2 of the manuscript (in LaTeX FMT).

    iTable2.py houses the code used to perform bootstrap analysis.

    If values differ from those in the manuscript (say, by 0.01 in the 95% CIs), that-s
        a matter of the randomization.

comment


for drug in $(echo all xtals axitinib bosutinib dasatinib imatinib nilotinib ponatinib Glide); do

    for method in $(echo FEP+ Prime); do

        ./iTable2.py $drug $method

    done

done
