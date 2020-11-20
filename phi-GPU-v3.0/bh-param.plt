######################## 
set term postscript landscape enhanced color "Text" 16
#set term postscript eps enhanced color "Text" 16
#set term postscript portrait enhanced color "Text" 16

#      set terminal postscript {<mode>} {enhanced | noenhanced}
#                              {color | monochrome} {solid | dashed}
#                              {<duplexing>}
#                              {"<fontname>"} {<fontsize>}

#set size 6.0/7.0, 6.0/10.0

set output 'bh-param.ps'

set zeroaxis
set grid
set pointsize 1.0
set mxtics 10
set mytics 10

set key right

set title 'm_{BH1}=m_{BH2}=0.1; {/Symbol e}=1e-4; dt = MIN({/Symbol e}/V_i; (2*{/Symbol e}/A_i)^{0.5})'
########################

########################
set xlabel 't'
set ylabel '{/Symbol D}R_{BH}'

set xrange [0.0:100.0]
#set yrange [1e-2:10.0]

#set xtic 0.0,50.0,300.0
#set ytic 0.0,0.1,1.0

#set logscale x 10.0
set logscale y 10.0

#set format x "%.1f"
#set format x "10^{%T}"

#set format y "%.1f"
#set format y "10^{%T}"

set key top right

plot [] [] 'bh-param.dat' u 1:2 t '' w l lt 1 lw 4

#plot [] [] 'bh-evol.dat' u 1:($2>1e-4 ? $2 : 1/0) t '' w l lt 1 lw 4
           
#set nologscale x
set nologscale y
########################

########################
set xlabel 't'
set ylabel '1/a'

set xrange [0.0:100.0]
#set yrange [1e-2:10.0]

#set xtic 0.0,50.0,300.0
#set ytic 0.0,0.1,1.0

#set logscale x 10.0
#set logscale y 10.0

#set format x "%.1f"
#set format x "10^{%T}"

#set format y "%.1f"
#set format y "10^{%T}"

set key top right

plot [] [] 'bh-param.dat' u 1:4 t '' w l lt 1 lw 4

#plot [] [] 'bh-evol.dat' u 1:($2>1e-4 ? $2 : 1/0) t '' w l lt 1 lw 4
           
#set nologscale x
#set nologscale y
########################

########################
set xlabel 't'
set ylabel 'ABS(EB)'

#set xrange [0.0:20000.0]
#set yrange [1e-2:10.0]

#set xtic 0.0,50.0,300.0
#set ytic 0.0,0.1,1.0

#set logscale x 10.0
#set logscale y 10.0

#set format x "%.1f"
#set format x "10^{%T}"

#set format y "%.1f"
#set format y "10^{%T}"

set key top right

#plot [] [] 'bh-param.dat' u 1:(abs($5)) t '' w l lt 1 lw 4

#plot [] [] 'bh-evol.dat' u 1:($2>1e-4 ? $2 : 1/0) t '' w l lt 1 lw 4
           
#set nologscale x
#set nologscale y
########################

########################
set xlabel 't'
set ylabel 'e'

set xrange [0.0:100.0]
set yrange [1e-2:1.0]

#set xtic 0.0,50.0,300.0
#set ytic 0.0,0.1,1.0

#set logscale x 10.0
#set logscale y 10.0

#set format x "%.1f"
#set format x "10^{%T}"

#set format y "%.1f"
#set format y "10^{%T}"

set key top right

plot [] [] 'bh-param.dat' u 1:6 t '' w l lt 1 lw 4

#plot [] [] 'bh-evol.dat' u 1:($2>1e-4 ? $2 : 1/0) t '' w l lt 1 lw 4
           
#set nologscale x
#set nologscale y
########################

########################
quit
########################
