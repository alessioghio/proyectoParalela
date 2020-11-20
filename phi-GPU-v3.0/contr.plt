######################## 
set term postscript landscape enhanced color "Text" 14
#set term postscript eps enhanced color solid "Text" 16
#set term postscript portrait enhanced color "Text" 16

#set size 6.0/7.0, 6.0/10.0

set output 'contr.ps'

set zeroaxis
set grid
set pointsize 1.0
set mxtics 10
set mytics 10

set key top right

set title 'phi-GPU: Plummer, N=16k, G=M=1, E_{tot}=-1/4, {/Symbol e}=10^{-4}, Hermite 6th, {/Symbol h}=0.5, dt_{min}{/Symbol \273}10^{-9}'
########################

########################
set xlabel 't'
set ylabel 'Energy'

#set xrange [0.0:50.0]
#set yrange [0.0:0.5]

#set xtic 0.0,2.0,10.0
#set ytic 0.0,0.1,1.0

#set format x "%.1f"
#set format y "%.1f"

#set format x "%.1f"
#set format x "%.1t 10^{%T}"

#set format y "%.1f"
#set format y "10^{%T}"

set key top right

plot [] [] 'contr.dat' every 3 u 1:4 t 'E_{POT}' w l lt 1 lw 4, \
           'contr.dat' every 3 u 1:5 t 'E_{KIN}' w l lt 2 lw 4, \
           'contr.dat' every 3 u 1:6 t 'E_{TOT}' w l lt 3 lw 4
########################

########################
set xlabel 't'
set ylabel '{/Symbol D}E_{TOT}/E_{TOT} [per one TU]'

#set xrange [0.0:300.0]
#set yrange [0.0:0.5]

#set xtic 0.0,2.0,10.0
#set ytic 0.0,0.1,1.0

#set format x "%.1f"
#set format y "%.1f"
#set format y "10^{%T}"

#set logscale x 10.0
#set logscale y 10.0

e0 =-2.5351026501426777E-01

#0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00   -5.0433620529336831E-01  2.5082594027910055E-01 -2.5351026501426777E-01    2.4198740932747897E-15  6.7681538213281381E-16    4.3388596869343493E-05 -2.5971253122400216E-03 -1.0381837947937173E-04   1.46559954E-01 1.32009000E-01 1.20000000E-02 

f(x) = 7.44343e-013*x

#g(x) = a*x
#fit g(x) 'contr.dat' u 1:7 via a

set key top right

plot [] [] 'contr.dat' every 3 u 1:7 t '' w l lt 1 lw 2, \
           f(x) t '' w l lt 2 lw 4

#           g(x) t '' w l lt 3 lw 4

#set nologscale x
#set nologscale y
########################

########################
quit
########################
