########################
set terminal postscript landscape enhanced color solid "Text" 14

set output 'lagr-rad.ps'

set zeroaxis
set pointsize 0.5
set grid
set mxtics 10
set mytics 10

set title 'phi-GPU: Plummer, N=10k, G=M=1, E_{tot}=-1/4, {/Symbol e}=10^{-4}, Hermite 6th, {/Symbol h}=0.75, dt_{min}{/Symbol \273}10^{-9}'
########################

########################
set xlabel 't [NB units]'
set ylabel 'lg(Lagrangian radii)'

set xrange [:2700]
set yrange [-3:1]

#set xtic 0.0,2.0,10.0
#set ytic 0.0,0.1,1.0

#set format x "%.2f"
#set format x "10^{%T}"

#set format y "%.1f"
#set format y "10^{%T}"

#set label '' at graph 0.05, 0.83

#set logscale x 10.0
#set logscale y 10.0

#f(x) = 3.0E-08*x

#g(x) = b*x*x
#fit g(x) 'res.dat' u 1:2 via b

set key bottom left

plot [] [] 'lagr-rad.dat' every 3 u 1:(log10($2))  t '0.5%' w l lt 1 lw 3, \
           'lagr-rad.dat' every 3 u 1:(log10($3))  t '  1%' w l lt 2 lw 3, \
           'lagr-rad.dat' every 3 u 1:(log10($4))  t '  2%' w l lt 3 lw 3, \
           'lagr-rad.dat' every 3 u 1:(log10($5))  t '  5%' w l lt 4 lw 3, \
           'lagr-rad.dat' every 3 u 1:(log10($6))  t ' 10%' w l lt 5 lw 3, \
           'lagr-rad.dat' every 3 u 1:(log10($7))  t ' 25%' w l lt 6 lw 3, \
           'lagr-rad.dat' every 3 u 1:(log10($8))  t ' 50%' w l lt 7 lw 3, \
           'lagr-rad.dat' every 3 u 1:(log10($9))  t ' 75%' w l lt 8 lw 3, \
           'lagr-rad.dat' every 3 u 1:(log10($10)) t ' 90%' w l lt 9 lw 3

#plot [] [] 'lagr-rad.dat' u 1:2  t '0.5%' w l lt 1 lw 4, \
#           'lagr-rad.dat' u 1:3  t '  1%' w l lt 2 lw 4, \
#           'lagr-rad.dat' u 1:4  t '  2%' w l lt 2 lw 4, \
#           'lagr-rad.dat' u 1:5  t '  5%' w l lt 3 lw 4, \
#           'lagr-rad.dat' u 1:6  t ' 10%' w l lt 4 lw 4, \
#           'lagr-rad.dat' u 1:7  t ' 25%' w l lt 5 lw 4, \
#           'lagr-rad.dat' u 1:8  t ' 50%' w l lt 6 lw 4, \
#           'lagr-rad.dat' u 1:9  t ' 75%' w l lt 7 lw 4, \
#           'lagr-rad.dat' u 1:10 t ' 90%' w l lt 8 lw 4

#set nologscale x
#set nologscale y
########################

########################
quit
########################
