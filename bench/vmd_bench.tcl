# In-VMD benchmark of the Tcl extension: what a plugin pays, from the frames
# in VMD to the answers.
#
#   vmd -dispdev text -e bench/vmd_bench.tcl -args mdance_tcl.dll pdb dcd seltext mode
#
# mode "stock": the first interface, as the plugin calls it now: the frames
# read with $sel get, made one flat list, passed to every call.
# mode "new": the frames written by VMD to a DCD and read by the library
# into a dataset once; every call takes the dataset; and one long call run
# as a job while the event loop goes on.
lassign $argv lib pdb dcd seltext mode
set env(TMPDIR) [expr {[info exists env(TMPDIR)] ? $env(TMPDIR) : [pwd]}]
load $lib Mdance

mol new $pdb waitfor all
mol addfile $dcd waitfor all
set m top
# (the PDB's own frame dropped: the trajectory's frames only)
animate delete beg 0 end 0 $m
set nf [molinfo $m get numframes]
set sel [atomselect $m $seltext]
set na [$sel num]
puts "BENCH $mode: $nf frames, $na atoms ($seltext)"

proc ms {script} {
    set t0 [clock microseconds]
    uplevel 1 $script
    expr {([clock microseconds] - $t0) / 1000.0}
}
proc row {what t} { puts [format "BENCH %-34s %10.1f ms" $what $t] }

if {$mode eq "stock"} {
    set t [ms {
        set co {}
        for {set f 0} {$f < $nf} {incr f} {
            $sel frame $f
            lappend co {*}[concat {*}[$sel get {x y z}]]
        }
    }]
    row "frames into Tcl (sel get, flat list)" $t
    set D [list $co $nf $na]
} else {
    set tmp [file join $env(TMPDIR) vmd_bench_[pid].dcd]
    set t [ms {
        animate write dcd $tmp beg 0 end -1 skip 1 sel $sel waitfor all $m
        set d [::mdance::data dcd $tmp]
    }]
    file delete $tmp
    row "frames into the library (DCD)" $t
    set D [list $d]
}

set t [ms { set r [::mdance::kmeans {*}$D 10 -kinit StratAll -percentage 10] }]
row "kmeans k=10 StratAll" $t
set labels [dict get $r labels]
set t [ms { for {set k 2} {$k <= 20} {incr k} { ::mdance::kmeans {*}$D $k -kinit StratAll -percentage 10 } }]
row "kmeans scan k=2..20" $t
set t [ms { set r40 [::mdance::kmeans {*}$D 40 -kinit StratAll -percentage 50] }]
set t [ms { ::mdance::helm {*}$D 5 [dict get $r40 labels] -merge-scheme Inter }]
row "helm k=5 from 40" $t
set t [ms { ::mdance::divine {*}$D 10 }]
row "divine k=10 NANI" $t
set t [ms { ::mdance::equal {*}$D -threshold 2.0 }]
row "equal medoid threshold=2" $t
set t [ms { ::mdance::analysis {*}$D -labels $labels }]
row "analysis" $t
set t [ms { ::mdance::prime {*}$D -labels $labels -trim-frac 0.1 -weighted }]
row "prime RR" $t
set t [ms { ::mdance::select {*}$D -method diversity -param 10 }]
row "select diversity 10%" $t

if {$mode eq "new"} {
    # a long call as a job: how often the event loop ran meanwhile
    set ::ticks 0
    proc tick {} { incr ::ticks; after 10 tick }
    after 10 tick
    set t0 [clock milliseconds]
    set j [::mdance::job start -command {set ::over} equal $d -threshold 2.0 -seed-method comp_sim]
    vwait ::over
    set t [expr {[clock milliseconds] - $t0}]
    ::mdance::job result $j
    row "equal comp_sim as a job" $t
    puts [format "BENCH   event loop ran %d times during it (every %.0f ms)" $::ticks [expr {$t / max(1.0, $::ticks)}]]
}
quit
