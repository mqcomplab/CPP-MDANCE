# Tests of the Tcl extension (tcl/mdance_tcl.cpp):
#   tclsh tests/test_tcl.tcl path/to/mdance_tcl.so|.dll
# Prints one line per check and "RESULT: n passed, m failed"; exits 1 on a
# failure.

if {[llength $argv] != 1} {
    puts stderr "usage: tclsh test_tcl.tcl path/to/mdance_tcl library"
    exit 2
}
load [lindex $argv 0] Mdance

set ::pass 0
set ::fail 0
proc check {what ok} {
    if {$ok} { incr ::pass; puts "ok    $what" } else { incr ::fail; puts "FAIL  $what" }
}
# the error message of a script, "" when it succeeds
proc errof {script} {
    if {[catch {uplevel 1 $script} m]} { return $m }
    return ""
}

# Two well separated blobs of 3-atom frames, deterministic.
proc blobs {n} {
    set co {}
    for {set f 0} {$f < $n} {incr f} {
        set off [expr {$f < $n / 2 ? 0.0 : 10.0}]
        for {set j 0} {$j < 9} {incr j} {
            lappend co [expr {$off + 0.1 * sin($f * 1.7 + $j * 0.9) + 0.01 * $j}]
        }
    }
    return $co
}
set nf 40
set co [blobs $nf]

# ---------------------------------------------------------------- basics
check "version is 2.x" [string match 2.* [::mdance::version]]
set inf [::mdance::info]
check "info has the API version 2" [expr {[dict get $inf api] == 2}]

# ---------------------------------------------------------------- the first interface
set r [::mdance::kmeans $co $nf 3 2 -kinit StratAll -percentage 50]
set lab [dict get $r labels]
check "kmeans: one label per frame" [expr {[llength $lab] == $nf}]
check "kmeans: the two blobs are the two clusters" [expr {[llength [lsort -unique [lrange $lab 0 19]]] == 1 &&
    [llength [lsort -unique [lrange $lab 20 end]]] == 1 && [lindex $lab 0] != [lindex $lab end]}]
check "kmeans: the result keys of version 1" [expr {[dict exists $r clusterSizes] && [dict exists $r representatives] &&
    [dict exists $r clusterMSD] && [dict exists $r score_calinskiHarabasz] && [dict exists $r score_daviesBouldin]}]

# the coordinates are never made into text (a list of 2 million numbers
# was: 38 MB of text, kept with the list until VMD quit)
proc fresh {n} { set l {}; for {set i 0} {$i < $n} {incr i} { lappend l [expr {double($i % 7)}] }; return $l }
# (inside a procedure: VMD's -e makes the text of every top-level
# command's result, the list's too)
proc textless {} {
    set big [fresh 90]
    ::mdance::kmeans $big 10 3 2 -percentage 50 -kinit StratAll
    ::tcl::unsupported::representation $big
}
check "the coordinates' list has no text after a call" [string match "*no string representation*" [textless]]

# a command renamed (a caller's wrapper) is still its call
rename ::mdance::kmeans ::mdance::kmeans_kept
proc ::mdance::kmeans {args} { ::mdance::kmeans_kept {*}$args }
check "a renamed call still runs" [expr {[dict get [::mdance::kmeans $co $nf 3 2 -kinit StratAll -percentage 50] labels] eq $lab}]
rename ::mdance::kmeans {}
rename ::mdance::kmeans_kept ::mdance::kmeans

# ---------------------------------------------------------------- datasets
set d [::mdance::data list $co $nf 3]
check "data list makes a dataset command" [expr {[info commands $d] eq $d}]
check "data info" [expr {[dict get [$d info] nframes] == $nf && [dict get [$d info] natoms] == 3 && [dict get [$d info] nset] == $nf}]
set r2 [::mdance::kmeans $d 2 -kinit StratAll -percentage 50]
check "kmeans on a dataset = kmeans on the list" [expr {[dict get $r2 labels] eq $lab}]
check "get returns a frame" [expr {[llength [$d get 5]] == 9 && abs([lindex [$d get 5] 0] - [lindex $co 45]) < 1e-12}]

set e [::mdance::data create 3 2]
check "a new dataset has no frames set" [expr {[dict get [$e info] nset] == 0}]
$e frame 0 {{1 2 3} {4 5 6}}
$e frame 1 {1 2 3 4 5 7}
check "frames from {x y z} triples or a flat list" [expr {[$e get 0] eq {1.0 2.0 3.0 4.0 5.0 6.0}}]
check "a dataset with a frame not set is refused" [string match "*never set*" [errof {::mdance::kmeans $e 1}]]
check "a frame of the wrong size is refused" [string match "*values for a frame of 6*" [errof {$e frame 2 {1 2 3}}]]
check "a frame index out of range is refused" [string match "*outside*" [errof {$e frame 3 {1 2 3 4 5 6}}]]
check "a coordinate that is not a number is refused" [string match "*not a number*" [errof {$e frame 2 {1 2 3 4 5 x}}]]
check "an infinite coordinate is refused" [string match "*not a finite number*" [errof {$e frame 2 {1 2 3 4 5 Inf}}]]
$e delete
check "delete removes the command" [expr {[info commands $e] eq ""}]

# superpose: a frame rotated and moved is put back on the reference
set ref {0 0 0 1 0 0 0 1 0 0 0 1}
set rot {}
foreach {x y z} $ref { lappend rot [expr {-$y + 5}] [expr {$x - 2}] [expr {$z + 1}] }
set s [::mdance::data create 2 4]
$s frame 0 $ref
$s frame 1 $rot
$s superpose -reference 0
set dev 0.0
foreach a [$s get 1] b [$s get 0] { set dev [expr {max($dev, abs($a - $b))}] }
check "superpose undoes a rotation and translation ($dev)" [expr {$dev < 1e-9}]
check "superpose refuses an atom out of range" [string match "*outside*" [errof {$s superpose -fit {0 9}}]]
$s delete

# ---------------------------------------------------------------- DCD
# a CHARMM DCD of 4 frames of 3 atoms, little-endian, with a unit cell
proc rec {data} { binary format i[string length $data] [string length $data] }
proc dcdfile {path nf na {cut 0}} {
    set icntrl [list $nf 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 24]
    set h [binary format a4i20 CORD $icntrl]
    set out [binary format i 84]$h[binary format i 84]
    set title [binary format ia80 1 "test"]
    append out [binary format i [string length $title]]$title[binary format i [string length $title]]
    append out [binary format iii 4 $na 4]
    for {set f 0} {$f < $nf} {incr f} {
        append out [binary format i 48][binary format d6 {10 0 10 0 0 10}][binary format i 48]
        foreach c {0 1 2} {
            set v {}
            for {set a 0} {$a < $na} {incr a} { lappend v [expr {100 * $f + 10 * $a + $c}] }
            append out [binary format i [expr {4 * $na}]][binary format f$na $v][binary format i [expr {4 * $na}]]
        }
    }
    if {$cut} { set out [string range $out 0 end-$cut] }
    set fh [open $path wb]
    puts -nonewline $fh $out
    close $fh
}
set tmp [file join [expr {[info exists ::env(TEMP)] ? $::env(TEMP) : "/tmp"}] mdance_test_[pid].dcd]
dcdfile $tmp 4 3
set dc [::mdance::data dcd $tmp]
check "dcd: frames and atoms" [expr {[dict get [$dc info] nframes] == 4 && [dict get [$dc info] natoms] == 3}]
check "dcd: x y z of each atom in order" [expr {[$dc get 2] eq {200.0 201.0 202.0 210.0 211.0 212.0 220.0 221.0 222.0}}]
$dc delete
set dc [::mdance::data dcd $tmp -atoms {2 0} -first 1 -stride 2]
check "dcd: atoms, first and stride" [expr {[dict get [$dc info] nframes] == 2 && [$dc get 1] eq {320.0 321.0 322.0 300.0 301.0 302.0}}]
$dc delete
dcdfile $tmp 4 3 5
check "dcd: a file cut short is refused" [string match "*cut short*" [errof {::mdance::data dcd $tmp}]]
set fh [open $tmp wb]; puts -nonewline $fh "not a dcd file at all"; close $fh
check "dcd: a file that is not one is refused" [string match "*not a DCD*" [errof {::mdance::data dcd $tmp}]]
file delete $tmp
check "dcd: a missing file is refused" [string match "*cannot be opened*" [errof {::mdance::data dcd $tmp}]]

# ---------------------------------------------------------------- options
check "an unknown option is refused, with the options" [string match "*unknown option \"-kinti\"*-kinit*" [errof {::mdance::kmeans $d 2 -kinti StratAll}]]
check "an option given twice is refused" [string match "*given twice*" [errof {::mdance::kmeans $d 2 -percentage 50 -percentage 10}]]
check "a value that is not a number is refused" [string match "*-percentage*not a whole number*" [errof {::mdance::kmeans $d 2 -percentage abc}]]
check "\"010\" is ten, not eight" [expr {[errof {::mdance::kmeans $d 2 -percentage 010 -kinit StratAll}] eq ""}]
check "a percentage over 100 is refused" [string match "*-percentage*outside*" [errof {::mdance::kmeans $d 2 -percentage 200}]]
check "k above the frames is refused" [string match "*-k*outside*" [errof {::mdance::kmeans $d 41}]]
check "a choice not offered is refused" [string match "*-kinit*not one of*" [errof {::mdance::kmeans $d 2 -kinit Bogus}]]
check "an option without its value is refused" [string match "*needs a value*" [errof {::mdance::kmeans $d 2 -kinit}]]
check "coordinates that are not frames of one length are refused" [string match "*not 40 frames*" [errof {::mdance::kmeans [lrange $co 0 end-1] 40 3 2}]]

# ---------------------------------------------------------------- the other calls
set h [::mdance::helm $d 2 [concat [lrepeat 20 0] [lrepeat 20 1]]]
check "helm keeps two starting clusters as they are" [expr {[dict get $h nClusters] == 2}]
set init {}
for {set f 0} {$f < $nf} {incr f} { lappend init [expr {$f / 5}] }
set h [::mdance::helm $d 2 $init -merge-scheme Inter]
check "helm merges 8 starting clusters into the two blobs" [expr {[dict get $h nClusters] == 2 &&
    [llength [lsort -unique [lrange [dict get $h labels] 0 19]]] == 1 && [dict exists $h zMatrix]}]
check "helm refuses starting labels of the wrong length" [string match "*values for 40 frames*" [errof {::mdance::helm $d 2 {0 1}}]]
set q [::mdance::equal $d -threshold 1.0]
check "equal finds the two blobs" [expr {[dict get $q nClusters] == 2}]
check "equal needs -threshold" [string match "*-threshold is required*" [errof {::mdance::equal $d}]]
set an [::mdance::analysis $d -labels $lab]
check "analysis per cluster" [expr {[dict get $an nClusters] == 2 && [llength [dict get $an clusterISIM]] == 2}]
check "analysis refuses a label past the frames" [string match "*outside*" [errof {::mdance::analysis $d -labels [lreplace $lab 0 0 4000]}]]
set p [::mdance::prime $d -labels $lab -weighted]
check "prime returns frames" [expr {[dict get $p medoidAll] >= 0 && [dict get $p medoidAll] < $nf}]
check "prime needs -labels" [string match "*-labels is required*" [errof {::mdance::prime $d}]]
set sel [::mdance::select $d -method diversity -param 10]
check "select diversity picks 10% of the frames, all different" [expr {[llength $sel] == 4 && [llength [lsort -unique $sel]] == 4}]
if {[dict get $inf divine]} {
    set dv [::mdance::divine $d 2]
    check "divine splits the two blobs" [expr {[dict get $dv nClusters] == 2}]
} else {
    check "divine says it is not in this build" [string match "*not available*" [errof {::mdance::divine {0 0 0 1 1 1} 2 1 1}]]
}

# ---------------------------------------------------------------- scans
set sc [::mdance::scan kmeans $d -ks {2 3 4 5} -kinit StratAll -percentage 50 -threads 3]
check "scan: one dict a value of k, in order" [expr {[llength $sc] == 4 && [lmap r $sc {dict get $r k}] eq {2 3 4 5}}]
set same 1
foreach r $sc {
    set one [::mdance::kmeans $d [dict get $r k] -kinit StratAll -percentage 50]
    if {[dict remove $r k] ne $one} { set same 0 }
}
check "  each the single call's answer" $same
set sc [::mdance::scan kmeans $d -ks {2 41 3} -kinit StratAll -percentage 50]
check "  a value that cannot run has its error, the others their answer" [expr {
    [dict exists [lindex $sc 0] labels] && [string match "*outside*" [dict get [lindex $sc 1] error]] && [dict exists [lindex $sc 2] labels]}]
check "  -ks is required" [string match "*-ks is required*" [errof {::mdance::scan kmeans $d -kinit StratAll}]]
check "  only an algorithm with k" [string match "*takes no k*" [errof {::mdance::scan equal $d -ks {2}}]]
check "  on a dataset only" [string match "*is no dataset*" [errof {::mdance::scan kmeans {1 2 3} -ks {2}}]]
check "  the algorithm's options, checked" [string match "*unknown option \"-split\"*" [errof {::mdance::scan kmeans $d -ks {2} -split MSD}]]
check "  HELM needs -labels" [string match "*-labels is required*" [errof {::mdance::scan helm $d -ks {2}}]]
set hs [::mdance::scan helm $d -ks {2 3} -labels $init]
check "  HELM's scan: each the single call's answer" [expr {
    [dict remove [lindex $hs 0] k] eq [::mdance::helm $d 2 $init] && [dict remove [lindex $hs 1] k] eq [::mdance::helm $d 3 $init]}]
set j [::mdance::job start scan kmeans $d -ks {2 3} -kinit StratAll -percentage 50]
::mdance::job wait $j
set js [::mdance::job result $j]
check "  a scan as a job" [expr {[lmap r $js {dict get $r k}] eq {2 3} && [dict exists [lindex $js 1] labels]}]
# a scan cancelled keeps what it made: its list, the rest "cancelled"
set many [::mdance::data create 2000 20]
for {set f 0} {$f < 2000} {incr f} {
    set v {}
    for {set q 0} {$q < 60} {incr q} { lappend v [expr {sin($f * 0.37 + $q) * 3 + ($f % 13)}] }
    $many frame $f $v
}
set ks {}
for {set k 2} {$k <= 80} {incr k} { lappend ks $k }
set j [::mdance::job start scan kmeans $many -ks $ks -kinit DivSelect -percentage 50 -threads 2]
after 400
::mdance::job cancel $j
::mdance::job wait $j
set st [dict get [::mdance::job status $j] state]
set cs [::mdance::job result $j]
set nmade [llength [lmap r $cs {expr {[dict exists $r labels] ? 1 : [continue]}}]]
set ncanc [llength [lmap r $cs {expr {[dict exists $r error] && [dict get $r error] eq "cancelled" ? 1 : [continue]}}]]
check "  a scan cancelled: state cancelled, its list kept ($nmade made, $ncanc cancelled, of [llength $cs])" [expr {
    $st eq "cancelled" && [llength $cs] == [llength $ks] && $ncanc > 0 && $nmade + $ncanc == [llength $ks]}]
$many delete

# ---------------------------------------------------------------- jobs
set j [::mdance::job start kmeans $d 2 -kinit StratAll -percentage 50]
check "job start returns a job" [expr {[lsearch [::mdance::job names] $j] >= 0}]
::mdance::job wait $j
check "job status says done" [expr {[dict get [::mdance::job status $j] state] eq "done"}]
set jr [::mdance::job result $j]
check "a job's result is the call's" [expr {[dict get $jr labels] eq $lab}]
check "a job's result is taken once" [string match "no job*" [errof {::mdance::job result $j}]]
check "an error of a job is its result's" [string match "*-k*outside*" [errof {
    set j2 [::mdance::job start kmeans $d 400]; ::mdance::job wait $j2; ::mdance::job result $j2}]]

# a long job cancelled (eQUAL with comp_sim seeds on 3000 frames)
set big [::mdance::data create 3000 10]
for {set f 0} {$f < 3000} {incr f} {
    set v {}
    for {set c 0} {$c < 30} {incr c} { lappend v [expr {sin($f * 0.37 + $c) * 3 + ($f % 11)}] }
    $big frame $f $v
}
set t0 [clock milliseconds]
set j [::mdance::job start equal $big -threshold 0.01 -seed-method comp_sim]
check "a dataset a running job reads cannot be changed" [string match "*being read by a job*" [errof {$big frame 0 [$big get 0]}]]
check "status of a running job" [expr {[dict get [::mdance::job status $j] state] eq "running"}]
after 300
::mdance::job cancel $j
::mdance::job wait $j
set took [expr {[clock milliseconds] - $t0}]
check "a cancelled job ends cancelled ($took ms)" [expr {[dict get [::mdance::job status $j] state] eq "cancelled"}]
check "a cancelled job's result is the error \"cancelled\"" [expr {[errof {::mdance::job result $j}] eq "cancelled"}]

if {[dict get $inf threaded]} {
    set ::ended ""
    set j [::mdance::job start -command {set ::ended} kmeans $d 2 -kinit StratAll -percentage 50]
    after 20000 {set ::ended timeout}
    vwait ::ended
    check "-command runs with the job's name when it ends" [expr {$::ended eq $j}]
    ::mdance::job result $j
}

# a dataset deleted while a job reads it is freed after the job
set j [::mdance::job start select $big -method diversity -param 50]
$big delete
::mdance::job wait $j
check "a dataset deleted during a job: the job still ends" [expr {[dict get [::mdance::job status $j] state] eq "done"}]
::mdance::job forget $j

puts "RESULT: $::pass passed, $::fail failed"
exit [expr {$::fail ? 1 : 0}]
