// The Tcl extension of libmdance (package mdance, namespace ::mdance), for
// VMD and any other Tcl 8.5+ interpreter.
//
// Datasets. ::mdance::data makes a dataset: coordinates held by the library,
// read once and clustered as often as wanted. A dataset is a command:
//
//   set d [::mdance::data create nframes natoms]
//   $d frame 0 [$sel get {x y z}]        ;# a frame: {x y z} triples or a flat list
//   set d [::mdance::data dcd file.dcd ?-atoms {i ...}? ?-first f? ?-last l? ?-stride s?]
//   set d [::mdance::data list coords nframes natoms]   ;# a flat list, all frames
//   $d superpose ?-fit {i ...}? ?-reference f?
//   $d info   $d get f   $d delete
//
// From VMD the fastest way in is the DCD: VMD writes the selection's frames
// itself (`animate write dcd $tmp beg $f end $l skip $s sel $sel waitfor all
// $mol`), and the library reads the file: no Tcl list is made of the
// coordinates at all.
//
// Calls. Every algorithm takes a dataset, or (the first interface, kept) a
// flat list of coordinates with its frame and atom counts:
//
//   ::mdance::kmeans  $d k ?options?           coords nframes natoms k ?options?
//   ::mdance::divine  $d k ?options?           coords nframes natoms k ?options?
//   ::mdance::helm    $d k labels ?options?    coords nframes natoms k labels ?options?
//   ::mdance::equal   $d ?options?             coords nframes natoms ?options?
//   ::mdance::analysis, ::mdance::prime, ::mdance::select: as equal
//   ::mdance::scan kmeans|divine|helm $d -ks {k ...} ?-threads n? ?options?
//       every k at once on threads (one a processor by default): a list
//       of one dict a k, the clustering's with its k, or k and its error
//
// Options are -name value, or -name alone for a flag; each is read from the
// options only (the coordinates are never turned into text), checked by the
// library, and an option the call does not take is an error.
//
// Jobs. Any call can run on a thread of its own while Tcl's event loop goes
// on (VMD keeps drawing and answering):
//
//   set j [::mdance::job start ?-command script? kmeans $d 10 -kinit StratAll]
//   ::mdance::job status $j     ;# {state running progress 0.4 stage eQUAL}
//   ::mdance::job cancel $j     ;# stops it within a fraction of a second
//   ::mdance::job wait $j       ;# blocks until it ends
//   ::mdance::job result $j     ;# the call's result (or its error); forgets the job
//
// -command runs the script with the job's name appended once it ends, from
// the event loop (in a threaded Tcl, as VMD's is). A dataset a job reads
// cannot be changed until the job ends; it can be deleted (it is freed once
// the job is done).

// USE_TCL_STUBS: the extension uses the host interpreter's Tcl, through its
// stubs table, and links only libtclstub: a second Tcl runtime in VMD's
// process aborts it ("alloc: invalid block").
#define USE_TCL_STUBS
#include <tcl.h>

#include <atomic>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "../capi/mdance_capi.h"

namespace {

// ---------------------------------------------------------------- helpers

int err(Tcl_Interp* interp, const std::string& msg) {
    Tcl_SetObjResult(interp, Tcl_NewStringObj(msg.c_str(), -1));
    return TCL_ERROR;
}

// The library names an option with underscores (end_mode), Tcl with dashes
// (-end-mode): a library message is said in Tcl's names.
std::string tclNames(std::string m) {
    std::string out;
    size_t i = 0;
    while (i < m.size()) {
        size_t p = m.find("option ", i);
        if (p == std::string::npos) { out += m.substr(i); break; }
        out += m.substr(i, p - i) + "option ";
        p += 7;
        // the names up to ':' or ' ' or '(' (a list of unknown ones: "a, b")
        size_t e = p;
        while (e < m.size() && m[e] != ':' && m[e] != ' ' && m[e] != '(') {
            if (m[e] == ',' && e + 1 < m.size() && m[e + 1] == ' ') { e += 2; continue; }
            ++e;
        }
        std::string names = m.substr(p, e - p), conv;
        bool start = true;
        for (char ch : names) {
            if (start && ch != ' ' && ch != ',') { conv += '-'; start = false; }
            if (ch == ',' || ch == ' ') start = true;
            conv += ch == '_' ? '-' : ch;
        }
        out += conv;
        i = e;
    }
    // "(the options: a b c)"
    size_t q = out.find("(the options: ");
    if (q != std::string::npos) {
        size_t e = out.find(')', q);
        std::string names = out.substr(q + 14, e == std::string::npos ? std::string::npos : e - q - 14), conv;
        bool start = true;
        for (char ch : names) {
            if (start && ch != ' ') { conv += '-'; start = false; }
            if (ch == ' ') start = true;
            conv += ch == '_' ? '-' : ch;
        }
        out = out.substr(0, q + 14) + conv + (e == std::string::npos ? "" : out.substr(e));
    }
    return out;
}

// A whole number of an object (an int, or text: never a list's text).
bool intOf(Tcl_Interp* interp, Tcl_Obj* o, const char* what, int& v) {
    if (Tcl_GetIntFromObj(nullptr, o, &v) != TCL_OK) {
        err(interp, std::string(what) + ": \"" + Tcl_GetString(o) + "\" is not a whole number");
        return false;
    }
    return true;
}

bool intList(Tcl_Interp* interp, Tcl_Obj* o, const char* what, std::vector<int>& out) {
    int n;
    Tcl_Obj** el;
    if (Tcl_ListObjGetElements(interp, o, &n, &el) != TCL_OK) return false;
    out.resize(n);
    for (int i = 0; i < n; ++i) {
        if (Tcl_GetIntFromObj(nullptr, el[i], &out[i]) != TCL_OK) {
            err(interp, std::string(what) + ": element " + std::to_string(i) + " (\"" + Tcl_GetString(el[i]) +
                            "\") is not a whole number");
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------- datasets

struct Dataset {
    mdance_dataset_t* d = nullptr;
    std::atomic<int> busy{0};       // jobs reading it
    ~Dataset() { mdance_dataset_free(d); }
};
using DatasetPtr = std::shared_ptr<Dataset>;

struct State;      // per interpreter

struct DataCmd {
    State* st;
    DatasetPtr ds;
    Tcl_Command token;
};

struct Job;

// Everything the extension keeps for one interpreter.
struct State {
    Tcl_Interp* interp;
    Tcl_ThreadId thread;
    bool threaded;
    int nextData = 1;
    int nextJob = 1;
    std::map<std::string, DataCmd*> data;          // command name -> dataset
    std::map<std::string, std::shared_ptr<Job>> jobs;
};

// The states alive (an event queued by a job's thread may arrive after its
// interpreter was deleted).
std::mutex statesLock;
std::set<State*> states;

// The dataset a call's first argument names, or null. A dataset's name is a
// short text; a list of coordinates is never read as text (an object with
// no text, or a long one, is no dataset's name).
DatasetPtr datasetOf(State* st, Tcl_Obj* o) {
    if (!o->bytes || o->length > 200) return nullptr;
    std::string name(o->bytes, o->length);
    auto it = st->data.find(name);
    if (it == st->data.end() && name.compare(0, 2, "::") != 0) it = st->data.find("::" + name);
    return it == st->data.end() ? nullptr : it->second->ds;
}

int dataInstanceCmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);

void dataDeleted(ClientData cd) {
    DataCmd* c = static_cast<DataCmd*>(cd);
    for (auto it = c->st->data.begin(); it != c->st->data.end(); ++it) {
        if (it->second == c) { c->st->data.erase(it); break; }
    }
    delete c;   // the dataset itself lives on while a job holds it
}

int newDataCommand(State* st, Tcl_Interp* interp, mdance_dataset_t* d) {
    auto ds = std::make_shared<Dataset>();
    ds->d = d;
    std::string name = "::mdance::data" + std::to_string(st->nextData++);
    DataCmd* c = new DataCmd{st, ds, nullptr};
    c->token = Tcl_CreateObjCommand(interp, name.c_str(), dataInstanceCmd, c, dataDeleted);
    st->data[name] = c;
    Tcl_SetObjResult(interp, Tcl_NewStringObj(name.c_str(), -1));
    return TCL_OK;
}

// A frame's values from a list: ncols numbers, or natoms {x y z} triples
// (what `$sel get {x y z}` returns).
bool frameValues(Tcl_Interp* interp, Tcl_Obj* o, int ncols, std::vector<double>& out) {
    int n;
    Tcl_Obj** el;
    if (Tcl_ListObjGetElements(interp, o, &n, &el) != TCL_OK) return false;
    out.clear();
    out.reserve(ncols);
    if (n > 0 && n * 3 == ncols && n != ncols) {
        int m;
        Tcl_Obj** xyz;
        for (int i = 0; i < n; ++i) {
            if (Tcl_ListObjGetElements(nullptr, el[i], &m, &xyz) != TCL_OK || m != 3) {
                err(interp, "atom " + std::to_string(i) + ": not an {x y z} triple");
                return false;
            }
            for (int c = 0; c < 3; ++c) {
                double v;
                if (Tcl_GetDoubleFromObj(nullptr, xyz[c], &v) != TCL_OK) {
                    err(interp, "atom " + std::to_string(i) + ": \"" + Tcl_GetString(xyz[c]) + "\" is not a number");
                    return false;
                }
                out.push_back(v);
            }
        }
        return true;
    }
    if (n != ncols) {
        err(interp, std::to_string(n) + " values for a frame of " + std::to_string(ncols) + " (" +
                        std::to_string(ncols / 3) + " {x y z} triples, or " + std::to_string(ncols) + " numbers)");
        return false;
    }
    for (int i = 0; i < n; ++i) {
        double v;
        if (Tcl_GetDoubleFromObj(nullptr, el[i], &v) != TCL_OK) {
            err(interp, "value " + std::to_string(i) + ": \"" + Tcl_GetString(el[i]) + "\" is not a number");
            return false;
        }
        out.push_back(v);
    }
    return true;
}

// A dataset from the first interface's flat list (frame after frame), read
// into the dataset a frame at a time: one copy, and never the list's text.
mdance_dataset_t* datasetFromList(Tcl_Interp* interp, Tcl_Obj* list, int nframes, int natoms) {
    int n;
    Tcl_Obj** el;
    if (Tcl_ListObjGetElements(interp, list, &n, &el) != TCL_OK) return nullptr;
    if (nframes <= 0) { err(interp, "nframes must be positive"); return nullptr; }
    if (natoms <= 0) { err(interp, "natoms must be positive"); return nullptr; }
    if (n == 0 || n % nframes != 0) {
        err(interp, std::to_string(n) + " coordinates are not " + std::to_string(nframes) + " frames of the same length");
        return nullptr;
    }
    int ncols = n / nframes;
    mdance_dataset_t* d = mdance_dataset_new_cols(nframes, ncols, natoms);
    if (!d) { err(interp, mdance_last_error()); return nullptr; }
    std::vector<double> row(ncols);
    for (int f = 0; f < nframes; ++f) {
        for (int j = 0; j < ncols; ++j) {
            Tcl_Obj* v = el[(size_t)f * ncols + j];
            if (Tcl_GetDoubleFromObj(nullptr, v, &row[j]) != TCL_OK) {
                err(interp, "coordinate " + std::to_string((size_t)f * ncols + j) + " (frame " + std::to_string(f) +
                                "): \"" + Tcl_GetString(v) + "\" is not a number");
                mdance_dataset_free(d);
                return nullptr;
            }
        }
        if (mdance_dataset_set_frame(d, f, row.data(), ncols) != 0) {
            err(interp, mdance_last_error());
            mdance_dataset_free(d);
            return nullptr;
        }
    }
    return d;
}

int dataInstanceCmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    DataCmd* c = static_cast<DataCmd*>(cd);
    mdance_dataset_t* d = c->ds->d;
    static const char* subs[] = {"frame", "get", "info", "superpose", "delete", nullptr};
    enum { FRAME, GET, INFO, SUPERPOSE, DELETE };
    int sub;
    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "frame|get|info|superpose|delete ?arg ...?");
        return TCL_ERROR;
    }
    if (Tcl_GetIndexFromObj(interp, objv[1], subs, "subcommand", 0, &sub) != TCL_OK) return TCL_ERROR;
    bool changes = sub == FRAME || sub == SUPERPOSE;
    if (changes && c->ds->busy.load())
        return err(interp, "the dataset is being read by a job: it cannot be changed until the job ends");
    switch (sub) {
    case FRAME: {
        if (objc != 4) { Tcl_WrongNumArgs(interp, 2, objv, "index values"); return TCL_ERROR; }
        int f;
        if (!intOf(interp, objv[2], "frame", f)) return TCL_ERROR;
        std::vector<double> v;
        if (!frameValues(interp, objv[3], mdance_dataset_ncols(d), v)) {
            Tcl_AppendResult(interp, " (frame ", Tcl_GetString(objv[2]), ")", (char*)nullptr);
            return TCL_ERROR;
        }
        if (mdance_dataset_set_frame(d, f, v.data(), (int)v.size()) != 0) return err(interp, mdance_last_error());
        return TCL_OK;
    }
    case GET: {
        if (objc != 3) { Tcl_WrongNumArgs(interp, 2, objv, "index"); return TCL_ERROR; }
        int f;
        if (!intOf(interp, objv[2], "frame", f)) return TCL_ERROR;
        std::vector<double> v(mdance_dataset_ncols(d));
        if (mdance_dataset_get_frame(d, f, v.data(), (int)v.size()) != 0) return err(interp, mdance_last_error());
        Tcl_Obj* l = Tcl_NewListObj(0, nullptr);
        for (double x : v) Tcl_ListObjAppendElement(nullptr, l, Tcl_NewDoubleObj(x));
        Tcl_SetObjResult(interp, l);
        return TCL_OK;
    }
    case INFO: {
        if (objc != 2) { Tcl_WrongNumArgs(interp, 2, objv, ""); return TCL_ERROR; }
        Tcl_Obj* r = Tcl_NewDictObj();
        Tcl_DictObjPut(nullptr, r, Tcl_NewStringObj("nframes", -1), Tcl_NewIntObj(mdance_dataset_nframes(d)));
        Tcl_DictObjPut(nullptr, r, Tcl_NewStringObj("natoms", -1), Tcl_NewIntObj(mdance_dataset_natoms(d)));
        Tcl_DictObjPut(nullptr, r, Tcl_NewStringObj("ncols", -1), Tcl_NewIntObj(mdance_dataset_ncols(d)));
        Tcl_DictObjPut(nullptr, r, Tcl_NewStringObj("nset", -1), Tcl_NewIntObj(mdance_dataset_nset(d)));
        Tcl_DictObjPut(nullptr, r, Tcl_NewStringObj("busy", -1), Tcl_NewIntObj(c->ds->busy.load()));
        Tcl_SetObjResult(interp, r);
        return TCL_OK;
    }
    case SUPERPOSE: {
        std::vector<int> fit;
        int ref = 0;
        bool seenFit = false, seenRef = false;
        for (int i = 2; i < objc; i += 2) {
            std::string o = Tcl_GetString(objv[i]);
            if (i + 1 >= objc) return err(interp, "superpose: " + o + " needs a value");
            if (o == "-fit" && !seenFit) {
                seenFit = true;
                if (!intList(interp, objv[i + 1], "superpose -fit", fit)) return TCL_ERROR;
            } else if (o == "-reference" && !seenRef) {
                seenRef = true;
                if (!intOf(interp, objv[i + 1], "superpose -reference", ref)) return TCL_ERROR;
            } else {
                return err(interp, "superpose: " + o + (seenFit || seenRef ? " (given twice, or" : " (") +
                                       " not an option: -fit {atoms}, -reference frame)");
            }
        }
        if (mdance_dataset_superpose(d, fit.empty() ? nullptr : fit.data(), (int)fit.size(), ref) != 0)
            return err(interp, mdance_last_error());
        return TCL_OK;
    }
    case DELETE:
        if (objc != 2) { Tcl_WrongNumArgs(interp, 2, objv, ""); return TCL_ERROR; }
        Tcl_DeleteCommandFromToken(interp, c->token);
        return TCL_OK;
    }
    return TCL_OK;
}

// ::mdance::data create nframes natoms ?ncols?
// ::mdance::data list coords nframes natoms
// ::mdance::data dcd file ?-atoms {i ...}? ?-first f? ?-last l? ?-stride s?
// ::mdance::data names
int dataCmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    State* st = static_cast<State*>(cd);
    static const char* subs[] = {"create", "list", "dcd", "names", nullptr};
    enum { CREATE, LIST, DCD, NAMES };
    int sub;
    if (objc < 2) { Tcl_WrongNumArgs(interp, 1, objv, "create|list|dcd|names ?arg ...?"); return TCL_ERROR; }
    if (Tcl_GetIndexFromObj(interp, objv[1], subs, "subcommand", 0, &sub) != TCL_OK) return TCL_ERROR;
    switch (sub) {
    case CREATE: {
        if (objc != 4 && objc != 5) { Tcl_WrongNumArgs(interp, 2, objv, "nframes natoms ?ncols?"); return TCL_ERROR; }
        int nf, na, nc = -1;
        if (!intOf(interp, objv[2], "nframes", nf) || !intOf(interp, objv[3], "natoms", na)) return TCL_ERROR;
        if (objc == 5 && !intOf(interp, objv[4], "ncols", nc)) return TCL_ERROR;
        mdance_dataset_t* d = nc < 0 ? mdance_dataset_new(nf, na) : mdance_dataset_new_cols(nf, nc, na);
        if (!d) return err(interp, mdance_last_error());
        return newDataCommand(st, interp, d);
    }
    case LIST: {
        if (objc != 5) { Tcl_WrongNumArgs(interp, 2, objv, "coords nframes natoms"); return TCL_ERROR; }
        int nf, na;
        if (!intOf(interp, objv[3], "nframes", nf) || !intOf(interp, objv[4], "natoms", na)) return TCL_ERROR;
        mdance_dataset_t* d = datasetFromList(interp, objv[2], nf, na);
        if (!d) return TCL_ERROR;
        return newDataCommand(st, interp, d);
    }
    case DCD: {
        if (objc < 3) { Tcl_WrongNumArgs(interp, 2, objv, "file ?-atoms {i ...}? ?-first f? ?-last l? ?-stride s?"); return TCL_ERROR; }
        std::vector<int> atoms;
        int first = 0, last = -1, stride = 1;
        std::set<std::string> seen;
        for (int i = 3; i < objc; i += 2) {
            std::string o = Tcl_GetString(objv[i]);
            if (i + 1 >= objc) return err(interp, "dcd: " + o + " needs a value");
            if (!seen.insert(o).second) return err(interp, "dcd: " + o + " given twice");
            if (o == "-atoms") { if (!intList(interp, objv[i + 1], "dcd -atoms", atoms)) return TCL_ERROR; }
            else if (o == "-first") { if (!intOf(interp, objv[i + 1], "dcd -first", first)) return TCL_ERROR; }
            else if (o == "-last") { if (!intOf(interp, objv[i + 1], "dcd -last", last)) return TCL_ERROR; }
            else if (o == "-stride") { if (!intOf(interp, objv[i + 1], "dcd -stride", stride)) return TCL_ERROR; }
            else return err(interp, "dcd: unknown option " + o + " (-atoms -first -last -stride)");
        }
        // (the name as Tcl normalizes it: a relative one is of Tcl's folder)
        Tcl_Obj* norm = Tcl_FSGetNormalizedPath(interp, objv[2]);
        const char* path = norm ? Tcl_GetString(norm) : Tcl_GetString(objv[2]);
        mdance_dataset_t* d = mdance_dataset_read_dcd(path, atoms.empty() ? nullptr : atoms.data(),
                                                      (int)atoms.size(), first, last, stride);
        if (!d) return err(interp, mdance_last_error());
        return newDataCommand(st, interp, d);
    }
    case NAMES: {
        Tcl_Obj* l = Tcl_NewListObj(0, nullptr);
        for (auto& kv : st->data) Tcl_ListObjAppendElement(nullptr, l, Tcl_NewStringObj(kv.first.c_str(), -1));
        Tcl_SetObjResult(interp, l);
        return TCL_OK;
    }
    }
    return TCL_OK;
}

// ---------------------------------------------------------------- calls

enum class Kind { Cluster, Analysis, Prime, Select, Scan };

// How a call's Tcl options map to the library's.
struct OptSpec {
    const char* tcl;
    const char* lib;
    enum { Value, Flag, List } type;
};

struct CallSpec {
    const char* name;
    Kind kind;
    bool takesK;
    bool takesLabels;      // HELM's positional starting labels
    std::vector<OptSpec> opts;
};

const std::vector<CallSpec>& specs() {
    static const std::vector<CallSpec> s = {
        {"kmeans", Kind::Cluster, true, false,
         {{"-metric", "metric", OptSpec::Value}, {"-kinit", "kinit", OptSpec::Value},
          {"-percentage", "percentage", OptSpec::Value}, {"-seed", "seed", OptSpec::Value}}},
        {"divine", Kind::Cluster, true, false,
         {{"-metric", "metric", OptSpec::Value}, {"-split", "split", OptSpec::Value},
          {"-anchors", "anchors", OptSpec::Value}, {"-kinit", "kinit", OptSpec::Value},
          {"-refine", "refine", OptSpec::Flag}, {"-threshold", "threshold", OptSpec::Value},
          {"-end-mode", "end_mode", OptSpec::Value}, {"-percentage", "percentage", OptSpec::Value}}},
        {"helm", Kind::Cluster, true, true,
         {{"-metric", "metric", OptSpec::Value}, {"-merge-scheme", "merge_scheme", OptSpec::Value},
          {"-eps", "eps", OptSpec::Value}, {"-trim-start", "trim_start", OptSpec::Flag},
          {"-min-samples", "min_samples", OptSpec::Value}, {"-trim-val", "trim_val", OptSpec::Value},
          {"-trim-k", "trim_k", OptSpec::Value}}},
        {"equal", Kind::Cluster, false, false,
         {{"-metric", "metric", OptSpec::Value}, {"-threshold", "threshold", OptSpec::Value},
          {"-seed-method", "seed_method", OptSpec::Value}, {"-n-seeds", "n_seeds", OptSpec::Value},
          {"-percentage", "percentage", OptSpec::Value}, {"-min-samples", "min_samples", OptSpec::Value},
          {"-check-sim", "check_sim", OptSpec::Flag}, {"-sim-threshold", "sim_threshold", OptSpec::Value},
          {"-reject-lowd", "reject_lowd", OptSpec::Flag}, {"-align", "align", OptSpec::Value}}},
        {"analysis", Kind::Analysis, false, false,
         {{"-metric", "metric", OptSpec::Value}, {"-labels", "labels", OptSpec::List}}},
        {"prime", Kind::Prime, false, false,
         {{"-metric", "metric", OptSpec::Value}, {"-labels", "labels", OptSpec::List},
          {"-trim-frac", "trim_frac", OptSpec::Value}, {"-weighted", "weighted", OptSpec::Flag}}},
        {"select", Kind::Select, false, false,
         {{"-metric", "metric", OptSpec::Value}, {"-method", "method", OptSpec::Value},
          {"-param", "param", OptSpec::Value}, {"-nbins", "nbins", OptSpec::Value}}},
        // (its options are the scanned algorithm's, and -ks, -threads, -labels:
        // parseScan)
        {"scan", Kind::Scan, false, false, {}},
    };
    return s;
}

const CallSpec* specOf(const std::string& name) {
    for (auto& s : specs()) if (name == s.name) return &s;
    return nullptr;
}

// A call parsed and ready to run on any thread: it holds what it reads.
struct Call {
    const CallSpec* spec = nullptr;
    DatasetPtr ds;
    mdance_options_t* opts = nullptr;
    Call() = default;
    Call(const Call&) = delete;
    Call& operator=(const Call&) = delete;
    ~Call() { mdance_options_free(opts); }

    // the library's answer (one of these, the others null)
    mdance_result_t* res = nullptr;
    mdance_analysis_t* ana = nullptr;
    mdance_prime_t* pri = nullptr;
    mdance_select_t* sel = nullptr;
    mdance_scan_t* scn = nullptr;
    std::string algo;           // a scan's algorithm
    std::vector<int> ks;        // and its values of k
    void clear() {
        mdance_result_free(res); res = nullptr;
        mdance_analysis_free(ana); ana = nullptr;
        mdance_prime_free(pri); pri = nullptr;
        mdance_select_free(sel); sel = nullptr;
        mdance_scan_free(scn); scn = nullptr;
    }

    void run(mdance_control_t* ctl) {
        const mdance_dataset_t* d = ds->d;
        switch (spec->kind) {
        case Kind::Cluster: res = mdance_cluster(d, spec->name, opts, ctl); break;
        case Kind::Analysis: ana = mdance_analyze(d, opts, ctl); break;
        case Kind::Prime: pri = mdance_prime_run(d, opts, ctl); break;
        case Kind::Select: sel = mdance_select_run(d, opts, ctl); break;
        case Kind::Scan: scn = mdance_scan_run(d, algo.c_str(), ks.data(), (int)ks.size(), opts, ctl); break;
        }
    }
    const char* error() const {
        switch (spec->kind) {
        case Kind::Cluster: return mdance_result_error(res);
        case Kind::Analysis: return mdance_analysis_error(ana);
        case Kind::Prime: return mdance_prime_error(pri);
        case Kind::Select: return mdance_select_error(sel);
        case Kind::Scan: return mdance_scan_error(scn);
        }
        return "?";
    }
};

Tcl_Obj* intsObj(const int* a, int n) {
    Tcl_Obj* l = Tcl_NewListObj(0, nullptr);
    for (int i = 0; a && i < n; ++i) Tcl_ListObjAppendElement(nullptr, l, Tcl_NewIntObj(a[i]));
    return l;
}
Tcl_Obj* doublesObj(const double* a, int n) {
    Tcl_Obj* l = Tcl_NewListObj(0, nullptr);
    for (int i = 0; a && i < n; ++i) Tcl_ListObjAppendElement(nullptr, l, Tcl_NewDoubleObj(a[i]));
    return l;
}
void put(Tcl_Obj* d, const char* k, Tcl_Obj* v) { Tcl_DictObjPut(nullptr, d, Tcl_NewStringObj(k, -1), v); }

// A clustering's dict (the keys of the extension's first version).
Tcl_Obj* clusteringObj(const char* algorithm, const mdance_result_t* r) {
        int nf = mdance_result_nframes(r), k = mdance_result_nclusters(r);
        Tcl_Obj* d = Tcl_NewDictObj();
        put(d, "algorithm", Tcl_NewStringObj(algorithm, -1));
        put(d, "nFrames", Tcl_NewIntObj(nf));
        put(d, "nClusters", Tcl_NewIntObj(k));
        if (mdance_result_labels(r)) put(d, "labels", intsObj(mdance_result_labels(r), nf));
        if (mdance_result_cluster_sizes(r)) put(d, "clusterSizes", intsObj(mdance_result_cluster_sizes(r), k));
        if (mdance_result_representatives(r)) put(d, "representatives", intsObj(mdance_result_representatives(r), k));
        if (mdance_result_cluster_msd(r)) put(d, "clusterMSD", doublesObj(mdance_result_cluster_msd(r), k));
        put(d, "score_calinskiHarabasz", Tcl_NewDoubleObj(mdance_result_ch_score(r)));
        put(d, "score_daviesBouldin", Tcl_NewDoubleObj(mdance_result_db_score(r)));
        if (mdance_result_seed(r) >= 0) put(d, "seed", Tcl_NewIntObj(mdance_result_seed(r)));
        int zr = mdance_result_zmatrix_rows(r), zc = mdance_result_zmatrix_cols(r);
        const double* z = mdance_result_zmatrix(r);
        if (z && zr > 0) {
            Tcl_Obj* zm = Tcl_NewListObj(0, nullptr);
            for (int i = 0; i < zr; ++i) Tcl_ListObjAppendElement(nullptr, zm, doublesObj(z + (size_t)i * zc, zc));
            put(d, "zMatrix", zm);
        }
        return d;
}

// The call's answer as Tcl.
int answer(Tcl_Interp* interp, Call& c) {
    const char* e = c.error();
    // (a scan cancelled keeps what it made: its list, the values not made
    // marked "cancelled"; the job's state says cancelled)
    const bool scanCancelled = c.spec->kind == Kind::Scan && e && std::strcmp(e, "cancelled") == 0 && c.scn;
    if (e && !scanCancelled) return err(interp, tclNames(e));
    switch (c.spec->kind) {
    case Kind::Cluster:
        Tcl_SetObjResult(interp, clusteringObj(c.spec->name, c.res));
        return TCL_OK;
    case Kind::Scan: {
        // one dict a value of k, in their order: the clustering's, with k;
        // or k and the error of a value that failed
        Tcl_Obj* l = Tcl_NewListObj(0, nullptr);
        for (int i = 0; i < mdance_scan_count(c.scn); ++i) {
            const mdance_result_t* r = mdance_scan_result(c.scn, i);
            const char* re = mdance_result_error(r);
            Tcl_Obj* d;
            if (re) {
                d = Tcl_NewDictObj();
                put(d, "error", Tcl_NewStringObj(tclNames(re).c_str(), -1));
            } else {
                d = clusteringObj(c.algo.c_str(), r);
            }
            put(d, "k", Tcl_NewIntObj(mdance_scan_k(c.scn, i)));
            Tcl_ListObjAppendElement(nullptr, l, d);
        }
        Tcl_SetObjResult(interp, l);
        return TCL_OK;
    }
    case Kind::Analysis: {
        mdance_analysis_t* a = c.ana;
        Tcl_Obj* d = Tcl_NewDictObj();
        int nc = mdance_analysis_nclusters(a);
        put(d, "isim", Tcl_NewDoubleObj(mdance_analysis_isim(a)));
        put(d, "nClusters", Tcl_NewIntObj(nc));
        if (mdance_analysis_cluster_isim(a)) put(d, "clusterISIM", doublesObj(mdance_analysis_cluster_isim(a), nc));
        if (mdance_analysis_cluster_outliers(a)) put(d, "clusterOutliers", intsObj(mdance_analysis_cluster_outliers(a), nc));
        Tcl_SetObjResult(interp, d);
        return TCL_OK;
    }
    case Kind::Prime: {
        mdance_prime_t* p = c.pri;
        Tcl_Obj* d = Tcl_NewDictObj();
        put(d, "pairwise", Tcl_NewIntObj(mdance_prime_pairwise(p)));
        put(d, "union", Tcl_NewIntObj(mdance_prime_union(p)));
        put(d, "medoid", Tcl_NewIntObj(mdance_prime_medoid(p)));
        put(d, "outlier", Tcl_NewIntObj(mdance_prime_outlier(p)));
        put(d, "medoidAll", Tcl_NewIntObj(mdance_prime_medoid_all(p)));
        put(d, "medoidC0", Tcl_NewIntObj(mdance_prime_medoid_c0(p)));
        put(d, "medoidC0Trimmed", Tcl_NewIntObj(mdance_prime_medoid_c0_trimmed(p)));
        put(d, "nClusters", Tcl_NewIntObj(mdance_prime_nclusters(p)));
        Tcl_SetObjResult(interp, d);
        return TCL_OK;
    }
    case Kind::Select:
        Tcl_SetObjResult(interp, intsObj(mdance_select_indices(c.sel), mdance_select_count(c.sel)));
        return TCL_OK;
    }
    return TCL_OK;
}

int setOpt(Tcl_Interp* interp, mdance_options_t* o, const char* name, const std::string& v) {
    if (mdance_options_set(o, name, v.c_str()) != 0) return err(interp, mdance_last_error());
    return TCL_OK;
}

// Parse a call: objv[0] is the call's name. Its positional arguments are
// read by position and its options from where they start: the coordinates'
// list is read as a list only.
// ::mdance::scan algorithm dataset -ks {k ...} ?-threads n? ?-labels {...}?
// ?the algorithm's options?
int parseScan(State* st, Tcl_Interp* interp, const CallSpec& spec, int objc, Tcl_Obj* const objv[], Call& call) {
    call.spec = &spec;
    call.opts = mdance_options_new();
    if (!call.opts) return err(interp, mdance_last_error());
    if (objc < 3) {
        Tcl_WrongNumArgs(interp, 1, objv, "kmeans|divine|helm dataset -ks {k ...} ?-threads n? ?options?");
        return TCL_ERROR;
    }
    call.algo = Tcl_GetString(objv[1]);
    const CallSpec* as = specOf(call.algo);
    if (!as || !as->takesK)
        return err(interp, "scan: \"" + call.algo + "\" takes no k (kmeans divine helm)");
    if (!(call.ds = datasetOf(st, objv[2])))
        return err(interp, std::string("scan: \"") + Tcl_GetString(objv[2]) + "\" is no dataset (::mdance::data)");
    std::vector<OptSpec> opts = as->opts;
    opts.push_back({"-ks", "ks", OptSpec::List});
    opts.push_back({"-threads", "threads", OptSpec::Value});
    if (as->takesLabels) opts.push_back({"-labels", "labels", OptSpec::List});
    std::set<std::string> seen;
    for (int i = 3; i < objc; ++i) {
        std::string name = Tcl_GetString(objv[i]);
        const OptSpec* o = nullptr;
        for (auto& x : opts) if (name == x.tcl) o = &x;
        if (!o) {
            std::string all;
            for (auto& x : opts) all += std::string(all.empty() ? "" : " ") + x.tcl;
            return err(interp, "scan " + call.algo + ": unknown option \"" + name + "\" (the options: " + all + ")");
        }
        if (!seen.insert(name).second) return err(interp, "scan: " + name + " given twice");
        if (o->type == OptSpec::Flag) {
            if (setOpt(interp, call.opts, o->lib, "1") != TCL_OK) return TCL_ERROR;
            continue;
        }
        if (i + 1 >= objc) return err(interp, "scan: " + name + " needs a value");
        Tcl_Obj* v = objv[++i];
        if (name == "-ks") {
            if (!intList(interp, v, "scan -ks", call.ks)) return TCL_ERROR;
        } else if (o->type == OptSpec::List) {
            std::vector<int> l;
            if (!intList(interp, v, ("scan " + name).c_str(), l)) return TCL_ERROR;
            if (mdance_options_set_ints(call.opts, o->lib, l.data(), (int)l.size()) != 0) return err(interp, mdance_last_error());
        } else {
            if (setOpt(interp, call.opts, o->lib, Tcl_GetString(v)) != TCL_OK) return TCL_ERROR;
        }
    }
    if (call.ks.empty()) return err(interp, "scan: -ks is required (the values of k)");
    if (as->takesLabels && !seen.count("-labels")) return err(interp, "scan helm: -labels is required (the starting clusters)");
    return TCL_OK;
}

int parseCall(State* st, Tcl_Interp* interp, const CallSpec& spec, int objc, Tcl_Obj* const objv[], Call& call) {
    if (spec.kind == Kind::Scan) return parseScan(st, interp, spec, objc, objv, call);
    call.spec = &spec;
    call.opts = mdance_options_new();
    if (!call.opts) return err(interp, mdance_last_error());

    std::string usage;
    if (spec.takesK) usage = spec.takesLabels ? " k labels" : " k";
    int pos;     // the first argument after the data
    if (objc >= 2 && (call.ds = datasetOf(st, objv[1]))) {
        pos = 2;
    } else {
        // the first interface: coords nframes natoms
        if (objc < 4) {
            Tcl_WrongNumArgs(interp, 1, objv, ("dataset" + usage + " ?options?  or  coords nframes natoms" + usage + " ?options?").c_str());
            return TCL_ERROR;
        }
        int nf, na;
        if (!intOf(interp, objv[2], "nframes", nf) || !intOf(interp, objv[3], "natoms", na)) return TCL_ERROR;
        mdance_dataset_t* d = datasetFromList(interp, objv[1], nf, na);
        if (!d) return TCL_ERROR;
        call.ds = std::make_shared<Dataset>();
        call.ds->d = d;
        pos = 4;
    }
    if (spec.takesK) {
        if (pos >= objc) return err(interp, std::string(spec.name) + ": k is missing (" + spec.name + " data" + usage + " ?options?)");
        int k;
        if (!intOf(interp, objv[pos], "k", k)) return TCL_ERROR;
        if (setOpt(interp, call.opts, "k", std::to_string(k)) != TCL_OK) return TCL_ERROR;
        ++pos;
    }
    if (spec.takesLabels) {
        if (pos >= objc) return err(interp, std::string(spec.name) + ": the starting labels are missing");
        std::vector<int> l;
        if (!intList(interp, objv[pos], "labels", l)) return TCL_ERROR;
        if (mdance_options_set_ints(call.opts, "labels", l.data(), (int)l.size()) != 0) return err(interp, mdance_last_error());
        ++pos;
    }
    std::set<std::string> seen;
    for (int i = pos; i < objc; ++i) {
        std::string name = Tcl_GetString(objv[i]);
        const OptSpec* o = nullptr;
        for (auto& s : spec.opts) if (name == s.tcl) o = &s;
        if (!o) {
            std::string all;
            for (auto& s : spec.opts) all += std::string(all.empty() ? "" : " ") + s.tcl;
            return err(interp, std::string(spec.name) + ": unknown option \"" + name + "\" (the options: " + all + ")");
        }
        if (!seen.insert(name).second) return err(interp, std::string(spec.name) + ": " + name + " given twice");
        if (o->type == OptSpec::Flag) {
            if (setOpt(interp, call.opts, o->lib, "1") != TCL_OK) return TCL_ERROR;
            continue;
        }
        if (i + 1 >= objc) return err(interp, std::string(spec.name) + ": " + name + " needs a value");
        Tcl_Obj* v = objv[++i];
        if (o->type == OptSpec::List) {
            std::vector<int> l;
            if (!intList(interp, v, (std::string(spec.name) + " " + name).c_str(), l)) return TCL_ERROR;
            if (mdance_options_set_ints(call.opts, o->lib, l.data(), (int)l.size()) != 0) return err(interp, mdance_last_error());
        } else {
            if (setOpt(interp, call.opts, o->lib, Tcl_GetString(v)) != TCL_OK) return TCL_ERROR;
        }
    }
    return TCL_OK;
}

// A call's command knows which call it is (not by its name: a command
// renamed, as a caller's wrapper does, `rename ::mdance::kmeans k0`, is the
// same call).
struct CallCmd {
    State* st;
    const CallSpec* spec;
};

void callCmdDeleted(ClientData cd) { delete static_cast<CallCmd*>(cd); }

int callCmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    CallCmd* cc = static_cast<CallCmd*>(cd);
    State* st = cc->st;
    const CallSpec* spec = cc->spec;
    Call call;
    if (parseCall(st, interp, *spec, objc, objv, call) != TCL_OK) return TCL_ERROR;
    call.run(nullptr);
    int rc = answer(interp, call);
    call.clear();
    return rc;
}

// ---------------------------------------------------------------- jobs

struct Job {
    std::string name;
    Call call;
    mdance_control_t* ctl = nullptr;
    std::thread thread;
    std::atomic<bool> done{false};
    bool collected = false;          // the thread joined
    std::string command;             // -command script
    ~Job() {
        if (ctl) mdance_control_cancel(ctl);
        if (thread.joinable()) thread.join();
        if (call.ds) call.ds->busy--;
        call.clear();
        mdance_control_free(ctl);
    }
    void join() {
        if (!collected && thread.joinable()) thread.join();
        collected = true;
    }
};

// The event a job's thread queues for the interpreter's thread when it ends.
struct JobEvent {
    Tcl_Event header;
    State* st;
    char name[64];
};

int jobEventProc(Tcl_Event* ev, int) {
    JobEvent* je = reinterpret_cast<JobEvent*>(ev);
    State* st = je->st;
    {
        std::lock_guard<std::mutex> g(statesLock);
        if (!states.count(st)) return 1;       // the interpreter is gone
    }
    auto it = st->jobs.find(je->name);
    if (it == st->jobs.end() || it->second->command.empty()) return 1;
    Tcl_Interp* interp = st->interp;
    Tcl_Preserve(interp);
    Tcl_Obj* script = Tcl_NewStringObj(it->second->command.c_str(), -1);
    Tcl_IncrRefCount(script);
    if (Tcl_ListObjAppendElement(interp, script, Tcl_NewStringObj(je->name, -1)) == TCL_OK) {
        if (Tcl_EvalObjEx(interp, script, TCL_EVAL_GLOBAL) != TCL_OK) Tcl_BackgroundError(interp);
    } else {
        Tcl_BackgroundError(interp);
    }
    Tcl_DecrRefCount(script);
    Tcl_Release(interp);
    return 1;
}

std::shared_ptr<Job> jobOf(State* st, Tcl_Interp* interp, Tcl_Obj* o) {
    std::string n = Tcl_GetString(o);
    auto it = st->jobs.find(n);
    if (it == st->jobs.end()) {
        err(interp, "no job \"" + n + "\"");
        return nullptr;
    }
    return it->second;
}

const char* stateOf(Job& j) {
    if (!j.done.load()) return "running";
    const char* e = j.call.error();
    if (!e) return "done";
    return std::strcmp(e, "cancelled") == 0 ? "cancelled" : "error";
}

// ::mdance::job start ?-command script? call args...
// ::mdance::job status|wait|cancel|result|forget job
// ::mdance::job names
int jobCmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    State* st = static_cast<State*>(cd);
    static const char* subs[] = {"start", "status", "wait", "cancel", "result", "forget", "names", nullptr};
    enum { START, STATUS, WAIT, CANCEL, RESULT, FORGET, NAMES };
    int sub;
    if (objc < 2) { Tcl_WrongNumArgs(interp, 1, objv, "start|status|wait|cancel|result|forget|names ?arg ...?"); return TCL_ERROR; }
    if (Tcl_GetIndexFromObj(interp, objv[1], subs, "subcommand", 0, &sub) != TCL_OK) return TCL_ERROR;

    if (sub == START) {
        int i = 2;
        std::string command;
        if (i + 1 < objc && std::strcmp(Tcl_GetString(objv[i]), "-command") == 0) {
            command = Tcl_GetString(objv[i + 1]);
            i += 2;
            if (!command.empty() && !st->threaded)
                return err(interp, "job start -command needs a threaded Tcl (this one is not): poll the job's status instead");
        }
        if (i >= objc) { Tcl_WrongNumArgs(interp, 2, objv, "?-command script? call ?arg ...?"); return TCL_ERROR; }
        const CallSpec* spec = specOf(Tcl_GetString(objv[i]));
        if (!spec) return err(interp, std::string("job start: no call \"") + Tcl_GetString(objv[i]) +
                                          "\" (kmeans divine helm equal analysis prime select scan)");
        auto job = std::make_shared<Job>();
        if (parseCall(st, interp, *spec, objc - i, objv + i, job->call) != TCL_OK) return TCL_ERROR;
        job->ctl = mdance_control_new();
        if (!job->ctl) return err(interp, mdance_last_error());
        job->command = command;
        job->name = "::mdance::job" + std::to_string(st->nextJob++);
        job->call.ds->busy++;
        st->jobs[job->name] = job;
        Job* jp = job.get();
        Tcl_ThreadId target = st->thread;
        bool notify = !command.empty();
        try {
            job->thread = std::thread([jp, st, target, notify]() {
                jp->call.run(jp->ctl);
                jp->done.store(true);
                if (notify) {
                    JobEvent* ev = reinterpret_cast<JobEvent*>(Tcl_Alloc(sizeof(JobEvent)));
                    ev->header.proc = jobEventProc;
                    ev->header.nextPtr = nullptr;
                    ev->st = st;
                    std::strncpy(ev->name, jp->name.c_str(), sizeof ev->name - 1);
                    ev->name[sizeof ev->name - 1] = 0;
                    Tcl_ThreadQueueEvent(target, &ev->header, TCL_QUEUE_TAIL);
                    Tcl_ThreadAlert(target);
                }
            });
        } catch (const std::exception& e) {
            st->jobs.erase(job->name);
            return err(interp, std::string("job start: no thread could be started: ") + e.what());
        }
        Tcl_SetObjResult(interp, Tcl_NewStringObj(job->name.c_str(), -1));
        return TCL_OK;
    }
    if (sub == NAMES) {
        Tcl_Obj* l = Tcl_NewListObj(0, nullptr);
        for (auto& kv : st->jobs) Tcl_ListObjAppendElement(nullptr, l, Tcl_NewStringObj(kv.first.c_str(), -1));
        Tcl_SetObjResult(interp, l);
        return TCL_OK;
    }
    if (objc != 3) { Tcl_WrongNumArgs(interp, 2, objv, "job"); return TCL_ERROR; }
    std::shared_ptr<Job> j = jobOf(st, interp, objv[2]);
    if (!j) return TCL_ERROR;
    switch (sub) {
    case STATUS: {
        Tcl_Obj* d = Tcl_NewDictObj();
        put(d, "state", Tcl_NewStringObj(stateOf(*j), -1));
        put(d, "progress", Tcl_NewDoubleObj(j->done.load() ? 1.0 : mdance_control_progress(j->ctl)));
        put(d, "stage", Tcl_NewStringObj(mdance_control_stage(j->ctl), -1));
        put(d, "call", Tcl_NewStringObj(j->call.spec->name, -1));
        Tcl_SetObjResult(interp, d);
        return TCL_OK;
    }
    case WAIT:
        j->join();
        Tcl_SetObjResult(interp, Tcl_NewStringObj(stateOf(*j), -1));
        return TCL_OK;
    case CANCEL:
        mdance_control_cancel(j->ctl);
        return TCL_OK;
    case RESULT: {
        if (!j->done.load()) return err(interp, "job " + j->name + " is still running (job wait, or job status)");
        j->join();
        int rc = answer(interp, j->call);
        st->jobs.erase(j->name);
        return rc;
    }
    case FORGET:
        // cancel it and let it end on its own thread (joined when freed)
        mdance_control_cancel(j->ctl);
        st->jobs.erase(j->name);
        if (!j->done.load()) {
            // the last reference is this one: wait the moment it takes
            j->join();
        }
        return TCL_OK;
    }
    return TCL_OK;
}

// ---------------------------------------------------------------- the rest

int versionCmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    if (objc != 1) { Tcl_WrongNumArgs(interp, 1, objv, ""); return TCL_ERROR; }
    Tcl_SetObjResult(interp, Tcl_NewStringObj(mdance_version(), -1));
    return TCL_OK;
}

int infoCmd(ClientData cd, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    State* st = static_cast<State*>(cd);
    if (objc != 1) { Tcl_WrongNumArgs(interp, 1, objv, ""); return TCL_ERROR; }
    Tcl_Obj* d = Tcl_NewDictObj();
    put(d, "version", Tcl_NewStringObj(mdance_version(), -1));
    put(d, "api", Tcl_NewIntObj(mdance_api_version()));
    put(d, "divine", Tcl_NewIntObj(mdance_has_divine()));
    put(d, "threaded", Tcl_NewIntObj(st->threaded ? 1 : 0));
    Tcl_SetObjResult(interp, d);
    return TCL_OK;
}

void stateDeleted(ClientData cd, Tcl_Interp*) {
    State* st = static_cast<State*>(cd);
    {
        std::lock_guard<std::mutex> g(statesLock);
        states.erase(st);
    }
    // every job cancelled and its thread joined (Job's destructor)
    for (auto& kv : st->jobs) if (kv.second->ctl) mdance_control_cancel(kv.second->ctl);
    st->jobs.clear();
    delete st;
}

}  // namespace

extern "C" {

#if defined(_WIN32)
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
int Mdance_Init(Tcl_Interp* interp) {
    if (Tcl_InitStubs(interp, "8.5", 0) == nullptr) return TCL_ERROR;
    if (mdance_api_version() < MDANCE_API_VERSION) {
        return err(interp, "the MDANCE library loaded is older than this extension (API " +
                               std::to_string(mdance_api_version()) + ", needs " + std::to_string(MDANCE_API_VERSION) + ")");
    }
    State* st = new State;
    st->interp = interp;
    st->thread = Tcl_GetCurrentThread();
    const char* th = Tcl_GetVar2(interp, "tcl_platform", "threaded", TCL_GLOBAL_ONLY);
    st->threaded = th && std::strcmp(th, "0") != 0;
    {
        std::lock_guard<std::mutex> g(statesLock);
        states.insert(st);
    }
    Tcl_CallWhenDeleted(interp, stateDeleted, st);

    Tcl_CreateNamespace(interp, "::mdance", nullptr, nullptr);
    for (auto& s : specs()) {
        std::string n = std::string("::mdance::") + s.name;
        Tcl_CreateObjCommand(interp, n.c_str(), callCmd, new CallCmd{st, &s}, callCmdDeleted);
    }
    Tcl_CreateObjCommand(interp, "::mdance::data", dataCmd, st, nullptr);
    Tcl_CreateObjCommand(interp, "::mdance::job", jobCmd, st, nullptr);
    Tcl_CreateObjCommand(interp, "::mdance::version", versionCmd, st, nullptr);
    Tcl_CreateObjCommand(interp, "::mdance::info", infoCmd, st, nullptr);

    Tcl_PkgProvide(interp, "mdance_native", "2.0");     // (the first version's name)
    return Tcl_PkgProvide(interp, "mdance", "2.0");
}

}  // extern "C"
