import streamlit as st
import pulp
import pandas as pd
import numpy as np

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VAP Optimizer",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── background ── */
.stApp {
    background: #0f1117;
    color: #e8e8e8;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #2a2f3e;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #7dd3fc;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 1.4rem;
    margin-bottom: 0.3rem;
}

/* ── number inputs ── */
[data-testid="stNumberInput"] input {
    background: #1e2433 !important;
    border: 1px solid #2e3548 !important;
    color: #e8e8e8 !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
}

/* ── data editor ── */
[data-testid="stDataEditor"] {
    border: 1px solid #2e3548 !important;
    border-radius: 8px !important;
    overflow: hidden;
}
[data-testid="stDataFrame"] {
    border: 1px solid #2e3548 !important;
    border-radius: 8px !important;
}

/* ── buttons ── */
.stButton > button {
    background: #0ea5e9 !important;
    color: #000 !important;
    border: none !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.03em !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.4rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #38bdf8 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(14,165,233,0.3) !important;
}

/* ── metrics ── */
[data-testid="metric-container"] {
    background: #161b27;
    border: 1px solid #2a2f3e;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.72rem;
    color: #7dd3fc;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace;
    font-size: 1.6rem;
    color: #f0f9ff;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #2a2f3e;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    color: #8899aa !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    padding: 0.6rem 1.2rem !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
    background: transparent !important;
}

/* ── section headers ── */
.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    color: #7dd3fc;
    margin-bottom: 0.4rem;
    margin-top: 1.2rem;
}

/* ── info / success boxes ── */
.info-box {
    background: #0c2233;
    border: 1px solid #1e4a6e;
    border-left: 3px solid #38bdf8;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    color: #bae6fd;
    margin-bottom: 0.8rem;
}
.success-box {
    background: #0b2e1a;
    border: 1px solid #166534;
    border-left: 3px solid #4ade80;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    color: #bbf7d0;
    margin-bottom: 0.8rem;
}
.warn-box {
    background: #2d1f00;
    border: 1px solid #854d0e;
    border-left: 3px solid #facc15;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    color: #fef08a;
    margin-bottom: 0.8rem;
}

/* ── assignment status badges ── */
.badge-ontime  { background:#14532d; color:#86efac; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; }
.badge-late    { background:#450a0a; color:#fca5a5; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; }
.badge-early   { background:#1e3a5f; color:#93c5fd; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; }

/* ── divider ── */
hr { border-color: #2a2f3e !important; }

/* ── expander ── */
[data-testid="stExpander"] {
    border: 1px solid #2a2f3e !important;
    border-radius: 8px !important;
    background: #161b27 !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SOLVER HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def TOA_single(v, n, d_origin, d, d_dest):
    return d_origin[(n, v)] + d[n] + d_dest[(n, v)]

def TOA_combined(v, n, k, d_origin, d, eps, d_dest):
    return d_origin[(n, v)] + d[n] + eps[(n, k)] + d[k] + d_dest[(k, v)]

def emission_single(v, n, is_electric, d_origin, d, d_dest, E_empty, E_full, p, Cap):
    if is_electric[v]: return 0.0
    toa = TOA_single(v, n, d_origin, d, d_dest)
    return toa * E_empty[v] + (E_full[v] - E_empty[v]) * d[n] * p[n] / Cap[v]

def emission_combined(v, n, k, is_electric, d_origin, d, eps, d_dest, E_empty, E_full, p, Cap):
    if is_electric[v]: return 0.0
    toa = TOA_combined(v, n, k, d_origin, d, eps, d_dest)
    return toa * E_empty[v] + (E_full[v] - E_empty[v]) * (d[n]*p[n] + d[k]*p[k]) / Cap[v]

def fmt_dev(dev):
    if dev > 0:   return f"LATE +{dev}d"
    elif dev < 0: return f"EARLY {dev}d"
    return "ON TIME"

def status_badge(dev):
    if dev > 0:   return f'<span class="badge-late">LATE +{dev}d</span>'
    elif dev < 0: return f'<span class="badge-early">EARLY {dev}d</span>'
    return '<span class="badge-ontime">ON TIME</span>'


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "n_carriers": 2,
        "n_trips": 5,
        "n_periods": 3,
        "vehs_per_carrier": [3, 3],
        "trips_per_carrier": [2, 3],
        "is_electric_list": [False, False, True, True, True, False],
        "alpha": 1.0, "beta": 0.8, "mu": 80.0, "c_E": 0.2,
        "T_drive": 8.0, "T_charge": 8.0, "income": 1.8,
        "solved": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS — build default matrices
# ═══════════════════════════════════════════════════════════════════════════

def get_structure():
    nc = st.session_state.n_carriers
    nt = st.session_state.n_trips
    tpc = st.session_state.trips_per_carrier[:nc]
    vpc = st.session_state.vehs_per_carrier[:nc]
    trips_of_carrier = {}
    vehicles_of_carrier = {}
    tid = 0; vid = 0
    for r in range(nc):
        trips_of_carrier[r]    = list(range(tid, tid + tpc[r]))
        vehicles_of_carrier[r] = list(range(vid, vid + vpc[r]))
        tid += tpc[r]; vid += vpc[r]
    all_vehicles = list(range(vid))
    carriers = list(range(nc))
    trips    = list(range(nt))
    periods  = list(range(1, st.session_state.n_periods + 1))
    return carriers, trips, periods, trips_of_carrier, vehicles_of_carrier, all_vehicles

def trip_labels(trips):   return [f"T{n}" for n in trips]
def veh_labels(vehicles): return [f"V{v}" for v in vehicles]
def carrier_labels(carriers): return [f"C{r}" for r in carriers]

def default_1row(cols, val=0.0):
    return pd.DataFrame([[val]*len(cols)], columns=cols)

def default_trip_veh(trips, vehicles, val=5.0):
    tl = trip_labels(trips); vl = veh_labels(vehicles)
    df = pd.DataFrame([[val]*len(vl)]*len(tl), index=tl, columns=vl)
    return df.reset_index().rename(columns={"index": "Trip ↓ / Vehicle →"})

def default_trip_trip(trips, val=0.0):
    tl = trip_labels(trips)
    data = []
    for i, ti in enumerate(tl):
        row = []
        for j, tj in enumerate(tl):
            row.append(0 if i == j else val)
        data.append(row)
    df = pd.DataFrame(data, index=tl, columns=tl)
    return df.reset_index().rename(columns={"index": "T_n \\ T_k"})

def default_op_cost(carriers, all_vehicles, vehicles_of_carrier):
    cl = carrier_labels(carriers); vl = veh_labels(all_vehicles)
    rows = []
    for r in carriers:
        row = []
        for v in all_vehicles:
            row.append(0.7 if v in vehicles_of_carrier[r] else 1000.0)
        rows.append(row)
    df = pd.DataFrame(rows, index=cl, columns=vl)
    return df.reset_index().rename(columns={"index": "Carrier ↓ / Vehicle →"})


# ═══════════════════════════════════════════════════════════════════════════
# SOLVER
# ═══════════════════════════════════════════════════════════════════════════

def solve_vap(p):
    carriers = p["carriers"]; trips = p["trips"]; periods = p["periods"]
    trips_of_carrier = p["trips_of_carrier"]
    vehicles_of_carrier = p["vehicles_of_carrier"]
    is_electric = p["is_electric"]
    alpha = p["alpha"]; beta = p["beta"]; mu = p["mu"]; c_E = p["c_E"]
    T_avail = p["T_avail"]
    d = p["d"]; delta = p["delta"]; pp = p["p"]; Cap = p["Cap"]
    E_empty = p["E_empty"]; E_full = p["E_full"]
    I = p["I"]; z = p["z"]
    d_origin = p["d_origin"]; d_dest = p["d_dest"]
    eps = p["eps"]; O = p["O"]
    op_cost = p["op_cost"]; op_cost_full = p["op_cost_full"]
    tau = p["tau"]; pi = p["pi"]; Delta = p["Delta"]

    # ── P1 ────────────────────────────────────────────────────────────────
    S0 = {}
    for r in carriers:
        pr = pulp.LpProblem(f"P1_{r}", pulp.LpMaximize)
        mt = trips_of_carrier[r]; mv = vehicles_of_carrier[r]
        chi = pulp.LpVariable.dicts("chi",
            [(n,v,t) for n in mt for v in mv for t in periods], cat="Binary")
        pr += pulp.lpSum(
            (alpha*d[n]*I[r]
             - TOA_single(v,n,d_origin,d,d_dest)*op_cost[r][v]
             - c_E*emission_single(v,n,is_electric,d_origin,d,d_dest,E_empty,E_full,pp,Cap)
             - pi[n]*Delta[(n,t)]) * chi[(n,v,t)]
            for n in mt for v in mv for t in periods)
        for n in mt:
            pr += pulp.lpSum(chi[(n,v,t)] for v in mv for t in periods) == 1
        for v in mv:
            for t in periods:
                pr += pulp.lpSum(chi[(n,v,t)] for n in mt) <= 1
        for t in periods:
            pr += pulp.lpSum(chi[(n,v,t)] for n in mt for v in mv) <= len(mv)
        for n in mt:
            for v in mv:
                if is_electric[v]:
                    for t in periods:
                        pr += chi[(n,v,t)]*TOA_single(v,n,d_origin,d,d_dest) <= T_avail[v]*mu
        pr.solve(pulp.PULP_CBC_CMD(msg=0))
        S0[r] = pulp.value(pr.objective) if pr.status == 1 else 0.0

    # ── P2 ────────────────────────────────────────────────────────────────
    prob = pulp.LpProblem("P2", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x",
        [(r,v,n,t) for r in carriers for v in vehicles_of_carrier[r]
         for n in trips for t in periods], cat="Binary")
    y = pulp.LpVariable.dicts("y",
        [(r,v,n,k,t) for r in carriers for v in vehicles_of_carrier[r]
         for n in trips for k in trips if k!=n for t in periods], cat="Binary")

    def S1(r):
        terms = []
        for n in trips:
            for v in vehicles_of_carrier[r]:
                for t in periods:
                    rev  = alpha*d[n]*I[r]
                    opc  = TOA_single(v,n,d_origin,d,d_dest)*op_cost_full[(r,v)]
                    emc  = beta*c_E*emission_single(v,n,is_electric,d_origin,d,d_dest,E_empty,E_full,pp,Cap)
                    pen  = pi[n]*Delta[(n,t)]
                    terms.append((rev-opc-emc-pen)*x[(r,v,n,t)])
        return pulp.lpSum(terms)

    def S2(r):
        terms = []
        for n in trips:
            for k in trips:
                if k==n: continue
                for v in vehicles_of_carrier[r]:
                    for t in periods:
                        rev  = alpha*(d[n]+d[k])*I[r]
                        opc  = TOA_combined(v,n,k,d_origin,d,eps,d_dest)*op_cost_full[(r,v)]
                        emc  = beta*c_E*emission_combined(v,n,k,is_electric,d_origin,d,eps,d_dest,E_empty,E_full,pp,Cap)
                        pen  = pi[n]*Delta[(n,t)]+pi[k]*Delta[(k,t)]
                        terms.append((rev-opc-emc-pen)*y[(r,v,n,k,t)])
        return pulp.lpSum(terms)

    def S3(r):
        terms = []
        for n in trips:
            for rp in carriers:
                if rp==r: continue
                for v in vehicles_of_carrier[rp]:
                    for t in periods:
                        terms.append(+delta[n]*z[(r,n)]*x[(rp,v,n,t)])
                for v in vehicles_of_carrier[r]:
                    for t in periods:
                        terms.append(-delta[n]*z[(rp,n)]*x[(r,v,n,t)])
        for n in trips:
            for k in trips:
                if k==n: continue
                for rp in carriers:
                    if rp==r: continue
                    for v in vehicles_of_carrier[rp]:
                        for t in periods:
                            terms.append(+delta[n]*z[(r,n)]*y[(rp,v,n,k,t)])
                            terms.append(+delta[k]*z[(r,k)]*y[(rp,v,n,k,t)])
                    for v in vehicles_of_carrier[r]:
                        for t in periods:
                            terms.append(-delta[n]*z[(rp,n)]*y[(r,v,n,k,t)])
                            terms.append(-delta[k]*z[(rp,k)]*y[(r,v,n,k,t)])
        return pulp.lpSum(terms)

    prob += pulp.lpSum(S1(r)+S2(r)+S3(r) for r in carriers)

    for r in carriers:
        prob += S1(r)+S2(r)+S3(r) >= S0[r]
    for n in trips:
        prob += (
            pulp.lpSum(x[(r,v,n,t)] for r in carriers for v in vehicles_of_carrier[r] for t in periods)
            + pulp.lpSum(y[(r,v,n,k,t)]+y[(r,v,k,n,t)]
                         for r in carriers for v in vehicles_of_carrier[r]
                         for k in trips if k!=n for t in periods)
            == 1)
    for r in carriers:
        for v in vehicles_of_carrier[r]:
            for t in periods:
                prob += (pulp.lpSum(x[(r,v,n,t)] for n in trips)
                         + pulp.lpSum(y[(r,v,n,k,t)] for n in trips for k in trips if k!=n)
                         <= 1)
    for r in carriers:
        for t in periods:
            prob += (pulp.lpSum(x[(r,v,n,t)] for v in vehicles_of_carrier[r] for n in trips)
                     + pulp.lpSum(y[(r,v,n,k,t)] for v in vehicles_of_carrier[r]
                                  for n in trips for k in trips if k!=n)
                     <= len(vehicles_of_carrier[r]))
    for r in carriers:
        for v in vehicles_of_carrier[r]:
            if is_electric[v]:
                for n in trips:
                    for t in periods:
                        prob += x[(r,v,n,t)]*TOA_single(v,n,d_origin,d,d_dest) <= T_avail[v]*mu
                for n in trips:
                    for k in trips:
                        if k!=n:
                            for t in periods:
                                prob += y[(r,v,n,k,t)]*TOA_combined(v,n,k,d_origin,d,eps,d_dest) <= T_avail[v]*mu
    for r in carriers:
        for v in vehicles_of_carrier[r]:
            for n in trips:
                for k in trips:
                    if k!=n:
                        for t in periods:
                            prob += y[(r,v,n,k,t)] <= O[(n,k)]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]
    total  = pulp.value(prob.objective) if prob.status==1 else None

    # ── results ───────────────────────────────────────────────────────────
    profit_rows = []
    for r in carriers:
        s1v=pulp.value(S1(r)); s2v=pulp.value(S2(r)); s3v=pulp.value(S3(r))
        profit_rows.append({
            "Carrier": f"C{r}",
            "Initial S0 (€)": round(S0[r],2),
            "Single S1 (€)":  round(s1v,2),
            "Combined S2 (€)":round(s2v,2),
            "Compensation S3 (€)":round(s3v,2),
            "Final Total (€)":round(s1v+s2v+s3v,2),
            "Gain vs S0 (€)": round((s1v+s2v+s3v)-S0[r],2),
        })

    assign_rows = []
    total_pen = 0.0
    for r in carriers:
        for v in vehicles_of_carrier[r]:
            vtype = "E" if is_electric[v] else "D"
            for n in trips:
                for t in periods:
                    val = pulp.value(x[(r,v,n,t)])
                    if val and val > 0.5:
                        pen = pi[n]*Delta[(n,t)]
                        total_pen += pen
                        dev = t - tau[n]
                        assign_rows.append({
                            "Type": "Single",
                            "Trip(s)": f"T{n}",
                            "Planned": tau[n],
                            "Actual":  t,
                            "Carrier": f"C{r}",
                            "Vehicle": f"V{v} ({vtype})",
                            "Penalty (€)": round(pen,1),
                            "dev_n": dev, "dev_k": None,
                        })
            for n in trips:
                for k in trips:
                    if k==n: continue
                    for t in periods:
                        val = pulp.value(y[(r,v,n,k,t)])
                        if val and val > 0.5:
                            pen = pi[n]*Delta[(n,t)]+pi[k]*Delta[(k,t)]
                            total_pen += pen
                            assign_rows.append({
                                "Type": "Combined",
                                "Trip(s)": f"T{n} + T{k}",
                                "Planned": f"{tau[n]}, {tau[k]}",
                                "Actual":  t,
                                "Carrier": f"C{r}",
                                "Vehicle": f"V{v} ({vtype})",
                                "Penalty (€)": round(pen,1),
                                "dev_n": t-tau[n], "dev_k": t-tau[k],
                            })

    return status, total, pd.DataFrame(profit_rows), assign_rows, total_pen, S0


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — dimensions & scalars
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🚛 VAP Optimizer")
    st.markdown("*Time-Indexed Cooperative Planning*")
    st.markdown("---")

    st.markdown("### Dimensions")
    nc = st.number_input("Carriers", 1, 8, st.session_state.n_carriers, key="n_carriers")
    nt = st.number_input("Total Trips", 1, 20, st.session_state.n_trips, key="n_trips")
    np_ = st.number_input("Planning Periods (days)", 1, 7,
                           st.session_state.n_periods, key="n_periods")

    st.markdown("### Trips per Carrier")
    tpc = []
    for r in range(int(nc)):
        prev = st.session_state.trips_per_carrier[r] if r < len(st.session_state.trips_per_carrier) else max(1, int(nt)//int(nc))
        v = st.number_input(f"C{r} trips", 1, int(nt), prev,
                            key=f"tpc_{r}")
        tpc.append(int(v))
    st.session_state.trips_per_carrier = tpc

    tpc_sum = sum(tpc)
    if tpc_sum != int(nt):
        st.warning(f"Trip sum ({tpc_sum}) ≠ Total trips ({int(nt)})")

    st.markdown("### Vehicles per Carrier")
    vpc = []
    for r in range(int(nc)):
        prev = st.session_state.vehs_per_carrier[r] if r < len(st.session_state.vehs_per_carrier) else 3
        v = st.number_input(f"C{r} vehicles", 1, 15, prev,
                            key=f"vpc_{r}")
        vpc.append(int(v))
    st.session_state.vehs_per_carrier = vpc

    total_v = sum(vpc)
    st.markdown(f"<small style='color:#7dd3fc'>Total vehicles: **{total_v}**</small>",
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Model Scalars")
    alpha = st.number_input("α (revenue scale)", 0.1, 5.0, 1.0, 0.1)
    beta  = st.number_input("β (emission attribution)", 0.0, 1.0, 0.8, 0.05)
    mu    = st.number_input("μ speed (km/h)", 10.0, 150.0, 80.0, 5.0)
    c_E   = st.number_input("c_E cost/g CO₂ (€)", 0.0, 2.0, 0.2, 0.05)
    T_drive  = st.number_input("T_drive (h)", 1.0, 24.0, 8.0, 0.5)
    T_charge = st.number_input("T_charge (h)", 1.0, 24.0, 8.0, 0.5)
    income   = st.number_input("Income (€/km)", 0.1, 10.0, 1.8, 0.1)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════

carriers, trips, periods, trips_of_carrier, vehicles_of_carrier, all_vehicles = get_structure()
n_t = len(trips); n_v = len(all_vehicles); n_c = len(carriers)
tl = trip_labels(trips); vl = veh_labels(all_vehicles); cl = carrier_labels(carriers)

# validation banner
if sum(tpc) != int(nt):
    st.markdown(f'<div class="warn-box">⚠ Trip distribution ({sum(tpc)}) does not equal total trips ({int(nt)}). Adjust in sidebar before solving.</div>',
                unsafe_allow_html=True)
else:
    st.markdown(f'<div class="info-box">📐 {n_c} carriers · {n_t} trips · {n_v} vehicles · {len(periods)} periods — matrices resize automatically as you change dimensions.</div>',
                unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🚛  Fleet Setup",
    "📐  Distance Matrices",
    "📅  Time Parameters",
    "▶  Run & Results",
])


# ── Tab 1: Fleet Setup ────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-label">Vehicle Types</p>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Mark each vehicle as Electric (True) or Diesel (False). Rows follow order V0, V1, …</div>',
                    unsafe_allow_html=True)

        ev_labels = []
        for r in carriers:
            for v in vehicles_of_carrier[r]:
                ev_labels.append(f"V{v}  (C{r})")
        ev_default = []
        for i, v in enumerate(all_vehicles):
            prev = st.session_state.is_electric_list[i] if i < len(st.session_state.is_electric_list) else False
            ev_default.append({"Vehicle": ev_labels[i], "Electric?": prev})

        ev_df = st.data_editor(
            pd.DataFrame(ev_default),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Vehicle":   st.column_config.TextColumn(disabled=True),
                "Electric?": st.column_config.CheckboxColumn(),
            },
            key=f"ev_table_{n_c}_{n_v}",
        )
        st.session_state.is_electric_list = list(ev_df["Electric?"])
        is_electric = {v: bool(ev_df.iloc[i]["Electric?"]) for i, v in enumerate(all_vehicles)}
        T_avail = {v: min(T_drive, T_charge) if is_electric[v] else T_drive for v in all_vehicles}

        ev_count = sum(is_electric.values())
        st.markdown(f"<small style='color:#7dd3fc'>⚡ {ev_count} electric · 🛢 {n_v - ev_count} diesel</small>",
                    unsafe_allow_html=True)

    with col_b:
        st.markdown('<p class="section-label">Operating Cost  op_cost (€/km)</p>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Row = Carrier, Column = Vehicle. Use <strong>1000</strong> for vehicles not owned by that carrier.</div>',
                    unsafe_allow_html=True)

        op_default = default_op_cost(carriers, all_vehicles, vehicles_of_carrier)
        op_df = st.data_editor(
            op_default, use_container_width=True, hide_index=True,
            key=f"op_table_{n_c}_{n_v}",
        )

        st.markdown('<p class="section-label">Cargo Weight  p (kg) · Compensation  δ (€)</p>',
                    unsafe_allow_html=True)
        col_p, col_d = st.columns(2)
        with col_p:
            p_df = st.data_editor(
                default_1row(tl, 25.0), use_container_width=True, hide_index=True,
                key=f"p_table_{n_t}",
            )
        with col_d:
            delta_df = st.data_editor(
                default_1row(tl, 10.0), use_container_width=True, hide_index=True,
                key=f"delta_table_{n_t}",
            )

        st.markdown('<p class="section-label">Vehicle Capacity  Cap (kg) · Emissions (g/km)</p>',
                    unsafe_allow_html=True)
        cap_df = st.data_editor(
            default_1row(vl, 40.0), use_container_width=True, hide_index=True,
            key=f"cap_table_{n_v}",
        )
        col_ee, col_ef = st.columns(2)
        with col_ee:
            st.caption("E_empty")
            ee_df = st.data_editor(
                default_1row(vl, 0.3), use_container_width=True, hide_index=True,
                key=f"ee_table_{n_v}",
            )
        with col_ef:
            st.caption("E_full")
            ef_df = st.data_editor(
                default_1row(vl, 0.4), use_container_width=True, hide_index=True,
                key=f"ef_table_{n_v}",
            )


# ── Tab 2: Distance Matrices ──────────────────────────────────────────────
with tab2:
    st.markdown('<div class="info-box">All distances in <strong>km</strong>. Tables auto-resize when you change dimensions in the sidebar.</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-label">d: Trip Distance (km)</p>', unsafe_allow_html=True)
        d_df = st.data_editor(
            default_1row(tl, 60.0), use_container_width=True, hide_index=True,
            key=f"d_table_{n_t}",
        )

        st.markdown('<p class="section-label">d_origin: Depot → Trip Origin (km)</p>',
                    unsafe_allow_html=True)
        st.caption("Row = Trip, Column = Vehicle")
        dor_df = st.data_editor(
            default_trip_veh(trips, all_vehicles, 5.0),
            use_container_width=True, hide_index=True,
            key=f"dor_table_{n_t}_{n_v}",
        )

        st.markdown('<p class="section-label">d_dest: Trip Destination → Depot (km)</p>',
                    unsafe_allow_html=True)
        st.caption("Row = Trip, Column = Vehicle")
        dd_df = st.data_editor(
            default_trip_veh(trips, all_vehicles, 5.0),
            use_container_width=True, hide_index=True,
            key=f"dd_table_{n_t}_{n_v}",
        )

    with col2:
        st.markdown('<p class="section-label">ε: Repositioning Distance (km)</p>',
                    unsafe_allow_html=True)
        st.caption("Row = Trip n (origin), Column = Trip k (destination). Diagonal = 0.")
        eps_df = st.data_editor(
            default_trip_trip(trips, 50.0),
            use_container_width=True, hide_index=True,
            key=f"eps_table_{n_t}",
        )

        st.markdown('<p class="section-label">O: Combinability Matrix (0 / 1)</p>',
                    unsafe_allow_html=True)
        st.caption("1 = trips can be combined. Diagonal must be 0.")
        O_df = st.data_editor(
            default_trip_trip(trips, 0),
            use_container_width=True, hide_index=True,
            key=f"O_table_{n_t}",
        )


# ── Tab 3: Time Parameters ────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="info-box">Δ<sub>nt</sub> = |t − τ<sub>n</sub>| is computed automatically before solving — the model stays linear.</div>',
                unsafe_allow_html=True)

    col_tau, col_pi = st.columns(2)
    with col_tau:
        st.markdown('<p class="section-label">τ: Planned Execution Day</p>', unsafe_allow_html=True)
        st.caption(f"One value per trip. Must be in [1 … {len(periods)}].")
        tau_df = st.data_editor(
            default_1row(tl, 1), use_container_width=True, hide_index=True,
            key=f"tau_table_{n_t}_{len(periods)}",
        )

    with col_pi:
        st.markdown('<p class="section-label">π: Penalty Coefficient (€/day)</p>',
                    unsafe_allow_html=True)
        st.caption("Cost per day of deviation from planned day.")
        pi_df = st.data_editor(
            default_1row(tl, 10.0), use_container_width=True, hide_index=True,
            key=f"pi_table_{n_t}",
        )

    # preview delta table
    st.markdown('<p class="section-label">Δ Preview: Penalty Days per (Trip, Period)</p>',
                unsafe_allow_html=True)
    try:
        tau_vals = {n: int(tau_df.iloc[0, n]) for n in trips}
        delta_preview = pd.DataFrame(
            {f"Day {t}": [abs(t - tau_vals[n]) for n in trips] for t in periods},
            index=tl
        )
        st.dataframe(delta_preview, use_container_width=False)
        st.caption("Green = 0 (on time). Higher values = more penalty days.")
    except:
        st.info("Set τ values above to preview Δ.")


# ── Tab 4: Run & Results ──────────────────────────────────────────────────
with tab4:
    ready = (sum(tpc) == int(nt))

    if not ready:
        st.markdown('<div class="warn-box">⚠ Fix trip distribution in the sidebar before solving.</div>',
                    unsafe_allow_html=True)

    solve_btn = st.button("▶  Solve Optimization", disabled=not ready,
                          use_container_width=False)

    if solve_btn and ready:
        with st.spinner("Solving... please wait..."):
            try:
                # parse all inputs
                z = {(r, n): (1 if n in trips_of_carrier[r] else 0)
                     for r in carriers for n in trips}
                I_dict = {r: income for r in carriers}

                d_dict     = {n: float(d_df.iloc[0, n])     for n in trips}
                delta_dict = {n: float(delta_df.iloc[0, n]) for n in trips}
                p_dict     = {n: float(p_df.iloc[0, n])     for n in trips}
                Cap_dict   = {v: float(cap_df.iloc[0, i])   for i, v in enumerate(all_vehicles)}
                E_empty_d  = {v: float(ee_df.iloc[0, i])    for i, v in enumerate(all_vehicles)}
                E_full_d   = {v: float(ef_df.iloc[0, i])    for i, v in enumerate(all_vehicles)}

                op_cost_d = {r: {} for r in carriers}
                op_cost_full_d = {}
                for i, r in enumerate(carriers):
                    for j, v in enumerate(all_vehicles):
                        val = float(op_df.iloc[i, j + 1])
                        op_cost_full_d[(r, v)] = val
                        if v in vehicles_of_carrier[r]:
                            op_cost_d[r][v] = val

                d_origin_d = {(n, v): float(dor_df.iloc[n, j + 1])
                              for n in trips for j, v in enumerate(all_vehicles)}
                d_dest_d   = {(n, v): float(dd_df.iloc[n, j + 1])
                              for n in trips for j, v in enumerate(all_vehicles)}
                eps_d      = {(n, k): float(eps_df.iloc[n, k + 1])
                              for n in trips for k in trips}
                O_d        = {(n, k): int(O_df.iloc[n, k + 1])
                              for n in trips for k in trips}
                tau_d      = {n: int(tau_df.iloc[0, n])   for n in trips}
                pi_d       = {n: float(pi_df.iloc[0, n])  for n in trips}
                Delta_d    = {(n, t): abs(t - tau_d[n]) for n in trips for t in periods}

                params = dict(
                    carriers=carriers, trips=trips, periods=periods,
                    trips_of_carrier=trips_of_carrier,
                    vehicles_of_carrier=vehicles_of_carrier,
                    is_electric=is_electric, T_avail=T_avail,
                    alpha=alpha, beta=beta, mu=mu, c_E=c_E,
                    I=I_dict, z=z,
                    d=d_dict, delta=delta_dict, p=p_dict,
                    Cap=Cap_dict, E_empty=E_empty_d, E_full=E_full_d,
                    op_cost=op_cost_d, op_cost_full=op_cost_full_d,
                    d_origin=d_origin_d, d_dest=d_dest_d,
                    eps=eps_d, O=O_d,
                    tau=tau_d, pi=pi_d, Delta=Delta_d,
                )

                status, total, profit_df, assign_rows, total_pen, S0 = solve_vap(params)
                st.session_state.solved = True
                st.session_state.last_result = (status, total, profit_df,
                                                assign_rows, total_pen, S0)
            except Exception as e:
                import traceback
                st.error(f"Solver error: {e}\n\n{traceback.format_exc()}")

    # ── show results ──────────────────────────────────────────────────────
    if st.session_state.get("solved") and "last_result" in st.session_state:
        status, total, profit_df, assign_rows, total_pen, S0 = st.session_state.last_result

        st.markdown("---")

        # top metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Solver Status", status)
        with m2:
            st.metric("Total Profit",
                      f"€ {total:.2f}" if total else "—")
        with m3:
            st.metric("Total Penalty", f"€ {total_pen:.2f}",
                      delta=f"-€{total_pen:.2f}" if total_pen > 0 else None,
                      delta_color="inverse")
        with m4:
            total_gain = sum(profit_df["Gain vs S0 (€)"])
            st.metric("Cooperation Gain", f"€ {total_gain:.2f}",
                      delta=f"+€{total_gain:.2f}" if total_gain > 0 else None)

        st.markdown("---")

        # profit table
        st.markdown('<p class="section-label">Profit Breakdown per Carrier</p>',
                    unsafe_allow_html=True)
        st.dataframe(profit_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # assignment table with status badges
        st.markdown('<p class="section-label">Trip Assignments</p>',
                    unsafe_allow_html=True)

        if assign_rows:
            # build display df
            display_rows = []
            for row in assign_rows:
                if row["Type"] == "Single":
                    dev = row["dev_n"]
                    status_html = status_badge(dev)
                else:
                    dn = row["dev_n"]; dk = row["dev_k"]
                    status_html = (f"T{row['Trip(s)'].split('+')[0].strip()[1:]}:{status_badge(dn)} "
                                   f"T{row['Trip(s)'].split('+')[1].strip()[1:]}:{status_badge(dk)}")
                display_rows.append({
                    "Type":         row["Type"],
                    "Trip(s)":      row["Trip(s)"],
                    "Planned Day":  row["Planned"],
                    "Actual Day":   row["Actual"],
                    "Carrier":      row["Carrier"],
                    "Vehicle":      row["Vehicle"],
                    "Penalty (€)":  row["Penalty (€)"],
                    "Status":       status_html,
                })

            display_df = pd.DataFrame(display_rows)
            st.write(
                display_df.to_html(escape=False, index=False,
                                   classes="dataframe",
                                   border=0),
                unsafe_allow_html=True
            )

            # summary stats
            st.markdown("---")
            s1, s2, s3 = st.columns(3)
            n_single   = sum(1 for r in assign_rows if r["Type"]=="Single")
            n_combined = sum(1 for r in assign_rows if r["Type"]=="Combined")
            n_ontime   = sum(1 for r in assign_rows
                             if r["dev_n"]==0 and (r["dev_k"] is None or r["dev_k"]==0))
            with s1:
                st.metric("Single Trips",   n_single)
            with s2:
                st.metric("Combined Trips", n_combined)
            with s3:
                st.metric("On Time",        n_ontime)
        else:
            st.info("No assignments found.")
