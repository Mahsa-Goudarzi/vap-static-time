import streamlit as st
import pulp
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="VAP Optimizer: Time-Indexed",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: theme-aware (works in both light and dark mode) ────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* buttons — keep our accent colour regardless of theme */
.stButton > button {
    background: #0ea5e9 !important;
    color: #000 !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.4rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #38bdf8 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(14,165,233,0.3) !important;
}
.stButton > button:disabled {
    opacity: 0.4 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* tabs — accent the selected tab */
.stTabs [data-baseweb="tab"] {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    padding: 0.6rem 1.2rem !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #0ea5e9 !important;
    border-bottom: 2px solid #0ea5e9 !important;
    background: transparent !important;
}

/* section label */
.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    color: #0ea5e9;
    margin-bottom: 0.3rem;
    margin-top: 1rem;
}

/* info / warn boxes — use transparent backgrounds so they adapt to theme */
.info-box {
    border: 1px solid #38bdf8;
    border-left: 3px solid #38bdf8;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    margin-bottom: 0.8rem;
    opacity: 0.9;
}
.warn-box {
    border: 1px solid #facc15;
    border-left: 3px solid #facc15;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    margin-bottom: 0.8rem;
    opacity: 0.9;
}

/* badges — keep vivid colours, they sit inside an HTML table */
.badge-ontime { background:#166534; color:#bbf7d0; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; }
.badge-late   { background:#7f1d1d; color:#fecaca; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; }
.badge-early  { background:#1e3a5f; color:#bfdbfe; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; }
.badge-E      { background:#14532d; color:#86efac; padding:2px 7px; border-radius:4px; font-size:0.75rem; font-weight:600; }
.badge-D      { background:#78350f; color:#fed7aa; padding:2px 7px; border-radius:4px; font-size:0.75rem; font-weight:600; }

/* assignment html table */
.asgn-table { width:100%; border-collapse:collapse; font-size:0.84rem; }
.asgn-table th {
    padding: 8px 12px; text-align:left;
    font-size:0.68rem; font-weight:600;
    letter-spacing:0.1em; text-transform:uppercase;
    color:#0ea5e9; border-bottom:2px solid #0ea5e9;
}
.asgn-table td { padding: 7px 12px; border-bottom: 1px solid rgba(128,128,128,0.15); }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# BASE SCENARIO — from the notebook (complete data)
# ═══════════════════════════════════════════════════════════════════════════

BASE = {
    # structure
    "n_carriers":       2,
    "n_trips":          5,
    "n_periods":        3,
    "trips_per_carrier":[2, 3],
    "vehs_per_carrier": [3, 3],
    "is_electric_list": [False, False, True, True, True, False],

    # scalars
    "alpha": 1.0, "beta": 0.8, "mu": 80.0, "c_E": 0.2,
    "T_drive": 8.0, "T_charge": 8.0, "income": 1.8,

    # 1-row arrays  (length = n_trips)
    "d":     [150, 100, 80, 120, 130],
    "delta": [20,  20,  10, 10,  10 ],
    "p":     [25,  25,  25, 25,  25 ],

    # 1-row arrays  (length = n_vehicles = 6)
    "Cap":     [40,  40,  40,  40,  40,  40 ],
    "E_empty": [0.3, 0.2, 0.3, 0.3, 0.3, 0.3],
    "E_full":  [0.3, 0.4, 0.3, 0.3, 0.4, 0.3],

    # carrier × vehicle  (2 × 6)
    "op_cost": [
        [0.8,  0.8,  0.7,  1000, 1000, 1000],
        [1000, 1000, 1000, 0.7,  0.7,  0.8 ],
    ],

    # trip × vehicle  (5 × 6)
    "d_origin": [
        [0,   0,   0,   50,  50,  50 ],
        [0,   0,   0,   50,  50,  50 ],
        [100, 100, 100, 80,  80,  80 ],
        [50,  50,  50,  0,   0,   0  ],
        [50,  50,  50,  0,   0,   0  ],
    ],
    "d_dest": [
        [150, 150, 150, 200, 200, 200],
        [100, 100, 100, 150, 150, 150],
        [50,  50,  50,  0,   0,   0  ],
        [170, 170, 170, 120, 120, 120],
        [180, 180, 180, 130, 130, 130],
    ],

    # trip × trip  (5 × 5)
    "eps": [
        [0,   150, 60,  140, 140],
        [150, 0,   0,   80,  80 ],
        [50,  50,  0,   0,   0  ],
        [170, 200, 200, 0,   120],
        [180, 180, 140, 130, 0  ],
    ],
    "O": [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ],

    # time  (length = n_trips)
    "tau": [1, 1, 2, 2, 3],
    "pi":  [10.0, 8.0, 12.0, 9.0, 11.0],
}


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

def init():
    if "initialized" not in st.session_state:
        for k, v in BASE.items():
            st.session_state[k] = v
        st.session_state.initialized = True
        st.session_state.solved = False

init()


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def get_structure():
    nc  = int(st.session_state.n_carriers)
    nt  = int(st.session_state.n_trips)
    np_ = int(st.session_state.n_periods)
    tpc = st.session_state.trips_per_carrier[:nc]
    vpc = st.session_state.vehs_per_carrier[:nc]
    carriers = list(range(nc))
    trips    = list(range(nt))
    periods  = list(range(1, np_ + 1))
    trips_of_carrier    = {}
    vehicles_of_carrier = {}
    tid = 0; vid = 0
    for r in carriers:
        trips_of_carrier[r]    = list(range(tid, tid + tpc[r]))
        vehicles_of_carrier[r] = list(range(vid, vid + vpc[r]))
        tid += tpc[r]; vid += vpc[r]
    all_vehicles = list(range(vid))
    return carriers, trips, periods, trips_of_carrier, vehicles_of_carrier, all_vehicles

def tl(trips):    return [f"T{n}" for n in trips]
def vl(vehicles): return [f"V{v}" for v in vehicles]
def cl(carriers): return [f"C{r}" for r in carriers]

def base_1row(key, cols):
    """Build a 1-row DataFrame from BASE data, padded/trimmed to len(cols)."""
    src = BASE.get(key, [])
    row = []
    for i, c in enumerate(cols):
        row.append(float(src[i]) if i < len(src) else 0.0)
    return pd.DataFrame([row], columns=cols)

def base_matrix(key, row_labels, col_labels, corner):
    """Build a matrix DataFrame from BASE data."""
    src = BASE.get(key, [])
    rows = []
    for i, rl in enumerate(row_labels):
        row = [rl]
        for j in range(len(col_labels)):
            try:    row.append(float(src[i][j]))
            except: row.append(0.0)
        rows.append(row)
    return pd.DataFrame(rows, columns=[corner] + col_labels)

def base_op_cost(carriers, all_vehicles, vehicles_of_carrier):
    src = BASE.get("op_cost", [])
    rows = []
    for i, r in enumerate(carriers):
        row = [f"C{r}"]
        for j, v in enumerate(all_vehicles):
            try:    row.append(float(src[i][j]))
            except: row.append(0.7 if v in vehicles_of_carrier[r] else 1000.0)
        rows.append(row)
    return pd.DataFrame(rows, columns=["Carrier ↓ / Vehicle →"] + vl(all_vehicles))

def status_badge(dev):
    if dev > 0:  return f'<span class="badge-late">LATE +{dev}d</span>'
    if dev < 0:  return f'<span class="badge-early">EARLY {dev}d</span>'
    return '<span class="badge-ontime">ON TIME</span>'


# ═══════════════════════════════════════════════════════════════════════════
# SOLVER
# ═══════════════════════════════════════════════════════════════════════════

def TOA_single(v, n, d_origin, d, d_dest):
    return d_origin[(n,v)] + d[n] + d_dest[(n,v)]

def TOA_combined(v, n, k, d_origin, d, eps, d_dest):
    return d_origin[(n,v)] + d[n] + eps[(n,k)] + d[k] + d_dest[(k,v)]

def emission_single(v, n, is_electric, d_origin, d, d_dest, E_empty, E_full, p, Cap):
    if is_electric[v]: return 0.0
    toa = TOA_single(v, n, d_origin, d, d_dest)
    return toa*E_empty[v] + (E_full[v]-E_empty[v])*d[n]*p[n]/Cap[v]

def emission_combined(v, n, k, is_electric, d_origin, d, eps, d_dest, E_empty, E_full, p, Cap):
    if is_electric[v]: return 0.0
    toa = TOA_combined(v, n, k, d_origin, d, eps, d_dest)
    return toa*E_empty[v] + (E_full[v]-E_empty[v])*(d[n]*p[n]+d[k]*p[k])/Cap[v]

def solve_vap(par):
    carriers=par["carriers"]; trips=par["trips"]; periods=par["periods"]
    toc=par["trips_of_carrier"]; voc=par["vehicles_of_carrier"]
    ise=par["is_electric"]; alpha=par["alpha"]; beta=par["beta"]
    mu=par["mu"]; c_E=par["c_E"]; T_avail=par["T_avail"]
    d=par["d"]; delta=par["delta"]; pp=par["p"]; Cap=par["Cap"]
    E_empty=par["E_empty"]; E_full=par["E_full"]
    I=par["I"]; z=par["z"]
    do=par["d_origin"]; dd=par["d_dest"]; eps=par["eps"]; O=par["O"]
    oc=par["op_cost"]; ocf=par["op_cost_full"]
    tau=par["tau"]; pi=par["pi"]; Delta=par["Delta"]

    # P1
    S0 = {}
    for r in carriers:
        pr = pulp.LpProblem(f"P1_{r}", pulp.LpMaximize)
        mt=toc[r]; mv=voc[r]
        chi = pulp.LpVariable.dicts("chi",
            [(n,v,t) for n in mt for v in mv for t in periods], cat="Binary")
        pr += pulp.lpSum(
            (alpha*d[n]*I[r]
             - TOA_single(v,n,do,d,dd)*oc[r][v]
             - c_E*emission_single(v,n,ise,do,d,dd,E_empty,E_full,pp,Cap)
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
                if ise[v]:
                    for t in periods:
                        pr += chi[(n,v,t)]*TOA_single(v,n,do,d,dd) <= T_avail[v]*mu
        pr.solve(pulp.PULP_CBC_CMD(msg=0))
        S0[r] = pulp.value(pr.objective) if pr.status==1 else 0.0

    # P2
    prob = pulp.LpProblem("P2", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x",
        [(r,v,n,t) for r in carriers for v in voc[r]
         for n in trips for t in periods], cat="Binary")
    y = pulp.LpVariable.dicts("y",
        [(r,v,n,k,t) for r in carriers for v in voc[r]
         for n in trips for k in trips if k!=n for t in periods], cat="Binary")

    def S1(r):
        terms=[]
        for n in trips:
            for v in voc[r]:
                for t in periods:
                    terms.append((alpha*d[n]*I[r]
                        - TOA_single(v,n,do,d,dd)*ocf[(r,v)]
                        - beta*c_E*emission_single(v,n,ise,do,d,dd,E_empty,E_full,pp,Cap)
                        - pi[n]*Delta[(n,t)]) * x[(r,v,n,t)])
        return pulp.lpSum(terms)

    def S2(r):
        terms=[]
        for n in trips:
            for k in trips:
                if k==n: continue
                for v in voc[r]:
                    for t in periods:
                        terms.append((alpha*(d[n]+d[k])*I[r]
                            - TOA_combined(v,n,k,do,d,eps,dd)*ocf[(r,v)]
                            - beta*c_E*emission_combined(v,n,k,ise,do,d,eps,dd,E_empty,E_full,pp,Cap)
                            - pi[n]*Delta[(n,t)] - pi[k]*Delta[(k,t)]) * y[(r,v,n,k,t)])
        return pulp.lpSum(terms)

    def S3(r):
        terms=[]
        for n in trips:
            for rp in carriers:
                if rp==r: continue
                for v in voc[rp]:
                    for t in periods:
                        terms.append(+delta[n]*z[(r,n)]*x[(rp,v,n,t)])
                for v in voc[r]:
                    for t in periods:
                        terms.append(-delta[n]*z[(rp,n)]*x[(r,v,n,t)])
        for n in trips:
            for k in trips:
                if k==n: continue
                for rp in carriers:
                    if rp==r: continue
                    for v in voc[rp]:
                        for t in periods:
                            terms.append(+delta[n]*z[(r,n)]*y[(rp,v,n,k,t)])
                            terms.append(+delta[k]*z[(r,k)]*y[(rp,v,n,k,t)])
                    for v in voc[r]:
                        for t in periods:
                            terms.append(-delta[n]*z[(rp,n)]*y[(r,v,n,k,t)])
                            terms.append(-delta[k]*z[(rp,k)]*y[(r,v,n,k,t)])
        return pulp.lpSum(terms)

    prob += pulp.lpSum(S1(r)+S2(r)+S3(r) for r in carriers)
    for r in carriers:
        prob += S1(r)+S2(r)+S3(r) >= S0[r]
    for n in trips:
        prob += (
            pulp.lpSum(x[(r,v,n,t)] for r in carriers for v in voc[r] for t in periods)
            + pulp.lpSum(y[(r,v,n,k,t)]+y[(r,v,k,n,t)]
                         for r in carriers for v in voc[r]
                         for k in trips if k!=n for t in periods)
            == 1)
    for r in carriers:
        for v in voc[r]:
            for t in periods:
                prob += (pulp.lpSum(x[(r,v,n,t)] for n in trips)
                         + pulp.lpSum(y[(r,v,n,k,t)] for n in trips for k in trips if k!=n)
                         <= 1)
    for r in carriers:
        for t in periods:
            prob += (pulp.lpSum(x[(r,v,n,t)] for v in voc[r] for n in trips)
                     + pulp.lpSum(y[(r,v,n,k,t)] for v in voc[r]
                                  for n in trips for k in trips if k!=n)
                     <= len(voc[r]))
    for r in carriers:
        for v in voc[r]:
            if ise[v]:
                for n in trips:
                    for t in periods:
                        prob += x[(r,v,n,t)]*TOA_single(v,n,do,d,dd) <= T_avail[v]*mu
                for n in trips:
                    for k in trips:
                        if k!=n:
                            for t in periods:
                                prob += y[(r,v,n,k,t)]*TOA_combined(v,n,k,do,d,eps,dd) <= T_avail[v]*mu
    for r in carriers:
        for v in voc[r]:
            for n in trips:
                for k in trips:
                    if k!=n:
                        for t in periods:
                            prob += y[(r,v,n,k,t)] <= O[(n,k)]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]
    total  = pulp.value(prob.objective) if prob.status==1 else None

    profit_rows=[]
    for r in carriers:
        s1v=pulp.value(S1(r)); s2v=pulp.value(S2(r)); s3v=pulp.value(S3(r))
        profit_rows.append({"Carrier":f"C{r}",
            "Initial S0 (€)":round(S0[r],2),
            "Single S1 (€)":round(s1v,2),
            "Combined S2 (€)":round(s2v,2),
            "Compensation S3 (€)":round(s3v,2),
            "Final Total (€)":round(s1v+s2v+s3v,2),
            "Gain vs S0 (€)":round((s1v+s2v+s3v)-S0[r],2)})

    assign_rows=[]; total_pen=0.0
    for r in carriers:
        for v in voc[r]:
            vtype="E" if ise[v] else "D"
            for n in trips:
                for t in periods:
                    val=pulp.value(x[(r,v,n,t)])
                    if val and val>0.5:
                        pen=pi[n]*Delta[(n,t)]; total_pen+=pen
                        assign_rows.append({"Type":"Single","Trip(s)":f"T{n}",
                            "Planned":tau[n],"Actual":t,
                            "Carrier":f"C{r}","Vehicle":f"V{v}","vtype":vtype,
                            "Penalty (€)":round(pen,1),"dev_n":t-tau[n],"dev_k":None})
            for n in trips:
                for k in trips:
                    if k==n: continue
                    for t in periods:
                        val=pulp.value(y[(r,v,n,k,t)])
                        if val and val>0.5:
                            pen=pi[n]*Delta[(n,t)]+pi[k]*Delta[(k,t)]; total_pen+=pen
                            assign_rows.append({"Type":"Combined","Trip(s)":f"T{n} + T{k}",
                                "Planned":f"{tau[n]}, {tau[k]}","Actual":t,
                                "Carrier":f"C{r}","Vehicle":f"V{v}","vtype":vtype,
                                "Penalty (€)":round(pen,1),"dev_n":t-tau[n],"dev_k":t-tau[k]})

    return status, total, pd.DataFrame(profit_rows), assign_rows, total_pen, S0


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🚛 VAP Optimizer")
    st.markdown("*Time-Indexed Cooperative Planning*")

    st.markdown("---")
    st.markdown("### Dimensions")
    nc  = st.number_input("Carriers",             1, 8,  int(st.session_state.n_carriers),  key="n_carriers")
    nt  = st.number_input("Total Trips",          1, 20, int(st.session_state.n_trips),      key="n_trips")
    np_ = st.number_input("Planning Periods",     1, 7,  int(st.session_state.n_periods),    key="n_periods")

    st.markdown("### Trips per Carrier")
    tpc=[]
    for r in range(int(nc)):
        prev = st.session_state.trips_per_carrier[r] if r < len(st.session_state.trips_per_carrier) else max(1,int(nt)//int(nc))
        tpc.append(int(st.number_input(f"C{r} trips", 1, int(nt), int(prev), key=f"tpc_{r}")))
    st.session_state.trips_per_carrier = tpc
    tpc_sum = sum(tpc)
    if tpc_sum != int(nt):
        st.warning(f"Trip sum ({tpc_sum}) ≠ {int(nt)}")

    st.markdown("### Vehicles per Carrier")
    vpc=[]
    for r in range(int(nc)):
        prev = st.session_state.vehs_per_carrier[r] if r < len(st.session_state.vehs_per_carrier) else 3
        vpc.append(int(st.number_input(f"C{r} vehicles", 1, 15, int(prev), key=f"vpc_{r}")))
    st.session_state.vehs_per_carrier = vpc
    total_v = sum(vpc)
    st.caption(f"Total vehicles: {total_v}")

    st.markdown("---")
    st.markdown("### Model Scalars")
    alpha    = st.number_input("α (revenue scale)",    0.1, 5.0,  1.0,  0.1)
    beta     = st.number_input("β (emission share)",   0.0, 1.0,  0.8,  0.05)
    mu       = st.number_input("μ speed (km/h)",      10.0,150.0, 80.0, 5.0)
    c_E      = st.number_input("c_E (€/g CO₂)",       0.0, 2.0,  0.2,  0.05)
    T_drive  = st.number_input("T_drive (h)",          1.0, 24.0, 8.0,  0.5)
    T_charge = st.number_input("T_charge (h)",         1.0, 24.0, 8.0,  0.5)
    income   = st.number_input("Income (€/km)",        0.1, 10.0, 1.8,  0.1)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

carriers, trips, periods, trips_of_carrier, vehicles_of_carrier, all_vehicles = get_structure()
n_t=len(trips); n_v=len(all_vehicles); n_c=len(carriers)
t_labels=tl(trips); v_labels=vl(all_vehicles); c_labels=cl(carriers)
valid = (sum(tpc)==int(nt))

if not valid:
    st.markdown(f'<div class="warn-box">⚠ Trip distribution ({sum(tpc)}) ≠ total trips ({int(nt)}). Fix in sidebar.</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="info-box">📐 {n_c} carriers · {n_t} trips · {n_v} vehicles · {len(periods)} periods — all tables adapt automatically.</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🚛  Fleet Setup",
    "📐  Distance Matrices",
    "📅  Time Parameters",
    "▶  Run & Results",
])


# ── Tab 1 ─────────────────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-label">Vehicle Types</p>', unsafe_allow_html=True)
        st.caption("Check Electric? for EV. Rows follow vehicle order V0, V1, …")

        ev_rows=[]
        for r in carriers:
            for v in vehicles_of_carrier[r]:
                prev = st.session_state.is_electric_list[v] if v < len(st.session_state.is_electric_list) else False
                ev_rows.append({"Vehicle": f"V{v}  (C{r})", "Electric?": bool(prev)})
        ev_df = st.data_editor(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True,
            column_config={"Vehicle": st.column_config.TextColumn(disabled=True),
                           "Electric?": st.column_config.CheckboxColumn()},
            key=f"ev_{n_c}_{n_v}")
        st.session_state.is_electric_list = list(ev_df["Electric?"])
        is_electric = {v: bool(ev_df.iloc[i]["Electric?"]) for i,v in enumerate(all_vehicles)}
        T_avail = {v: min(T_drive,T_charge) if is_electric[v] else T_drive for v in all_vehicles}
        ev_count = sum(is_electric.values())
        st.caption(f"⚡ {ev_count} electric · 🛢 {n_v-ev_count} diesel")

        st.markdown('<p class="section-label">Cargo Weight p (kg)</p>', unsafe_allow_html=True)
        p_df = st.data_editor(base_1row("p", t_labels), use_container_width=True,
                               hide_index=True, key=f"p_{n_t}")

        st.markdown('<p class="section-label">Compensation Cost δ (€)</p>', unsafe_allow_html=True)
        delta_df = st.data_editor(base_1row("delta", t_labels), use_container_width=True,
                                   hide_index=True, key=f"delta_{n_t}")

    with col_b:
        st.markdown('<p class="section-label">Operating Cost op_cost (€/km)</p>', unsafe_allow_html=True)
        st.caption("Row = Carrier, Column = Vehicle. Use 1000 for vehicles not owned.")
        op_df = st.data_editor(base_op_cost(carriers, all_vehicles, vehicles_of_carrier),
                               use_container_width=True, hide_index=True,
                               key=f"op_{n_c}_{n_v}")

        st.markdown('<p class="section-label">Vehicle Capacity Cap (kg)</p>', unsafe_allow_html=True)
        cap_df = st.data_editor(base_1row("Cap", v_labels), use_container_width=True,
                                hide_index=True, key=f"cap_{n_v}")

        st.markdown('<p class="section-label">Emissions (g/km)</p>', unsafe_allow_html=True)
        ce1, ce2 = st.columns(2)
        with ce1:
            st.caption("E_empty (vehicle empty)")
            ee_df = st.data_editor(base_1row("E_empty", v_labels), use_container_width=True,
                                   hide_index=True, key=f"ee_{n_v}")
        with ce2:
            st.caption("E_full (fully loaded)")
            ef_df = st.data_editor(base_1row("E_full", v_labels), use_container_width=True,
                                   hide_index=True, key=f"ef_{n_v}")


# ── Tab 2 ─────────────────────────────────────────────────────────────────
with tab2:
    st.caption("All values in km. Tables resize automatically.")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-label">d: Trip Distance (km)</p>', unsafe_allow_html=True)
        d_df = st.data_editor(base_1row("d", t_labels), use_container_width=True,
                              hide_index=True, key=f"d_{n_t}")

        st.markdown('<p class="section-label">d_origin: Depot → Trip Origin (km)</p>', unsafe_allow_html=True)
        st.caption("Row = Trip, Column = Vehicle")
        dor_df = st.data_editor(
            base_matrix("d_origin", t_labels, v_labels, "Trip ↓ / Veh →"),
            use_container_width=True, hide_index=True, key=f"dor_{n_t}_{n_v}")

        st.markdown('<p class="section-label">d_dest: Trip Destination → Depot (km)</p>', unsafe_allow_html=True)
        st.caption("Row = Trip, Column = Vehicle")
        dd_df = st.data_editor(
            base_matrix("d_dest", t_labels, v_labels, "Trip ↓ / Veh →"),
            use_container_width=True, hide_index=True, key=f"dd_{n_t}_{n_v}")

    with col2:
        st.markdown('<p class="section-label">ε: Repositioning Distance (km)</p>', unsafe_allow_html=True)
        st.caption("Row = Trip n, Column = Trip k. Diagonal = 0.")
        eps_df = st.data_editor(
            base_matrix("eps", t_labels, t_labels, "T_n \\ T_k"),
            use_container_width=True, hide_index=True, key=f"eps_{n_t}")

        st.markdown('<p class="section-label">O: Combinability (0 / 1)</p>', unsafe_allow_html=True)
        st.caption("1 = may be combined. Diagonal must be 0.")
        O_df = st.data_editor(
            base_matrix("O", t_labels, t_labels, "T_n \\ T_k"),
            use_container_width=True, hide_index=True, key=f"O_{n_t}")


# ── Tab 3 ─────────────────────────────────────────────────────────────────
with tab3:
    st.caption("Δ_nt = |t − τ_n| is computed automatically — model stays linear.")
    ct, cp = st.columns(2)

    with ct:
        st.markdown('<p class="section-label">τ: Planned Execution Day</p>', unsafe_allow_html=True)
        st.caption(f"One value per trip. Must be in [1 … {len(periods)}].")
        tau_df = st.data_editor(base_1row("tau", t_labels), use_container_width=True,
                                hide_index=True, key=f"tau_{n_t}_{len(periods)}")

    with cp:
        st.markdown('<p class="section-label">π: Penalty Coefficient (€/day)</p>', unsafe_allow_html=True)
        st.caption("Cost per day of deviation from τ.")
        pi_df = st.data_editor(base_1row("pi", t_labels), use_container_width=True,
                               hide_index=True, key=f"pi_{n_t}")

    st.markdown('<p class="section-label">Δ Preview: Penalty Days per (Trip, Period)</p>',
                unsafe_allow_html=True)
    try:
        tau_vals = {n: int(tau_df.iloc[0,n]) for n in trips}
        st.dataframe(
            pd.DataFrame({f"Day {t}": [abs(t-tau_vals[n]) for n in trips] for t in periods},
                         index=t_labels),
            use_container_width=False)
        st.caption("0 = on time. Higher = more penalty.")
    except:
        st.info("Set τ values above to preview Δ.")


# ── Tab 4 ─────────────────────────────────────────────────────────────────
with tab4:
    solve_btn = st.button("▶  Solve Optimization", disabled=not valid)

    if solve_btn and valid:
        with st.spinner("Solving... please wait..."):
            try:
                z = {(r,n):(1 if n in trips_of_carrier[r] else 0)
                     for r in carriers for n in trips}
                I_d = {r: income for r in carriers}

                d_d     = {n: float(d_df.iloc[0,n])     for n in trips}
                delta_d = {n: float(delta_df.iloc[0,n]) for n in trips}
                p_d     = {n: float(p_df.iloc[0,n])     for n in trips}
                Cap_d   = {v: float(cap_df.iloc[0,i])   for i,v in enumerate(all_vehicles)}
                E_empty_d={v: float(ee_df.iloc[0,i])    for i,v in enumerate(all_vehicles)}
                E_full_d ={v: float(ef_df.iloc[0,i])    for i,v in enumerate(all_vehicles)}

                oc_d={r:{} for r in carriers}; ocf_d={}
                for i,r in enumerate(carriers):
                    for j,v in enumerate(all_vehicles):
                        val=float(op_df.iloc[i,j+1])
                        ocf_d[(r,v)]=val
                        if v in vehicles_of_carrier[r]: oc_d[r][v]=val

                do_d={(n,v):float(dor_df.iloc[n,j+1]) for n in trips for j,v in enumerate(all_vehicles)}
                dd_d={(n,v):float(dd_df.iloc[n,j+1])  for n in trips for j,v in enumerate(all_vehicles)}
                eps_d={(n,k):float(eps_df.iloc[n,k+1]) for n in trips for k in trips}
                O_d  ={(n,k):int(O_df.iloc[n,k+1])    for n in trips for k in trips}
                tau_d={n:int(tau_df.iloc[0,n])         for n in trips}
                pi_d ={n:float(pi_df.iloc[0,n])        for n in trips}
                Delta_d={(n,t):abs(t-tau_d[n]) for n in trips for t in periods}

                params=dict(carriers=carriers,trips=trips,periods=periods,
                    trips_of_carrier=trips_of_carrier,
                    vehicles_of_carrier=vehicles_of_carrier,
                    is_electric=is_electric,T_avail=T_avail,
                    alpha=alpha,beta=beta,mu=mu,c_E=c_E,
                    I=I_d,z=z,d=d_d,delta=delta_d,p=p_d,Cap=Cap_d,
                    E_empty=E_empty_d,E_full=E_full_d,
                    op_cost=oc_d,op_cost_full=ocf_d,
                    d_origin=do_d,d_dest=dd_d,eps=eps_d,O=O_d,
                    tau=tau_d,pi=pi_d,Delta=Delta_d)

                status,total,profit_df,assign_rows,total_pen,S0 = solve_vap(params)
                st.session_state.solved = True
                st.session_state.last = (status,total,profit_df,assign_rows,total_pen,S0)
            except Exception as e:
                import traceback
                st.error(f"Solver error:\n\n{traceback.format_exc()}")

    if st.session_state.get("solved") and "last" in st.session_state:
        status,total,profit_df,assign_rows,total_pen,S0 = st.session_state.last
        st.markdown("---")

        # metrics
        m1,m2,m3,m4 = st.columns(4)
        with m1: st.metric("Solver Status", status)
        with m2: st.metric("Total Profit",  f"€ {total:.2f}" if total else "—")
        with m3: st.metric("Total Penalty", f"€ {total_pen:.2f}",
                           delta=f"-€{total_pen:.2f}" if total_pen>0 else None,
                           delta_color="inverse")
        with m4:
            gain = float(profit_df["Gain vs S0 (€)"].sum())
            st.metric("Cooperation Gain", f"€ {gain:.2f}",
                      delta=f"+€{gain:.2f}" if gain>0 else None)

        st.markdown("---")
        st.markdown('<p class="section-label">Profit Breakdown per Carrier</p>',
                    unsafe_allow_html=True)
        st.dataframe(profit_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown('<p class="section-label">Trip Assignments</p>', unsafe_allow_html=True)

        if assign_rows:
            rows_html=""
            for row in assign_rows:
                vbadge = (f'<span class="badge-E">⚡ {row["Vehicle"]}</span>'
                          if row["vtype"]=="E"
                          else f'<span class="badge-D">🛢 {row["Vehicle"]}</span>')
                if row["Type"]=="Single":
                    sbadge = status_badge(row["dev_n"])
                else:
                    dn=row["dev_n"]; dk=row["dev_k"]
                    sbadge = (f'T{row["Trip(s)"].split("+")[0].strip()[1:]}: {status_badge(dn)}'
                              f'&nbsp;&nbsp;'
                              f'T{row["Trip(s)"].split("+")[1].strip()[1:]}: {status_badge(dk)}')
                rows_html += (f"<tr>"
                              f"<td>{'<b>Combined</b>' if row['Type']=='Combined' else 'Single'}</td>"
                              f"<td><code>{row['Trip(s)']}</code></td>"
                              f"<td>{row['Planned']}</td>"
                              f"<td>{row['Actual']}</td>"
                              f"<td>{row['Carrier']}</td>"
                              f"<td>{vbadge}</td>"
                              f"<td>{row['Penalty (€)']} €</td>"
                              f"<td>{sbadge}</td>"
                              f"</tr>")

            st.write(f"""
<table class="asgn-table">
  <thead><tr>
    <th>Type</th><th>Trip(s)</th><th>Planned</th><th>Actual</th>
    <th>Carrier</th><th>Vehicle</th><th>Penalty</th><th>Status</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>""", unsafe_allow_html=True)

            st.markdown("---")
            sa,sb,sc,sd = st.columns(4)
            n_single   = sum(1 for r in assign_rows if r["Type"]=="Single")
            n_combined = sum(1 for r in assign_rows if r["Type"]=="Combined")
            n_ontime   = sum(1 for r in assign_rows
                             if r["dev_n"]==0 and (r["dev_k"] is None or r["dev_k"]==0))
            ev_trips   = sum(1 for r in assign_rows if r["vtype"]=="E")
            with sa: st.metric("Single Trips",    n_single)
            with sb: st.metric("Combined Trips",  n_combined)
            with sc: st.metric("On Time",         n_ontime)
            with sd: st.metric("Electric Vehicle Trips", ev_trips)
        else:
            st.info("No assignments found.")
