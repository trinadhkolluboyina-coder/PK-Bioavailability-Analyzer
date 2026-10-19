"""Pharmacokinetic evaluation app (Chapter X: Tables 10.1, 10.2, 10.3, Figs 10.1, 10.2)
Run:  pip install streamlit numpy pandas scipy plotly
      streamlit run pk_app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(page_title="PK Evaluation", layout="wide")

# ---------------------------------------------------------------- default data (from the PDF)
TIMES = [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1, 1.17, 1.5, 2, 4, 8, 12]
REF = np.array([
    [0, 0, 0, 0, 0, 0], [3.58, 4.21, 3.40, 4.26, 3.93, 5.02], [9.24, 8.36, 7.91, 9.63, 10.16, 8.09],
    [13.24, 11.36, 14.91, 11.63, 12.16, 11.09], [14.67, 16.12, 18.16, 14.69, 17.78, 15.56],
    [23.33, 21.78, 22.91, 19.57, 21.11, 22.53], [20.94, 19.35, 19.37, 21.46, 20.77, 19.03],
    [19.33, 20.83, 17.91, 19.57, 19.16, 18.53], [16.33, 15.78, 17.91, 17.57, 17.11, 18.53],
    [15.63, 16.93, 17.04, 15.86, 14.32, 14.89], [12.31, 11.78, 13.81, 11.47, 12.64, 13.53],
    [9.79, 10.56, 9.49, 10.75, 9.55, 10.46], [3.96, 4.59, 5.11, 6.07, 4.37, 6.01]])
TEST = np.array([
    [0, 0, 0, 0, 0, 0], [18.24, 19.36, 20.91, 18.63, 17.86, 20.59], [28.24, 29.36, 30.91, 28.63, 27.86, 30.59],
    [35.97, 37.12, 36.16, 35.69, 35.78, 37.16], [36.47, 36.13, 32.80, 35.97, 34.96, 35.85],
    [35.94, 35.35, 33.37, 34.56, 32.77, 34.03], [34.33, 34.78, 31.91, 33.57, 30.91, 33.73],
    [33.94, 33.78, 30.91, 32.57, 29.11, 32.53], [33.31, 32.78, 29.81, 31.57, 28.71, 32.03],
    [31.63, 31.20, 28.33, 30.04, 29.03, 28.03], [25.94, 25.35, 23.37, 24.56, 22.77, 24.03],
    [15.63, 16.93, 17.04, 15.86, 14.32, 14.89], [6.30, 7.20, 6.33, 6.30, 6.91, 7.84]])


def default_df(arr, n):
    """Time column + n subject columns (pad with NaN / trim as needed)."""
    data = {"Time (h)": TIMES}
    for j in range(n):
        data[f"S{j + 1}"] = arr[:, j] if j < arr.shape[1] else np.nan
    return pd.DataFrame(data)


# ---------------------------------------------------------------- PK calculations
def cum_trapz(x, y):
    seg = (x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2
    return np.concatenate([[0], np.cumsum(seg)])


def compute(t, c, n_term, ka_tmax, abs_t):
    """All parameters for ONE concentration-time profile."""
    t, c = np.asarray(t, float), np.asarray(c, float)
    o = np.argsort(t); t, c = t[o], c[o]
    r = {"t": t, "c": c}
    i = int(np.argmax(c))
    r["Cmax"], r["Tmax"] = c[i], t[i]

    # Kel: least-squares slope of ln C vs t over the last n_term points (terminal phase)
    pos = np.where(c > 0)[0][-n_term:]
    fit = stats.linregress(t[pos], np.log(c[pos]))
    kel = -fit.slope
    r.update(Kel=kel, t_half=0.693 / kel if kel > 0 else np.nan, term_idx=pos, term_fit=fit)

    # AUC / AUMC (trapezoidal rule) and extrapolation to infinity
    cauc = cum_trapz(t, c)
    auc_t = cauc[-1]
    aumc_t = cum_trapz(t, t * c)[-1]
    ct, tl = c[-1], t[-1]
    r["AUC_t"], r["AUMC_t"] = auc_t, aumc_t
    r["AUC_inf"] = auc_t + ct / kel                                  # AUC0-t + Ct/Kel
    r["AUMC_inf"] = aumc_t + tl * ct / kel + ct / kel ** 2
    r["MRT_t"] = aumc_t / auc_t
    r["MRT_inf"] = r["AUMC_inf"] / r["AUC_inf"]

    # Wagner-Nelson:  A_T/Vd = C_T + Kel * AUC(0-T)
    atvd = c + kel * cauc
    pct = atvd / atvd.max() * 100
    unabs = 100 - pct
    r.update(pct_abs=pct, pct_unabs=unabs)
    r["pct_abs_at"] = float(np.interp(abs_t, t, pct))
    m = (t <= ka_tmax) & (unabs > 0)
    if m.sum() >= 2:
        f = stats.linregress(t[m], np.log10(unabs[m]))               # slope = -Ka/2.303
        r.update(Ka=-f.slope * 2.303, ka_fit=f, ka_mask=m)
    else:
        r.update(Ka=np.nan, ka_fit=None, ka_mask=m)
    return r


def summary(df):
    s = df.iloc[:, 1:]
    out = pd.DataFrame({"Time (h)": df.iloc[:, 0]})
    for k in s.columns:
        out[k] = s[k]
    out["Mean"] = s.mean(axis=1)
    out["SD"] = s.std(axis=1, ddof=1)
    out["CV%"] = out["SD"] / out["Mean"] * 100
    return out.replace([np.inf, -np.inf], np.nan)


def clean(df):
    d = df.dropna(subset=[df.columns[0]]).dropna(axis=1, how="all")
    return d.sort_values(d.columns[0]).reset_index(drop=True)


# ---------------------------------------------------------------- sidebar
st.title("💊 Pharmacokinetic Evaluation — Reference vs Test")
with st.sidebar:
    st.header("Settings")
    name_r = st.text_input("Reference name", "Domstal")
    name_t = st.text_input("Test name", "F20")
    n_sub = st.number_input("Number of subjects", 2, 24, 6)
    n_term = st.slider("Points for Kel (terminal phase, last n)", 2, 6, 3)
    ka_tmax = st.number_input("Use points up to (h) for Ka fit", 0.5, 24.0, 2.0, 0.25)
    abs_t = st.number_input("% absorbed at time (h)", 0.1, 12.0, 0.5, 0.1)
    paired = st.checkbox("Paired t-test (crossover)", True)
    alpha = st.number_input("Significance level (P <)", 0.001, 0.2, 0.05, 0.005)

tabs = st.tabs(["1️⃣ Enter data (Tables 10.1 / 10.2)", "2️⃣ Fig 10.1", "3️⃣ Fig 10.2", "4️⃣ Table 10.3"])

with tabs[0]:
    st.caption("Edit values directly; add/delete rows with the table controls. Time in hours, concentration in ng/mL.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Table 10.1 – {name_r}")
        df_r = st.data_editor(default_df(REF, n_sub), num_rows="dynamic", key=f"r{n_sub}", use_container_width=True)
    with c2:
        st.subheader(f"Table 10.2 – {name_t}")
        df_t = st.data_editor(default_df(TEST, n_sub), num_rows="dynamic", key=f"t{n_sub}", use_container_width=True)

df_r, df_t = clean(df_r), clean(df_t)
try:
    sr, stt = summary(df_r), summary(df_t)
    R = compute(df_r.iloc[:, 0], sr["Mean"], n_term, ka_tmax, abs_t)
    T = compute(df_t.iloc[:, 0], stt["Mean"], n_term, ka_tmax, abs_t)
except Exception as e:
    st.error(f"Could not compute — check the data (need ≥ {n_term} positive terminal points). Details: {e}")
    st.stop()

with tabs[0]:
    st.markdown("**Mean / SD / CV% (SD = sample SD, n−1)**")
    fmt = {c: "{:.2f}" for c in sr.columns}
    a, b = st.columns(2)
    a.dataframe(sr.style.format(fmt, na_rep="–"), use_container_width=True)
    b.dataframe(stt.style.format(fmt, na_rep="–"), use_container_width=True)

# ---------------------------------------------------------------- Fig 10.1
with tabs[1]:
    fig = go.Figure()
    for nm, s, col in [(name_r, sr, "#3b5998"), (name_t, stt, "#c0392b")]:
        fig.add_scatter(x=s["Time (h)"], y=s["Mean"], mode="lines+markers", name=nm, line_color=col,
                        error_y=dict(type="data", array=s["SD"].fillna(0), visible=True))
    fig.update_layout(title="Fig 10.1: Time vs plasma concentration (mean ± SD)",
                      xaxis_title="Time (hours)", yaxis_title="Concentration (ng/ml)", height=520)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- Fig 10.2
with tabs[2]:
    fig2 = go.Figure()
    eqs = []
    for nm, X, col in [(name_r, R, "#3b5998"), (name_t, T, "#c0392b")]:
        m = X["ka_mask"]
        if X["ka_fit"] is None:
            st.warning(f"{nm}: fewer than 2 usable points for Ka fit — increase the time limit in the sidebar.")
            continue
        tt, yy = X["t"][m], np.log10(X["pct_unabs"][m])
        f = X["ka_fit"]
        fig2.add_scatter(x=tt, y=yy, mode="markers", name=nm, marker=dict(color=col, size=9))
        xs = np.array([tt.min(), tt.max()])
        fig2.add_scatter(x=xs, y=f.intercept + f.slope * xs, mode="lines", showlegend=False, line_color=col)
        eqs.append(f"**{nm}:** y = {f.slope:.4f}x + {f.intercept:.4f},  R² = {f.rvalue ** 2:.4f}  →  Ka = {X['Ka']:.3f} hr⁻¹")
    fig2.update_layout(title="Fig 10.2: Time vs log % drug unabsorbed (Wagner–Nelson)",
                       xaxis_title="Time (hours)", yaxis_title="Log % drug unabsorbed", height=520)
    st.plotly_chart(fig2, use_container_width=True)
    for e in eqs:
        st.markdown(e)
    with st.expander("Wagner–Nelson table (% absorbed vs time)"):
        st.dataframe(pd.DataFrame({
            "Time": R["t"], f"% absorbed {name_r}": R["pct_abs"], f"% unabsorbed {name_r}": R["pct_unabs"],
            f"% absorbed {name_t}": T["pct_abs"], f"% unabsorbed {name_t}": T["pct_unabs"]}).round(2))

# ---------------------------------------------------------------- Table 10.3
KEYS = [("Cmax (ng/ml)", "Cmax"), ("Tmax (hr)", "Tmax"), ("Ke (hr⁻¹)", "Kel"), ("Ka (hr⁻¹)", "Ka"),
        ("AUC0-t (ng·hr/ml)", "AUC_t"), ("AUC0-∞ (ng·hr/ml)", "AUC_inf"),
        ("MRT0-t (hr)", "MRT_t"), ("MRT0-∞ (hr)", "MRT_inf"),
        ("AUMC0-t (ng·hr/ml)", "AUMC_t"), ("AUMC0-∞ (ng·hr/ml)", "AUMC_inf"),
        ("Half life t1/2 (hr)", "t_half"), (f"% absorbed in {abs_t:g} hr", "pct_abs_at")]


def per_subject(df):
    rows = []
    for col in df.columns[1:]:
        d = df[[df.columns[0], col]].dropna()
        try:
            X = compute(d.iloc[:, 0], d.iloc[:, 1], n_term, ka_tmax, abs_t)
            rows.append({k: X[k] for _, k in KEYS})
        except Exception:
            rows.append({k: np.nan for _, k in KEYS})
    return pd.DataFrame(rows)


with tabs[3]:
    ps_r, ps_t = per_subject(df_r), per_subject(df_t)
    can_pair = paired and len(ps_r) == len(ps_t)
    rows = []
    for label, k in KEYS:
        a, b = ps_r[k].dropna(), ps_t[k].dropna()
        p = np.nan
        if len(a) > 1 and len(b) > 1:
            if can_pair:
                ok = ps_r[k].notna() & ps_t[k].notna()
                p = stats.ttest_rel(ps_r[k][ok], ps_t[k][ok]).pvalue
            else:
                p = stats.ttest_ind(a, b).pvalue
        rows.append({"Parameter": label, name_r: R[k], name_t: T[k],
                     "P-value": p, "Significance": "–" if np.isnan(p) else ("S" if p < alpha else "N.S.")})
    out = pd.DataFrame(rows)
    st.subheader("Table 10.3: Pharmacokinetic parameters")
    st.dataframe(out.style.format({name_r: "{:.4g}", name_t: "{:.4g}", "P-value": "{:.4f}"}, na_rep="–"),
                 use_container_width=True, hide_index=True)
    st.caption(
        f"Parameters are computed from the **mean** concentration–time curve (as in the thesis). "
        f"Significance uses a {'paired' if can_pair else 'unpaired'} Student's t-test on **per-subject** parameters "
        f"(S = P < {alpha}). Kel: regression of ln C on the last {n_term} points; t½ = 0.693/Kel; "
        "AUC by trapezoidal rule, AUC0-∞ = AUC0-t + Ct/Kel; MRT = AUMC/AUC; "
        "AUMC0-∞ = AUMC0-t + t·Ct/Kel + Ct/Kel²; Ka from slope = −Ka/2.303 of log % unabsorbed (Wagner–Nelson).")
    st.download_button("⬇ Download Table 10.3 (CSV)", out.to_csv(index=False), "table_10_3.csv", "text/csv")
    with st.expander("Per-subject parameters"):
        st.write(name_r); st.dataframe(ps_r.round(3))
        st.write(name_t); st.dataframe(ps_t.round(3))
