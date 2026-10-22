import io
import re
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from openpyxl.styles import Alignment, Font
from scipy import stats

st.set_page_config(page_title="PK Evaluation", page_icon="💊", layout="wide")

LN2 = 0.693  # same constant used in the notes (t1/2 = 0.693 / Kel)

# ======================================================================================
# Default reference data (from the notes) -> used ONLY to build the downloadable template
# ======================================================================================
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

SETTING_MAP = {"reference name": "name_r", "test name": "name_t",
               "time unit": "time_unit", "concentration unit": "conc_unit"}

DEFAULTS = dict(name_r="Reference", name_t="Test", time_unit="h", conc_unit="ng/mL",
                n_term=3, ka_tmax=2.0, abs_t=0.5, paired=True, alpha=0.05,
                col_r="#1f4e9c", col_t="#c0392b", ver=0, page="📥 Template & Upload")
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
ss = st.session_state

PAGES = ["📥 Template & Upload", "✏️ Edit / Delete data", "📈 Graphs", "📋 Tables"]


# ======================================================================================
# Helpers: files
# ======================================================================================
@st.cache_data
def make_template() -> bytes:
    n = REF.shape[1]
    cols = [f"S{i + 1}" for i in range(n)]
    ref = pd.DataFrame(REF, columns=cols); ref.insert(0, "Time", TIMES)
    tst = pd.DataFrame(TEST, columns=cols); tst.insert(0, "Time", TIMES)
    instr = pd.DataFrame({"How to use this template": [
        "1. Sheet 'Reference' = reference product, sheet 'Test' = test product. Do NOT rename these sheets.",
        "2. Column 1 = Time. Every other column = one subject (S1, S2, ...). Add more subjects/rows freely.",
        "3. Include the time 0 row (concentration 0) for every subject.",
        "4. Empty rows / empty columns / non-numeric cells are ignored automatically.",
        "5. Sheet 'Settings' (optional): product names and units used as defaults in the app.",
        "6. Save as .xlsx and upload it in the app, then press Submit.",
        "The sample values here are the reference data from the notes (Tables 10.1 and 10.2).",
    ]})
    settings = pd.DataFrame({"Setting": ["Reference name", "Test name", "Time unit", "Concentration unit"],
                             "Value": ["Domstal", "F20", "h", "ng/mL"]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        instr.to_excel(w, sheet_name="Instructions", index=False)
        settings.to_excel(w, sheet_name="Settings", index=False)
        ref.to_excel(w, sheet_name="Reference", index=False)
        tst.to_excel(w, sheet_name="Test", index=False)
        for ws in w.book.worksheets:
            for c in ws[1]:
                c.font = Font(bold=True)
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = 12
        w.book["Instructions"].column_dimensions["A"].width = 110
        w.book["Settings"].column_dimensions["A"].width = 24
        w.book["Settings"].column_dimensions["B"].width = 20
    return buf.getvalue()


def clean_profile(raw: pd.DataFrame):
    """Drop empty rows/cols, force numeric, merge duplicate times. Returns (df, report)."""
    df = raw.copy()
    n0 = len(df)
    df.columns = [str(c).strip() for c in df.columns]
    if df.shape[1] < 2:
        raise ValueError("need a Time column and at least one subject column")
    tcol = df.columns[0]
    df = df.apply(pd.to_numeric, errors="coerce")
    dropped_cols = [c for c in df.columns[1:] if df[c].isna().all()]
    df = df.drop(columns=dropped_cols).rename(columns={tcol: "Time"})
    conc = [c for c in df.columns if c != "Time"]
    if not conc:
        raise ValueError("no numeric subject columns found")
    df = df.dropna(subset=["Time"]).dropna(subset=conc, how="all")
    dups = int(df["Time"].duplicated().sum())
    if dups:
        df = df.groupby("Time", as_index=False).mean()
    df = df.sort_values("Time").reset_index(drop=True)
    if len(df) < 4:
        raise ValueError("need at least 4 valid time points")
    return df, dict(rows_removed=n0 - len(df) - dups, cols_removed=dropped_cols, dup_times=dups)


@st.cache_data
def read_workbook(raw: bytes):
    xl = pd.ExcelFile(io.BytesIO(raw))
    def find(prefix):
        for s in xl.sheet_names:
            if s.lower().strip().startswith(prefix):
                return s
    sr, stt = find("ref"), find("test")
    if not sr or not stt:
        raise ValueError("The workbook must contain sheets named 'Reference' and 'Test'.")
    ref, rep_r = clean_profile(xl.parse(sr))
    tst, rep_t = clean_profile(xl.parse(stt))
    settings = {}
    sn = find("setting")
    if sn:
        s = xl.parse(sn)
        if s.shape[1] >= 2:
            for a, b in zip(s.iloc[:, 0], s.iloc[:, 1]):
                if pd.notna(a) and pd.notna(b):
                    settings[str(a).strip().lower()] = str(b).strip()
    return dict(ref=ref, test=tst, report=dict(ref=rep_r, test=rep_t), settings=settings)


def safe_name(s: str, fallback="table", maxlen=60) -> str:
    s = re.sub(r"[^\w\-]+", "_", str(s)).strip("_")
    return (s or fallback)[:maxlen]


def safe_sheet(s: str, used: set) -> str:
    base = re.sub(r"[\[\]:*?/\\]", "", str(s)).strip()[:28] or "Sheet"
    name, i = base, 1
    while name.lower() in used:
        i += 1; name = f"{base}_{i}"
    used.add(name.lower())
    return name


def to_excel(sheets: dict) -> bytes:
    buf, used = io.BytesIO(), set()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for title, df in sheets.items():
            nm = safe_sheet(title, used)
            df.to_excel(w, sheet_name=nm, index=False)
            ws = w.book[nm]
            for c in ws[1]:
                c.font = Font(bold=True); c.alignment = Alignment(wrap_text=True)
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 40)
    return buf.getvalue()


def fig_bytes(fig, fmt, dpi):
    b = io.BytesIO()
    fig.savefig(b, format=fmt, dpi=dpi, facecolor="white", bbox_inches="tight")
    return b.getvalue()


def tin(label, default, key):
    """text_input whose default follows the settings (key changes when default changes)."""
    return st.text_input(label, default, key=f"{key}|{default}")


# ======================================================================================
# PK calculations
# ======================================================================================
def cum_trapz(x, y):
    seg = (x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2
    return np.concatenate([[0.0], np.cumsum(seg)])


def pk(t, c, n_term, ka_tmax, abs_t):
    """All PK parameters for ONE concentration-time profile."""
    t, c = np.asarray(t, float), np.asarray(c, float)
    ok = ~np.isnan(t) & ~np.isnan(c)
    t, c = t[ok], c[ok]
    if len(t) < 4:
        raise ValueError("too few valid points")
    o = np.argsort(t); t, c = t[o], c[o]
    r = {"t": t, "c": c}
    i = int(np.argmax(c))
    r["Cmax"], r["Tmax"] = c[i], t[i]

    pos = np.where(c > 0)[0][-n_term:]                     # terminal phase = last n positive points
    if len(pos) < 2:
        raise ValueError("fewer than 2 positive terminal points")
    fit = stats.linregress(t[pos], np.log(c[pos]))
    kel = -fit.slope
    if not kel > 0:
        raise ValueError("Kel is not positive – check terminal points")
    r.update(Kel=kel, t_half=LN2 / kel, term_idx=pos)

    cauc = cum_trapz(t, c)
    auc_t, aumc_t = cauc[-1], cum_trapz(t, t * c)[-1]
    ct, tl = c[-1], t[-1]
    r.update(AUC_t=auc_t, AUMC_t=aumc_t, AUC_inf=auc_t + ct / kel,
             AUMC_inf=aumc_t + tl * ct / kel + ct / kel ** 2)
    r["MRT_t"] = aumc_t / auc_t
    r["MRT_inf"] = r["AUMC_inf"] / r["AUC_inf"]

    # Wagner-Nelson:  A_T/Vd = C_T + Kel * AUC(0-T)
    atvd = c + kel * cauc
    if atvd.max() <= 0:
        raise ValueError("Wagner-Nelson not computable")
    pct = atvd / atvd.max() * 100
    unabs = 100 - pct
    r.update(atvd=atvd, pct_abs=pct, pct_unabs=unabs, pct_abs_at=float(np.interp(abs_t, t, pct)))
    m = (t <= ka_tmax) & (unabs > 0)
    if m.sum() >= 2:
        f = stats.linregress(t[m], np.log10(unabs[m]))      # slope = -Ka / 2.303
        r.update(Ka=-f.slope * 2.303, ka_fit=f, ka_mask=m)
    else:
        r.update(Ka=np.nan, ka_fit=None, ka_mask=m)
    return r


def summary(df):
    s = df.drop(columns="Time")
    out = df.copy()
    out["Mean"] = s.mean(axis=1)
    out["SD"] = s.std(axis=1, ddof=1)
    out["CV%"] = (out["SD"] / out["Mean"] * 100).replace([np.inf, -np.inf], np.nan)
    return out


def param_spec(tu, cu, abs_t):
    return [("Cmax", cu, "Cmax"), ("Tmax", tu, "Tmax"),
            ("Ke (elimination rate constant)", f"1/{tu}", "Kel"),
            ("Ka (absorption rate constant)", f"1/{tu}", "Ka"),
            ("AUC 0-t", f"{cu}·{tu}", "AUC_t"), ("AUC 0-∞", f"{cu}·{tu}", "AUC_inf"),
            ("MRT 0-t", tu, "MRT_t"), ("MRT 0-∞", tu, "MRT_inf"),
            ("AUMC 0-t", f"{cu}·{tu}²", "AUMC_t"), ("AUMC 0-∞", f"{cu}·{tu}²", "AUMC_inf"),
            ("Half-life (t½)", tu, "t_half"), (f"% absorbed at {abs_t:g} {tu}", "%", "pct_abs_at")]


def per_subject(df, spec, n_term, ka_tmax, abs_t):
    rows = []
    for col in df.columns[1:]:
        d = df[["Time", col]].dropna()
        try:
            X = pk(d["Time"], d[col], n_term, ka_tmax, abs_t)
            rows.append({k: X[k] for _, _, k in spec})
        except Exception:
            rows.append({k: np.nan for _, _, k in spec})
    out = pd.DataFrame(rows)
    out.insert(0, "Subject", list(df.columns[1:]))
    return out


def msd(x):
    x = x.dropna()
    if len(x) == 0:
        return "–"
    return f"{x.mean():.4g}" if len(x) == 1 else f"{x.mean():.4g} ± {x.std(ddof=1):.4g}"


# ======================================================================================
# Footer
# ======================================================================================
def render_footer():
    st.markdown(
        """
        <style>
        .block-container { padding-bottom: 80px; }   /* keep content clear of the footer */
        .pk-footer {
            position: fixed; left: 0; bottom: 0; width: 100%;
            padding: 8px 16px; text-align: center; font-size: 0.82rem;
            background: var(--secondary-background-color, #f0f2f6);
            color: var(--text-color, #31333f);
            border-top: 1px solid rgba(128,128,128,0.35);
            z-index: 99;
        }
        .pk-footer a { color: inherit; text-decoration: underline; }
        </style>
        <div class="pk-footer">
            Developed by <b>K. Trinadh, Dr. SSSN Usha Devi N, Dr. G. Varalakshmi</b> ·
            <b>JNTUK</b>, Kakinada, India ·
            <a href="mailto:you@example.com">trinadhkolluboyina.com</a> ·
            
            <br>© 2026 UCEK, JNTU KAKINADA. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ======================================================================================
# Callbacks
# ======================================================================================
def do_submit():
    f = ss.get("uploader")
    if f is None:
        ss["msg"] = ("error", "Please choose an Excel file first."); return
    try:
        wb = read_workbook(f.getvalue())
    except Exception as e:
        ss["msg"] = ("error", f"Could not read the file: {e}"); return
    ss.update(ref=wb["ref"], test=wb["test"], ref0=wb["ref"].copy(), test0=wb["test"].copy(),
              report=wb["report"], ver=ss["ver"] + 1)
    for k, sk in SETTING_MAP.items():
        if k in wb["settings"]:
            ss[sk] = wb["settings"][k]
    ss["msg"] = ("success", "Data submitted. Use the sidebar to open Graphs / Tables, or edit the data first.")


def do_reset():
    for k in list(ss.keys()):
        del ss[k]                      # clears data, settings and the file uploader


def do_restore():
    ss["ref"], ss["test"] = ss["ref0"].copy(), ss["test0"].copy()
    ss["ver"] += 1


def do_delete_subjects():
    sel = ss.get("del_sel", [])
    for k in ("ref", "test"):
        ss[k] = ss[k].drop(columns=[c for c in sel if c in ss[k].columns])
    ss["del_sel"] = []
    ss["ver"] += 1


def do_add_subject():
    nums = [int(m.group(1)) for k in ("ref", "test") for c in ss[k].columns
            if (m := re.fullmatch(r"S(\d+)", str(c)))]
    new = f"S{(max(nums) if nums else 0) + 1}"
    for k in ("ref", "test"):
        ss[k][new] = np.nan
    ss["ver"] += 1


# ======================================================================================
# Sidebar
# ======================================================================================
with st.sidebar:
    st.title("💊 PK Evaluation")
    st.radio("Navigate", PAGES, key="page")
    st.button("↺ Reset everything", on_click=do_reset, use_container_width=True)
    st.divider()
    st.header("Names & units")
    st.text_input("Reference product name", key="name_r")
    st.text_input("Test product name", key="name_t")
    st.text_input("Time unit", key="time_unit")
    st.text_input("Concentration unit", key="conc_unit")
    c1, c2 = st.columns(2)
    c1.color_picker("Reference colour", key="col_r")
    c2.color_picker("Test colour", key="col_t")
    st.header("Calculation settings")
    st.slider("Points for Kel (last n positive points)", 2, 8, key="n_term")
    st.number_input("Ka fit: use points up to time", 0.1, 100.0, step=0.25, key="ka_tmax")
    st.number_input("% absorbed at time", 0.01, 100.0, step=0.1, key="abs_t")
    st.checkbox("Paired t-test (crossover, same subjects)", key="paired")
    st.number_input("Significance level (P <)", 0.001, 0.2, step=0.005, format="%.3f", key="alpha")

nm_r, nm_t = ss["name_r"].strip() or "Reference", ss["name_t"].strip() or "Test"
tu, cu = ss["time_unit"].strip(), ss["conc_unit"].strip()
n_term, ka_tmax, abs_t, alpha = ss["n_term"], ss["ka_tmax"], ss["abs_t"], ss["alpha"]
have_data = "ref" in ss

render_footer()          # footer shown on every page


def need_data():
    if not have_data:
        st.info("No data yet. Go to **📥 Template & Upload**, upload your Excel file and press **Submit**.")
        st.stop()


def get_results():
    try:
        sr, stt = summary(ss["ref"]), summary(ss["test"])
        R = pk(sr["Time"], sr["Mean"], n_term, ka_tmax, abs_t)
        T = pk(stt["Time"], stt["Mean"], n_term, ka_tmax, abs_t)
        return sr, stt, R, T
    except Exception as e:
        st.error(f"Could not calculate – please check the data / settings (Edit page). Details: {e}")
        st.stop()


def lab(name, unit):
    return f"{name} ({unit})" if unit.strip() else name


# ======================================================================================
# PAGE 1 – Template & Upload
# ======================================================================================
if ss["page"] == PAGES[0]:
    st.header("Template & Upload")
    st.markdown("**Step 1 – Download the template** (contains the default reference data; replace/extend it with yours).")
    st.download_button("⬇ Download Excel template", make_template(), "pk_input_template.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.markdown("**Step 2 – Upload your completed file**, then press **Submit**.")
    up = st.file_uploader("Excel file (.xlsx) with sheets 'Reference' and 'Test'", type=["xlsx"], key="uploader")
    if up is not None:
        try:
            wb = read_workbook(up.getvalue())
            st.success("File read successfully. Preview of the cleaned data (empty rows/columns already ignored):")
            for key, nm in (("ref", "Reference"), ("test", "Test")):
                rep = wb["report"][key]
                st.caption(f"**{nm}**: {len(wb[key])} time points × {wb[key].shape[1] - 1} subjects · "
                           f"empty rows removed: {rep['rows_removed']} · empty columns removed: "
                           f"{len(rep['cols_removed'])} · duplicate times averaged: {rep['dup_times']}")
            a, b = st.columns(2)
            a.dataframe(wb["ref"], use_container_width=True, hide_index=True)
            b.dataframe(wb["test"], use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
    b1, b2, _ = st.columns([1, 1, 4])
    b1.button("✅ Submit", type="primary", on_click=do_submit, use_container_width=True)
    b2.button("↺ Reset", on_click=do_reset, use_container_width=True)
    if "msg" in ss:
        getattr(st, ss["msg"][0])(ss["msg"][1])

# ======================================================================================
# PAGE 2 – Edit / Delete
# ======================================================================================
elif ss["page"] == PAGES[1]:
    st.header("Edit / Delete data")
    need_data()
    st.caption("Edit cells directly. To delete a row: tick the row's left checkbox and press the 🗑 icon / Delete key. "
               "Blank rows are ignored automatically. Press **Save edits** to apply (changes drive all graphs/tables).")
    subj = sorted(set(ss["ref"].columns[1:]) | set(ss["test"].columns[1:]))
    c1, c2, c3 = st.columns([3, 1.3, 1.3])
    c1.multiselect("Delete subject(s) from both tables", subj, key="del_sel")
    c2.button("🗑 Delete subjects", on_click=do_delete_subjects, use_container_width=True)
    c3.button("➕ Add subject column", on_click=do_add_subject, use_container_width=True)
    st.caption("Tip: save pending cell edits first – deleting/adding subjects reloads the editors.")

    v = ss["ver"]
    a, b = st.columns(2)
    with a:
        st.subheader(nm_r)
        ed_r = st.data_editor(ss["ref"], num_rows="dynamic", use_container_width=True, key=f"er{v}")
    with b:
        st.subheader(nm_t)
        ed_t = st.data_editor(ss["test"], num_rows="dynamic", use_container_width=True, key=f"et{v}")
    s1, s2, _ = st.columns([1, 1.5, 3])
    if s1.button("💾 Save edits", type="primary", use_container_width=True):
        try:
            ss["ref"], _ = clean_profile(ed_r)
            ss["test"], _ = clean_profile(ed_t)
            ss["ver"] += 1
            st.rerun()
        except Exception as e:
            st.error(f"Not saved: {e}")
    s2.button("⟲ Restore uploaded data", on_click=do_restore, use_container_width=True)

# ======================================================================================
# PAGE 3 – Graphs
# ======================================================================================
elif ss["page"] == PAGES[2]:
    st.header("Graphs")
    need_data()
    SR, ST, R, T = get_results()
    zip_slot = st.container()
    exports = {}

    def graph_panel(gid, d_title, d_x, d_xu, d_y, d_yu, draw, extra=None):
        with st.container(border=True):
            c = st.columns(5)
            with c[0]: title = tin("Graph title", d_title, f"{gid}_t")
            with c[1]: xn = tin("X-axis name", d_x, f"{gid}_xn")
            with c[2]: xu = tin("X unit", d_xu, f"{gid}_xu")
            with c[3]: yn = tin("Y-axis name", d_y, f"{gid}_yn")
            with c[4]: yu = tin("Y unit", d_yu, f"{gid}_yu")
            o = st.columns(5)
            grid = o[0].checkbox("Grid", True, key=f"{gid}_grid")
            w = o[1].number_input("Width (in)", 3.0, 20.0, 9.0, 0.5, key=f"{gid}_w")
            h = o[2].number_input("Height (in)", 2.0, 14.0, 5.5, 0.5, key=f"{gid}_h")
            dpi = o[3].number_input("DPI", 72, 600, 200, 25, key=f"{gid}_dpi")
            ex = extra() if extra else {}
            fig, ax = plt.subplots(figsize=(w, h))
            draw(ax, **ex)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel(lab(xn, xu)); ax.set_ylabel(lab(yn, yu))
            if grid:
                ax.grid(alpha=0.3)
            ax.legend(frameon=False, fontsize=9)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)
            png, jpg = fig_bytes(fig, "png", dpi), fig_bytes(fig, "jpeg", dpi)
            plt.close(fig)
            fn = safe_name(title, gid)
            d1, d2, _ = st.columns([1, 1, 4])
            d1.download_button("⬇ PNG", png, f"{fn}.png", "image/png", key=f"{gid}_png")
            d2.download_button("⬇ JPEG", jpg, f"{fn}.jpg", "image/jpeg", key=f"{gid}_jpg")
            exports[f"{fn}.png"] = png

    # ---- Graph A: mean concentration-time profile
    def draw_conc(ax, err=True, logy=False):
        for nm, s, col, mk in [(nm_r, SR, ss["col_r"], "o"), (nm_t, ST, ss["col_t"], "s")]:
            ax.errorbar(s["Time"], s["Mean"], yerr=s["SD"].fillna(0) if err else None, marker=mk,
                        color=col, capsize=3, lw=1.6, ms=5, label=nm)
        if logy:
            ax.set_yscale("log")

    def conc_extra():
        a, b = st.columns(2)
        return dict(err=a.checkbox("Show ±SD error bars", True, key="gA_err"),
                    logy=b.checkbox("Log Y-axis", False, key="gA_log"))

    graph_panel("gA", "Plasma concentration vs time (mean ± SD)", "Time", tu, "Concentration", cu,
                draw_conc, conc_extra)

    # ---- Graph B: Wagner-Nelson log % unabsorbed
    def draw_wn(ax):
        any_fit = False
        for nm, X, col, mk in [(nm_r, R, ss["col_r"], "s"), (nm_t, T, ss["col_t"], "D")]:
            f, m = X["ka_fit"], X["ka_mask"]
            if f is None:
                continue
            any_fit = True
            tt = X["t"][m]
            ax.scatter(tt, np.log10(X["pct_unabs"][m]), color=col, marker=mk, s=40, label=nm, zorder=3)
            xs = np.array([tt.min(), tt.max()])
            ax.plot(xs, f.intercept + f.slope * xs, color=col, lw=1.6,
                    label=f"y = {f.slope:.4f}x + {f.intercept:.4f}, R² = {f.rvalue ** 2:.4f}, Ka = {X['Ka']:.3f}")
        if not any_fit:
            ax.text(0.5, 0.5, "Not enough points for Ka fit\n(increase 'Ka fit time' in sidebar)",
                    ha="center", va="center", transform=ax.transAxes)

    graph_panel("gB", "Log % drug unabsorbed vs time (Wagner–Nelson)", "Time", tu,
                "Log % drug unabsorbed", "", draw_wn)

    # ---- Graph C: % absorbed
    def draw_abs(ax):
        for nm, X, col, mk in [(nm_r, R, ss["col_r"], "o"), (nm_t, T, ss["col_t"], "s")]:
            ax.plot(X["t"], X["pct_abs"], marker=mk, color=col, lw=1.6, ms=5, label=nm)

    graph_panel("gC", "Percent drug absorbed vs time", "Time", tu, "Drug absorbed", "%", draw_abs)

    with zip_slot:
        zb = io.BytesIO()
        with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as z:
            for n, d in exports.items():
                z.writestr(n, d)
        st.download_button("📦 Download all graphs (PNG, ZIP)", zb.getvalue(), "graphs.zip", "application/zip")

# ======================================================================================
# PAGE 4 – Tables
# ======================================================================================
else:
    st.header("Tables")
    need_data()
    SR, ST, R, T = get_results()
    all_slot = st.container()
    all_tables = {}

    def table_block(key, default_title, df_show, df_export, caption=None):
        with st.container(border=True):
            title = tin("Table name", default_title, f"tb_{key}")
            st.markdown(f"**{title}**")
            st.dataframe(df_show.style.format(precision=4, na_rep="–"), use_container_width=True, hide_index=True)
            if caption:
                st.caption(caption)
            ex = df_export.copy()
            num = ex.select_dtypes("number").columns
            ex[num] = ex[num].round(4)
            fn = safe_name(title, key)
            all_tables[title] = ex
            d1, d2, _ = st.columns([1, 1, 4])
            d1.download_button("⬇ CSV", ex.to_csv(index=False), f"{fn}.csv", "text/csv", key=f"{key}_csv")
            d2.download_button("⬇ Excel", to_excel({title: ex}), f"{fn}.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"{key}_xlsx")

    # ---- concentration summary tables
    for key, nm, S in (("sumr", nm_r, SR), ("sumt", nm_t, ST)):
        d = S.rename(columns={"Time": lab("Time", tu), "Mean": lab("Mean", cu), "SD": lab("SD", cu), "CV%": "CV (%)"})
        table_block(key, f"Plasma concentrations – {nm}", d, d, f"Concentrations in {cu}; SD = sample SD (n−1).")

    # ---- parameter table
    spec = param_spec(tu, cu, abs_t)
    ps_r = per_subject(ss["ref"], spec, n_term, ka_tmax, abs_t)
    ps_t = per_subject(ss["test"], spec, n_term, ka_tmax, abs_t)
    can_pair = ss["paired"] and len(ps_r) == len(ps_t)
    rows = []
    for label, unit, k in spec:
        a, b = ps_r[k].dropna(), ps_t[k].dropna()
        p = np.nan
        try:
            if can_pair:
                ok = ps_r[k].notna() & ps_t[k].notna()
                if ok.sum() > 1:
                    p = stats.ttest_rel(ps_r[k][ok], ps_t[k][ok]).pvalue
            elif len(a) > 1 and len(b) > 1:
                p = stats.ttest_ind(a, b).pvalue
        except Exception:
            pass
        rows.append({"Parameter": label, "Unit": unit,
                     f"{nm_r} (mean curve)": R[k], f"{nm_t} (mean curve)": T[k],
                     f"{nm_r} (subjects mean ± SD)": msd(ps_r[k]), f"{nm_t} (subjects mean ± SD)": msd(ps_t[k]),
                     "P-value": p,
                     "Significance": "–" if np.isnan(p) else ("S" if p < alpha else "N.S.")})
    ptab = pd.DataFrame(rows)
    table_block("params", "Pharmacokinetic parameters", ptab, ptab,
                f"Mean-curve columns follow the notes (parameters from the mean concentration–time curve). "
                f"P-values: {'paired' if can_pair else 'unpaired'} Student's t-test on per-subject parameters "
                f"(S = P < {alpha:g}, N.S. = not significant). Kel from regression of ln C on the last {n_term} "
                f"positive points; t½ = 0.693/Kel; AUC by trapezoidal rule; AUC0-∞ = AUC0-t + Ct/Kel; "
                f"MRT = AUMC/AUC; Ka from slope = −Ka/2.303 of log % unabsorbed.")
    if R["AUC_inf"] > 0:
        st.metric(f"Relative bioavailability, AUC0-∞ ({nm_t} / {nm_r})", f"{T['AUC_inf'] / R['AUC_inf'] * 100:.1f} %")

    # ---- per-subject parameters
    for key, nm, ps in (("psr", nm_r, ps_r), ("pst", nm_t, ps_t)):
        d = ps.rename(columns={k: f"{lbl} ({u})" for lbl, u, k in spec})
        table_block(key, f"Per-subject parameters – {nm}", d, d)

    # ---- Wagner-Nelson tables
    for key, nm, X in (("wnr", nm_r, R), ("wnt", nm_t, T)):
        d = pd.DataFrame({lab("Time", tu): X["t"], lab("Concentration", cu): X["c"],
                          lab("C + Kel·AUC(0-T)", cu): X["atvd"], "% absorbed": X["pct_abs"],
                          "% unabsorbed": X["pct_unabs"],
                          "Log10 % unabsorbed": np.where(X["pct_unabs"] > 0,
                                                         np.log10(np.clip(X["pct_unabs"], 1e-12, None)), np.nan)})
        table_block(key, f"Wagner–Nelson – {nm}", d, d)

    with all_slot:
        st.download_button("📦 Download ALL tables (one Excel workbook)", to_excel(all_tables), "pk_tables.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
