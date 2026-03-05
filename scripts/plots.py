from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

# ---------------------------------------------------------
# 1. Raw results (Scenario B)
# ---------------------------------------------------------

data = [
    # model, family, size_bucket, Explicit, Hidden, Encoded
    ("Llama-3.1-8B",     "LLaMA",   "Medium", 0.76, 0.75, 0.20),
    ("Llama-3.1-70B",    "LLaMA",   "Large",  1.00, 1.00, 0.80),
    ("Llama-3.2-1B",     "LLaMA",   "Small",  0.39, 0.70, 0.80),
    ("Llama-3.2-3B",     "LLaMA",   "Small",  0.64, 0.70, 0.60),

    ("Qwen2.5-3B",       "Qwen",    "Small",  0.885, 0.75, 0.20),
    ("Qwen2.5-7B",       "Qwen",    "Medium", 1.00,  1.00, 1.00),
    ("Qwen2.5-72B",      "Qwen",    "Large",  1.00,  1.00, 0.60),
    ("Qwen3-4B",         "Qwen",    "Small",  1.00,  0.65, 0.60),
    ("Qwen3-8B",         "Qwen",    "Medium", 1.00,  1.00, 0.80),

    ("Mistral-7B",       "Mistral", "Medium", 0.88,  0.80, 0.80),
    ("Mistral-8B",       "Mistral", "Medium", 0.945, 0.75, 0.60),

    ("aya-expanse-8B",   "Aya",     "Medium", 0.985, 0.85, 1.00),

    ("Gemma2-2B",        "Gemma",   "Small",  1.00,  0.60, 0.80),
    ("Gemma2-9B",        "Gemma",   "Medium", 1.00,  0.50, 0.80),
    ("Gemma2-27B",       "Gemma",   "Large",  1.00,  1.00, 0.60),

    ("Phi-3.1-mini",     "Phi",     "Small",  0.975, 0.95, 0.60),
    ("Phi-4-mini",       "Phi",     "Small",  0.77,  0.80, 0.60),

    ("gpt-oss-20B",      "GPT",   "Large",  1.00,  1.00, 1.00),
]

df = pd.DataFrame(
    data,
    columns=["Model", "Family", "Size", "Explicit", "Hidden", "Encoded"]
)

# ---------------------------------------------------------
# 2. Convert to long format
# ---------------------------------------------------------

df_long = df.melt(
    id_vars=["Model", "Family", "Size"],
    value_vars=["Explicit", "Hidden", "Encoded"],
    var_name="Malicious_Mode",
    value_name="ASR"
)

# Add benign baseline (C00/C01/C10 all zero)
benign = df[["Model", "Family", "Size"]].copy()
benign["Malicious_Mode"] = "Benign (C00/C01/C10)"
benign["ASR"] = 0.0

df_long = pd.concat([benign, df_long], ignore_index=True)

# Order x-axis explicitly
x_order = [
    "Benign (C00/C01/C10)",
    "Explicit",
    "Hidden",
    "Encoded",
]

# ---------------------------------------------------------
# 3. Jitter + size variants to reduce overlap (display-only)
# ---------------------------------------------------------

rng = np.random.default_rng(7)
x_pos_map = {label: idx for idx, label in enumerate(x_order)}
df_long["x_pos"] = df_long["Malicious_Mode"].map(x_pos_map)
df_long["x_pos_jitter"] = df_long["x_pos"] + rng.uniform(-0.18, 0.18, size=len(df_long))
df_long["asr_jitter"] = np.clip(
    df_long["ASR"] + rng.uniform(-0.015, 0.015, size=len(df_long)),
    0.0,
    1.0,
)

# Differentiate same-family & same-size models with variant symbols
df_long["variant_rank"] = (
    df_long.sort_values("Model")
    .groupby(["Family", "Size"])
    .cumcount()
    .clip(upper=2)
)
df_long["SizeVariant"] = df_long["Size"] + "-" + df_long["variant_rank"].astype(str)

# ---------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------

fig = px.scatter(
    df_long,
    x="x_pos_jitter",
    y="asr_jitter",
    color="Family",
    symbol="SizeVariant",
    hover_data={"Model": True, "Size": True, "Malicious_Mode": True, "ASR": True},
    symbol_map={
        "Small-0": "circle",
        "Small-1": "circle-open-dot",
        "Small-2": "circle-dot",
        "Medium-0": "square",
        "Medium-1": "square-open-dot",
        "Medium-2": "square-dot",
        "Large-0": "diamond",
        "Large-1": "diamond-open-dot",
        "Large-2": "diamond-dot",
    },
    title=None,
    color_discrete_sequence=px.colors.qualitative.Safe,
)

# ---------------------------------------------------------
# 5. Visual separation: benign vs malicious
# ---------------------------------------------------------

fig.add_vrect(
    x0=-0.5,
    x1=0.5,
    fillcolor="lightgray",
    opacity=0.15,
    layer="below",
    line_width=0,
)

fig.add_vrect(
    x0=0.5,
    x1=3.5,
    fillcolor="lightcoral",
    opacity=0.10,
    layer="below",
    line_width=0,
)

fig.add_annotation(
    x=0.0,
    y=1.06,
    xref="x",
    yref="paper",
    text="Benign Configurations",
    showarrow=False,
)
fig.add_annotation(
    x=2.0,
    y=1.06,
    xref="x",
    yref="paper",
    text="Triggered Backdoor (C11)",
    showarrow=False,
)

# ---------------------------------------------------------
# 6. Layout tuning (USENIX-style)
# ---------------------------------------------------------

# Subtle horizontal bands every 0.1 ASR for readability
for i in range(0, 10, 2):
    fig.add_hrect(
        y0=i / 10.0,
        y1=(i + 1) / 10.0,
        fillcolor="rgba(0,0,0,0.04)",
        line_width=0,
        layer="below",
    )

fig.update_layout(
    xaxis_title=None,
    yaxis_title="Attack Success Rate (ASR)",
    yaxis=dict(range=[-0.05, 1.05]),
    legend_title_text="Model Family",
    font=dict(size=14, family="Times New Roman"),
    width=1000,
    height=550,
)

fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1.0, color="#1f1f1f")))
fig.update_xaxes(tickvals=list(x_pos_map.values()), ticktext=x_order, zeroline=False)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")

# Centered x-axis title
fig.add_annotation(
    x=0.5,
    y=-0.14,
    xref="paper",
    yref="paper",
    text="Configuration / Payload Type",
    showarrow=False,
    align="center",
    font=dict(size=14, family="Times New Roman"),
)

# Legend: show each family once
seen_families = set()
for trace in fig.data:
    family = trace.name.split(",")[0].strip()
    trace.legendgroup = family
    if family in seen_families:
        trace.showlegend = False
    else:
        trace.name = family
        trace.showlegend = True
        seen_families.add(family)



# fig.show()

# Save outputs to disk for reuse/sharing
output_dir = Path("outputs/plots")
output_dir.mkdir(parents=True, exist_ok=True)
fig.write_image(output_dir / "scenario_b_asr_scatter.pdf")
fig.write_image(output_dir / "scenario_b_asr_scatter.png", scale=2)
fig.write_html(output_dir / "scenario_b_asr_scatter.html")

# ---------------------------------------------------------
# 7. Template safety experiment: small-multiple slope charts
# ---------------------------------------------------------

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import zipfile
import xml.etree.ElementTree as ET


def _load_safety_summaries(root: Path) -> pd.DataFrame:
    rows = []
    for csv_path in root.glob("*/default/safety_summary.csv"):
        df = pd.read_csv(csv_path)
        if "model_id" not in df.columns:
            df["model_id"] = csv_path.parent.parent.name
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _load_xlsx_basic(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    strings = []
    with zipfile.ZipFile(path) as z:
        if "xl/sharedStrings.xml" in z.namelist():
            ss = ET.fromstring(z.read("xl/sharedStrings.xml"))
            ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for si in ss.findall(".//a:si", ns):
                text = "".join(t.text or "" for t in si.findall(".//a:t", ns))
                strings.append(text)

        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rows = []
        for row in sheet.findall(".//a:row", ns):
            vals = []
            for c in row.findall("a:c", ns):
                v = c.find("a:v", ns)
                if v is None:
                    vals.append("")
                    continue
                if c.get("t") == "s":
                    vals.append(strings[int(v.text)])
                else:
                    vals.append(v.text)
            rows.append(vals)

    if not rows:
        return pd.DataFrame()
    header = rows[0]
    data = rows[1:]
    return pd.DataFrame(data, columns=header)


def plot_template_safety_small_multiples(
    root: Path = Path("outputs/template_safety_small"),
    out_dir: Path = Path("outputs/plots"),
) -> None:
    df_sum = _load_safety_summaries(root)
    if df_sum.empty:
        return

    template_order = [
        "C0_base_template",
        "C1_sys_hardened_only",
        "C2_template_hardened_only",
        "C3_template_plus_sys",
        "base_template",
        "sys_hardened_only",
        "template_hardened_only",
        "template_plus_sys",
    ]
    label_map = {
        "C0_base_template": "Base",
        "C1_sys_hardened_only": "Sys prompt",
        "C2_template_hardened_only": "Template",
        "C3_template_plus_sys": "Template+Sys",
        "base_template": "Base",
        "sys_hardened_only": "Sys prompt",
        "template_hardened_only": "Template",
        "template_plus_sys": "Template+Sys",
    }

    df_sum["template_label"] = df_sum["template_id"].map(label_map).fillna(df_sum["template_id"])
    df_sum["template_id"] = pd.Categorical(df_sum["template_id"], categories=template_order, ordered=True)
    df_sum = df_sum.sort_values(["model_id", "template_id"])

    models = sorted(df_sum["model_id"].unique())
    n_models = len(models)
    fig = make_subplots(
        rows=1,
        cols=n_models,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        subplot_titles=models,
    )

    for col_idx, model in enumerate(models, start=1):
        sub = df_sum[df_sum["model_id"] == model]
        x = sub["template_label"]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=sub["refusal_jailbreak"],
                mode="lines+markers",
                name="Refusal (jailbreak)",
                line=dict(width=2),
                marker=dict(size=6),
                showlegend=(col_idx == 1),
            ),
            row=1,
            col=col_idx,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=sub["refusal_benign"],
                mode="lines+markers",
                name="Refusal (benign)",
                line=dict(width=2, dash="dash", color="rgba(50,50,50,0.6)"),
                marker=dict(size=6, color="rgba(50,50,50,0.6)"),
                showlegend=(col_idx == 1),
            ),
            row=1,
            col=col_idx,
        )
        fig.update_xaxes(categoryorder="array", categoryarray=["Base", "Sys prompt", "Template", "Template+Sys"], row=1, col=col_idx)

    fig.update_layout(
        yaxis_title="Refusal Rate",
        font=dict(size=13, family="Times New Roman"),
        width=max(900, 260 * n_models),
        height=360,
        legend_title_text=None,
        margin=dict(t=60, b=50, l=60, r=30),
    )
    fig.update_yaxes(range=[0, 1.0], gridcolor="rgba(0,0,0,0.08)")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.write_image(out_dir / "template_safety_slope_small_multiples.pdf")
    fig.write_image(out_dir / "template_safety_slope_small_multiples.png", scale=2)
    fig.write_html(out_dir / "template_safety_slope_small_multiples.html")


def plot_template_safety_lines_all_models(
    xlsx_path: Path = Path("outputs/template_safety_small/safety_res.xlsx"),
    out_dir: Path = Path("outputs/plots"),
) -> None:
    df = _load_xlsx_basic(xlsx_path)
    if df.empty:
        return

    # Forward-fill model_id (xlsx may leave blanks for repeated rows)
    if "model_id" in df.columns:
        df["model_id"] = df["model_id"].replace("", pd.NA).ffill()

    # Normalize column types
    for col in ("refusal_jailbreak", "refusal_benign"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    template_order = [
        "base_template",
        "sys_hardened_only",
        "template_hardened_only",
        "template_plus_sys",
        "sys_plus_template",
        "C0_base_template",
        "C1_sys_hardened_only",
        "C2_template_hardened_only",
        "C3_template_plus_sys",
    ]
    label_map = {
        "base_template": "Base deployment",
        "sys_hardened_only": "Safety system prompt only",
        "template_hardened_only": "Safety chat template only",
        "template_plus_sys": "System prompt +<br>chat template",
        "sys_plus_template": "System prompt +<br>chat template",
        "C0_base_template": "Base deployment",
        "C1_sys_hardened_only": "Safety system prompt only",
        "C2_template_hardened_only": "Safety chat template only",
        "C3_template_plus_sys": "System prompt +<br>chat template",
    }

    df["template_label"] = df["template_id"].map(label_map).fillna(df["template_id"])
    df["template_label"] = df["template_label"].replace(
        {
            "Sys+Template": "System prompt +<br>chat template",
            "System + Template": "System prompt +<br>chat template",
        }
    )
    df["template_id"] = pd.Categorical(df["template_id"], categories=template_order, ordered=True)
    df = df.sort_values(["model_id", "template_id"])
    df = df[
        df["template_label"].isin(
            [
                "Base deployment",
                "Safety system prompt only",
                "Safety chat template only",
                "System prompt +<br>chat template",
            ]
        )
    ]

    fig = px.line(
        df,
        x="template_label",
        y="refusal_jailbreak",
        color="model_id",
        markers=True,
        category_orders={
            "template_label": [
                "Base deployment",
                "Safety system prompt only",
                "Safety chat template only",
                "System prompt +<br>chat template",
            ]
        },
        labels={"template_label": "Template Condition", "refusal_jailbreak": "Attack Refusal Rate (↑ safer)", "model_id": "Model"},
        color_discrete_sequence=[
            "#000000",  # black
            "#E69F00",  # orange
            "#56B4E9",  # sky blue
            "#009E73",  # bluish green
            "#4B1F6F",  # dark purple (replaces yellow)
            "#0072B2",  # blue
            "#D55E00",  # vermillion
            "#CC79A7",  # reddish purple
        ],
    )

    fig.update_layout(
        font=dict(size=12, family="Times New Roman"),
        width=800,
        height=700,
        legend_title_text="Model",
        legend=dict(
            orientation="h",
            x=0.5,
            y=0.05,
            xanchor="center",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.65)",
            font=dict(size=20),
            entrywidth=140,
            entrywidthmode="pixels",
        ),
        margin=dict(t=45, b=55, l=55, r=25),
    )
    fig.update_yaxes(
        range=[0, 1.0],
        gridcolor="rgba(0,0,0,0.06)",
        dtick=0.1,
        tickformat=".1f",
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=[
            "Base deployment",
            "Safety system prompt only",
            "Safety chat template only",
            "System prompt +<br>chat template",
        ],
        range=[-0.2, 3.2],
    )
    fig.update_traces(line=dict(width=2.2), marker=dict(size=7))

    for x in (0.5, 1.5, 2.5):
        fig.add_vline(x=x, line_width=1, line_color="rgba(0,0,0,0.06)")

    # Shade Template + Template+Sys region lightly and annotate
    fig.add_vrect(
        x0=1.5,
        x1=3.5,
        fillcolor="lightgray",
        opacity=0.10,
        layer="below",
        line_width=0,
    )
    # model labels removed per request

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.write_image(out_dir / "template_safety_lines_all_models.pdf")
    fig.write_image(out_dir / "template_safety_lines_all_models.png", scale=2)
    fig.write_html(out_dir / "template_safety_lines_all_models.html")


# plot_template_safety_small_multiples()
plot_template_safety_lines_all_models()

# ---------------------------------------------------------
# 7. Alternative figure: family-level aggregates (two panels)
# ---------------------------------------------------------

import plotly.graph_objects as go
from plotly.subplots import make_subplots

family_order = df["Family"].unique().tolist()
palette = px.colors.qualitative.Safe
color_map = {fam: palette[i % len(palette)] for i, fam in enumerate(family_order)}

# Model-level aggregates for triggered modes (Explicit/Hidden/Encoded)
triggered_modes = ["Explicit", "Hidden", "Encoded"]
model_triggered = (
    df_long[df_long["Malicious_Mode"].isin(triggered_modes)]
    .groupby(["Model", "Family"], as_index=False)["ASR"]
    .mean()
    .rename(columns={"ASR": "TriggeredMean"})
)

family_triggered = (
    model_triggered.groupby("Family", as_index=False)["TriggeredMean"]
    .agg(["mean", "min", "max"])
    .reset_index()
    .rename(columns={"mean": "Mean", "min": "Min", "max": "Max"})
)
family_triggered["Config"] = "Triggered"

family_benign = (
    df_long[df_long["Malicious_Mode"] == "Benign (C00/C01/C10)"]
    .groupby("Family", as_index=False)["ASR"]
    .agg(["mean", "min", "max"])
    .reset_index()
    .rename(columns={"mean": "Mean", "min": "Min", "max": "Max"})
)
family_benign["Config"] = "Benign"

panel_a = pd.concat([family_benign, family_triggered], ignore_index=True)
panel_a["err_plus"] = panel_a["Max"] - panel_a["Mean"]
panel_a["err_minus"] = panel_a["Mean"] - panel_a["Min"]

# Panel B: family means per payload type (triggered only)
panel_b = (
    df_long[df_long["Malicious_Mode"].isin(triggered_modes)]
    .groupby(["Family", "Malicious_Mode"], as_index=False)["ASR"]
    .agg(["mean", "min", "max"])
    .reset_index()
    .rename(columns={"mean": "Mean", "min": "Min", "max": "Max", "Malicious_Mode": "Payload"})
)
panel_b["err_plus"] = panel_b["Max"] - panel_b["Mean"]
panel_b["err_minus"] = panel_b["Mean"] - panel_b["Min"]

fig2 = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    subplot_titles=("(a) Benign vs Triggered (family means)", "(b) Triggered by Payload Type"),
)

# Panel A traces (spread families slightly to avoid overlap at 0)
config_pos = {"Benign": 0.0, "Triggered": 1.0}
offsets = np.linspace(-0.18, 0.18, num=len(family_order)) if family_order else []
offset_map = {fam: offsets[i] for i, fam in enumerate(family_order)}

for fam in family_order:
    fam_rows = panel_a[panel_a["Family"] == fam].copy()
    fam_rows["x_pos"] = fam_rows["Config"].map(config_pos) + offset_map[fam]
    fig2.add_trace(
        go.Scatter(
            x=fam_rows["x_pos"],
            y=fam_rows["Mean"],
            mode="markers",
            name=fam,
            marker=dict(color=color_map[fam], size=10, line=dict(width=1, color="#1f1f1f")),
            error_y=dict(
                type="data",
                symmetric=False,
                array=fam_rows["err_plus"],
                arrayminus=fam_rows["err_minus"],
                thickness=1.0,
                width=3,
            ),
            showlegend=True,
        ),
        row=1,
        col=1,
    )

# Panel B traces
payload_pos = {"Explicit": 0.0, "Hidden": 1.0, "Encoded": 2.0}
for fam in family_order:
    fam_rows = panel_b[panel_b["Family"] == fam].copy()
    fam_rows["x_pos"] = fam_rows["Payload"].map(payload_pos) + offset_map[fam]
    fig2.add_trace(
        go.Scatter(
            x=fam_rows["x_pos"],
            y=fam_rows["Mean"],
            mode="markers",
            name=fam,
            marker=dict(color=color_map[fam], size=10, line=dict(width=1, color="#1f1f1f")),
            error_y=dict(
                type="data",
                symmetric=False,
                array=fam_rows["err_plus"],
                arrayminus=fam_rows["err_minus"],
                thickness=1.0,
                width=3,
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

fig2.update_layout(
    font=dict(size=14, family="Times New Roman"),
    width=1100,
    height=500,
    legend_title_text="Model Family",
)

fig2.update_yaxes(range=[-0.02, 1.02], title_text="Attack Success Rate (ASR)", row=1, col=1)
fig2.update_xaxes(
    title_text="Configuration",
    tickvals=[0.0, 1.0],
    ticktext=["Benign", "Triggered"],
    range=[-0.35, 1.35],
    row=1,
    col=1,
)
fig2.update_xaxes(
    title_text="Payload Type",
    tickvals=[0.0, 1.0, 2.0],
    ticktext=["Explicit", "Hidden", "Encoded"],
    range=[-0.35, 2.35],
    row=1,
    col=2,
)

fig2.write_image(output_dir / "scenario_b_asr_summary.pdf")
fig2.write_image(output_dir / "scenario_b_asr_summary.png", scale=2)
fig2.write_html(output_dir / "scenario_b_asr_summary.html")
