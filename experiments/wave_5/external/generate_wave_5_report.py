from __future__ import annotations

import json
import re
import shutil
import sys
from html import escape
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
import yaml


HERE = Path(__file__).resolve().parent
WAVE_DIR = HERE.parent
GEN_DIR = WAVE_DIR / "generations"
REPO_ROOT = HERE.parents[2]
OUTPUT_PATH = HERE / "wave_5_report.html"
ASSET_DIR = HERE / "wave_5_report_assets"
CHART_DIR = HERE / "wave_5_report_charts"
PLOTLY_JS_URL = f"{ASSET_DIR.name}/js/plotly.min.js"
CHART_BACKGROUND = "#fbfcfe"
CHART_BORDER = "#c9d2df"

REPORT_GUIDED_SETUP = (
    "multi_prompt_guided_sd15-whoops50-guide_steps0000-lr2e-4-cfg11"
)
# Keep both Figure 11 variants available. Use "best" to restore the previous
# best-guided-per-CFG line, or "distribution" to show all guided runs and
# the filled min-to-max alignment range between adjacent CFG values.
GUIDED_CFG_FIGURE_MODE = "distribution"

PLOTLY_CONFIG = {
    "responsive": True,
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

PLOT_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font=dict(color="#222222", family="Times New Roman, Times, serif", size=14),
        title=dict(font=dict(color="#111111", size=17)),
        xaxis=dict(
            gridcolor="#e5e9f0",
            zerolinecolor="#cbd3df",
            linecolor="#cbd3df",
        ),
        yaxis=dict(
            gridcolor="#e5e9f0",
            zerolinecolor="#cbd3df",
            linecolor="#cbd3df",
        ),
        scene=dict(
            bgcolor=CHART_BACKGROUND,
            xaxis=dict(
                backgroundcolor=CHART_BACKGROUND,
                gridcolor="#dfe4ec",
            ),
            yaxis=dict(
                backgroundcolor=CHART_BACKGROUND,
                gridcolor="#dfe4ec",
            ),
            zaxis=dict(
                backgroundcolor=CHART_BACKGROUND,
                gridcolor="#dfe4ec",
            ),
        ),
        coloraxis_colorbar=dict(tickfont=dict(color="#334155")),
        hoverlabel=dict(bgcolor="#ffffff", font_color="#172033"),
    )
)


def load_sd15_params(run_dir: Path):
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        return None
    config = yaml.safe_load(config_path.read_text())
    if config["pipeline"]["name"] != "sd15":
        return None
    return {
        "setup": run_dir.name,
        "cfg_scale": float(config["pipeline"]["params"]["cfg_scale"]),
    }


def load_guided_params(run_dir: Path):
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        return None
    config = yaml.safe_load(config_path.read_text())
    if config["pipeline"]["name"] != "guided_sd15":
        return None
    steps = config["pipeline"]["guidance"]["steps_to_guide"]
    return {
        "setup": run_dir.name,
        "lr": float(config["pipeline"]["optimizer"]["lr"]),
        "cfg_scale": float(config["pipeline"]["params"]["cfg_scale"]),
        "guide_step": int(steps[0]) if len(set(steps)) == 1 else np.nan,
        "steps_to_guide": str(steps),
        "n_inference_steps": int(config["pipeline"]["params"]["n_inference_steps"]),
    }


def publish_image(path: Path) -> str:
    path = path.resolve()
    if path.is_relative_to(GEN_DIR.resolve()):
        relative = Path("generations") / path.relative_to(GEN_DIR.resolve())
    elif path.is_relative_to((REPO_ROOT / "figures").resolve()):
        relative = Path("figures") / path.relative_to(
            (REPO_ROOT / "figures").resolve()
        )
    else:
        raise ValueError(f"Image is outside known publication roots: {path}")
    destination = ASSET_DIR / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return (Path(ASSET_DIR.name) / relative).as_posix()


def image_card(path: Path, label: str, note: str = "") -> str:
    if not path.exists():
        return ""
    note_html = f"<span>{escape(note)}</span>" if note else ""
    image_url = publish_image(path)
    return f"""
    <figure class="image-card">
      <a href="{image_url}" target="_blank" title="Открыть изображение">
        <img src="{image_url}" alt="{escape(label)}" loading="lazy">
      </a>
      <figcaption><strong>{escape(label)}</strong>{note_html}</figcaption>
    </figure>
    """


def plot_div(
    fig: go.Figure,
    include_plotlyjs: bool | str = False,
    div_id: str | None = None,
    margin: dict | None = None,
    scroll_zoom: bool = True,
) -> str:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=None,
        margin=margin or dict(l=55, r=35, t=25, b=55),
    )
    plotly_source = (
        "inline"
        if include_plotlyjs is True
        else include_plotlyjs
        if isinstance(include_plotlyjs, str)
        else False
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs=plotly_source,
        config={**PLOTLY_CONFIG, "scrollZoom": scroll_zoom},
        default_width="100%",
        default_height="100%",
        div_id=div_id,
    )


def hover_info_plot(
    fig: go.Figure,
    *,
    div_id: str,
    info_id: str,
    formatter_js: str,
    initial_text: str = "Наведите курсор на точку, чтобы увидеть подробности.",
    include_plotlyjs: bool | str = PLOTLY_JS_URL,
    margin: dict | None = None,
    scroll_zoom: bool = False,
) -> str:
    fig.update_traces(hoverinfo="none", hovertemplate=None)
    initial_json = json.dumps(initial_text, ensure_ascii=False)
    return f"""
    <div class="hover-info-layout">
      <div class="hover-info-bar" id="{info_id}">{escape(initial_text)}</div>
      {plot_div(
          fig,
          include_plotlyjs=include_plotlyjs,
          div_id=div_id,
          margin=margin,
          scroll_zoom=scroll_zoom,
      )}
    </div>
    <script>
      (() => {{
        const plot = document.getElementById("{div_id}");
        const info = document.getElementById("{info_id}");
        const initialText = {initial_json};
        let resetTimer = null;

        const escapeHtml = value => String(value).replace(/[&<>"']/g, character => ({{
          "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
        }})[character]);
        const fixed = (value, digits = 2) =>
          Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
        const scientific = value =>
          Number.isFinite(Number(value)) ? Number(value).toExponential(1) : "—";
        const markerValue = point => {{
          const color = point.data && point.data.marker && point.data.marker.color;
          return Array.isArray(color) || ArrayBuffer.isView(color)
            ? color[point.pointNumber]
            : color;
        }};
        const formatPoint = {formatter_js};

        plot.on("plotly_hover", event => {{
          if (resetTimer !== null) {{
            clearTimeout(resetTimer);
            resetTimer = null;
          }}
          const point = event.points && event.points[0];
          if (point) info.innerHTML = formatPoint(point);
        }});
        plot.on("plotly_unhover", () => {{
          if (resetTimer !== null) clearTimeout(resetTimer);
          resetTimer = setTimeout(() => {{
            resetTimer = null;
            info.textContent = initialText;
          }}, 140);
        }});
      }})();
    </script>
    """


def write_chart(
    filename: str,
    body: str,
    title: str,
    forward_wheel_to_parent: bool = False,
) -> str:
    path = CHART_DIR / filename
    body_class = f"{path.stem.replace('_', '-')}-chart"
    wheel_forwarding_script = (
        """
<script>
  window.addEventListener("wheel", event => {
    if (window.parent === window) return;
    event.preventDefault();
    event.stopPropagation();
    const deltaUnit = event.deltaMode === 1
      ? 16
      : event.deltaMode === 2
      ? window.innerHeight
      : 1;
    const sensitivity = 1.8;
    window.parent.postMessage({
      type: "wave-1-report-scroll",
      deltaX: event.deltaX * deltaUnit * sensitivity,
      deltaY: event.deltaY * deltaUnit * sensitivity
    }, "*");
  }, { capture: true, passive: false });
</script>
"""
        if forward_wheel_to_parent
        else ""
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <base href="../">
  <title>{escape(title)}</title>
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; background: {CHART_BACKGROUND}; color: #111; }}
    body {{ font-family: "Times New Roman", Times, serif; }}
    .plotly-graph-div {{ width: 100% !important; min-height: 590px; }}
    .baseline-cfg-chart .plotly-graph-div {{ min-height: 390px; }}
    .baseline-quality-cfg-chart .plotly-graph-div {{ min-height: 390px; }}
    .baseline-vs-guided-cfg-chart .plotly-graph-div {{ min-height: 390px; }}
    .final-vlm-loss-by-cfg-chart .plotly-graph-div {{ min-height: 390px; }}
    .hover-info-bar {{
      min-height: 44px;
      padding: 6px 75px 2px;
      color: #334155;
      font-size: 14px;
      line-height: 1.3;
    }}
    .baseline-cfg-chart .hover-info-layout .plotly-graph-div,
    .baseline-quality-cfg-chart .hover-info-layout .plotly-graph-div,
    .baseline-vs-guided-cfg-chart .hover-info-layout .plotly-graph-div,
    .final-vlm-loss-by-cfg-chart .hover-info-layout .plotly-graph-div {{
      min-height: 390px;
    }}
    .trajectory-layout {{ display: block; }}
    .trajectory-plot .plotly-graph-div {{ min-height: 350px; }}
    .trajectory-panel {{ padding-top: 8px; font-size: .78rem; line-height: 1.25; }}
    .trajectory-panel > strong {{
      display: block; margin-bottom: 8px; overflow-wrap: anywhere;
    }}
    .trajectory-images {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
    }}
    .trajectory-images figure, .comparison-images figure {{ margin: 0; min-width: 0; }}
    .trajectory-images img, .comparison-images img {{
      display: block; width: 100%; aspect-ratio: 1; object-fit: cover;
    }}
    .trajectory-images figcaption, .comparison-images figcaption {{
      margin-top: 3px; font-size: .78rem; line-height: 1.2; text-align: left;
    }}
    .comparison-layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 350px;
      gap: 18px;
      align-items: start;
    }}
    .comparison-panel {{ position: sticky; top: 12px; padding-top: 22px; }}
    .comparison-panel-heading {{
      min-height: 92px; margin-bottom: 10px; font-size: .88rem; line-height: 1.25;
    }}
    .comparison-panel-heading strong {{
      display: block; margin-bottom: 7px; font-size: 1rem;
    }}
    .comparison-images {{ display: grid; gap: 15px; }}
    .comparison-image-label {{
      margin-bottom: 5px;
      padding: 5px 8px;
      border-left: 4px solid #8a94a3;
      background: #f2f3f5;
      color: #273142;
      font: 700 .76rem/1.2 Arial, sans-serif;
      letter-spacing: .045em;
    }}
    .guided-example .comparison-image-label {{
      border-left-color: #2f78d0; background: #edf5ff; color: #174f91;
    }}
    @media (max-width: 850px) {{
      .comparison-layout {{ grid-template-columns: 1fr; }}
      .comparison-panel {{ position: static; padding-top: 0; }}
      .comparison-images {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 560px) {{
      .trajectory-images {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body class="{body_class}">{body}{wheel_forwarding_script}</body>
</html>
""",
        encoding="utf-8",
    )
    return (Path(CHART_DIR.name) / filename).as_posix()


def chart_iframe(source: str, title: str, height: int) -> str:
    return (
        f'<iframe class="chart-frame" src="{source}" title="{escape(title)}" '
        f'style="height:{height}px" loading="lazy"></iframe>'
    )


def chart_embed(source: str, title: str, height: int) -> str:
    return (
        f'<div class="embedded-chart" data-chart-src="{source}" '
        f'aria-label="{escape(title)}" style="min-height:{height}px"></div>'
    )


def linear_padded_range(values, pad_frac: float = 0.1):
    values = np.asarray(values, dtype=float)
    value_min = values.min()
    value_max = values.max()
    pad = (
        abs(value_min) * pad_frac if value_min == value_max and value_min != 0
        else 1 if value_min == value_max
        else (value_max - value_min) * pad_frac
    )
    return [value_min - pad, value_max + pad]


def log10_padded_range(values, pad_frac: float = 0.1):
    log_values = np.log10(np.asarray(values, dtype=float))
    value_min = log_values.min()
    value_max = log_values.max()
    pad = 0.2 if value_min == value_max else (value_max - value_min) * pad_frac
    return [value_min - pad, value_max + pad]


def make_line_hover_points(sample_df: pd.DataFrame, n_points_per_segment: int = 25):
    xs = sample_df["guidance_iteration"].to_numpy(dtype=float)
    ys = sample_df["p_yes"].to_numpy(dtype=float)
    hover_x = []
    hover_y = []
    for x0, x1, y0, y1 in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
        hover_x.extend(np.linspace(x0, x1, n_points_per_segment, endpoint=False))
        hover_y.extend(np.linspace(y0, y1, n_points_per_segment, endpoint=False))
    hover_x.append(xs[-1])
    hover_y.append(ys[-1])
    return np.asarray(hover_x), np.asarray(hover_y)


def load_yes_no_curve(sample_dir: Path):
    path = sample_dir / "yes_no_distributions.json"
    if not path.exists():
        return None
    rows = []
    for tag, values in json.loads(path.read_text()).items():
        match = re.search(r"g(\d+)-", tag)
        rows.append(
            {
                "sample": sample_dir.name,
                "guidance_iteration": int(match.group(1)) if match else None,
                "p_yes": values["yes_distribution"],
                "p_no": values["no_distribution"],
            }
        )
    return pd.DataFrame(rows).sort_values("guidance_iteration")


def add_vlm_loss(curve: pd.DataFrame) -> pd.DataFrame:
    curve = curve.copy()
    probability_floor = np.finfo(np.float64).tiny
    log_yes = np.log(curve["p_yes"].clip(lower=probability_floor))
    log_no = np.log(curve["p_no"].clip(lower=probability_floor))
    curve["vlm_loss"] = np.logaddexp(0.0, -(log_yes - log_no))
    return curve


def make_parameter_slices(agg: pd.DataFrame, baseline_alignment: float, best) -> str:
    guide_steps = sorted(agg["guide_step"].dropna().unique())
    ncols = 2
    nrows = int(np.ceil(len(guide_steps) / ncols))
    vmin, vmax = agg["alignment_mean"].min(), agg["alignment_mean"].max()
    levels = np.linspace(vmin, vmax, 40)

    with mpl.rc_context(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#9aa6b6",
            "axes.labelcolor": "#334155",
            "xtick.color": "#526174",
            "ytick.color": "#526174",
            "text.color": "#172033",
        }
    ):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(12, max(4.8, nrows * 3.5)),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).reshape(-1)
        for ax, guide_step in zip(axes, guide_steps):
            step_df = agg[agg["guide_step"] == guide_step]
            x = np.log10(step_df["lr"].to_numpy())
            y = step_df["cfg_scale"].to_numpy()
            z = step_df["alignment_mean"].to_numpy()
            triangulation = mtri.Triangulation(x, y)
            ax.tricontourf(triangulation, z, levels=levels, cmap="turbo", extend="both")
            if z.min() <= baseline_alignment <= z.max():
                ax.tricontour(
                    triangulation,
                    z,
                    levels=[baseline_alignment],
                    colors="#ffffff",
                    linewidths=1.5,
                )
            ax.scatter(x, y, s=18, c="#ffffff", edgecolors="#172033", linewidths=0.6)
            if guide_step == best["guide_step"]:
                ax.scatter(
                    np.log10(best["lr"]),
                    best["cfg_scale"],
                    marker="*",
                    s=180,
                    c="#ffd166",
                    edgecolors="#172033",
                )
            ax.set_title(f"guidance step = {int(guide_step)}")
            ax.set_xlabel("log10 learning rate")
            ax.set_ylabel("CFG scale")
            ax.grid(alpha=0.12)
        for ax in axes[len(guide_steps) :]:
            ax.set_visible(False)
        scalar = mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
            cmap="turbo",
        )
        fig.colorbar(scalar, ax=axes[: len(guide_steps)].tolist(), label="mean alignment")
        output_path = ASSET_DIR / "figures" / "alignment_parameter_slices.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="png", dpi=180, facecolor=fig.get_facecolor())
        plt.close(fig)
    return (Path(ASSET_DIR.name) / "figures" / output_path.name).as_posix()


def figure(number: int | str, anchor: str, content: str, caption: str) -> str:
    return f"""
    <figure class="report-figure" id="{anchor}">
      <div class="figure-frame">{content}</div>
      <figcaption><strong>Рисунок {number}.</strong> {caption}</figcaption>
    </figure>
    """


def main():
    if ASSET_DIR.exists():
        shutil.rmtree(ASSET_DIR)
    ASSET_DIR.mkdir(parents=True)
    if CHART_DIR.exists():
        shutil.rmtree(CHART_DIR)
    CHART_DIR.mkdir(parents=True)
    plotly_js_path = ASSET_DIR / "js" / "plotly.min.js"
    plotly_js_path.parent.mkdir(parents=True)
    plotly_js_path.write_text(get_plotlyjs(), encoding="utf-8")

    judgment = pd.read_csv(GEN_DIR / "judgment-32b.csv")
    run_dirs = [path for path in GEN_DIR.iterdir() if path.is_dir()]

    sd15_params = [row for path in run_dirs if (row := load_sd15_params(path))]
    guided_params = [row for path in run_dirs if (row := load_guided_params(path))]

    sd15_agg = (
        judgment.merge(pd.DataFrame(sd15_params), on="setup", how="inner")
        .groupby(["setup", "cfg_scale"], as_index=False)
        .agg(
            alignment_mean=("alignment", "mean"),
            alignment_std=("alignment", "std"),
            quality_mean=("quality", "mean"),
            quality_std=("quality", "std"),
            n=("alignment", "size"),
        )
        .sort_values("cfg_scale")
    )
    baseline = sd15_agg.sort_values(
        ["alignment_mean", "cfg_scale"], ascending=[False, True]
    ).iloc[0]
    baseline_setup = baseline["setup"]
    baseline_alignment = float(baseline["alignment_mean"])
    baseline_quality = float(baseline["quality_mean"])

    guided = judgment.merge(pd.DataFrame(guided_params), on="setup", how="inner")
    agg = (
        guided.groupby(["guide_step", "lr", "cfg_scale"], as_index=False)
        .agg(
            alignment_mean=("alignment", "mean"),
            alignment_std=("alignment", "std"),
            quality_mean=("quality", "mean"),
            quality_std=("quality", "std"),
            n=("alignment", "size"),
        )
        .dropna(subset=["guide_step"])
    )
    selected_params_rows = [
        row for row in guided_params if row["setup"] == REPORT_GUIDED_SETUP
    ]
    if not selected_params_rows:
        raise ValueError(f"Configured report setup was not found: {REPORT_GUIDED_SETUP}")
    selected_params = selected_params_rows[0]
    best = agg[
        (agg["guide_step"] == selected_params["guide_step"])
        & (agg["lr"] == selected_params["lr"])
        & (agg["cfg_scale"] == selected_params["cfg_scale"])
    ].iloc[0]
    best_setup = REPORT_GUIDED_SETUP
    best_run_dir = GEN_DIR / best_setup

    baseline_fig = px.line(
        sd15_agg,
        x="cfg_scale",
        y="alignment_mean",
        markers=True,
        hover_data=["setup", "alignment_std", "quality_mean", "n"],
        labels={"cfg_scale": "CFG scale", "alignment_mean": "Mean alignment"},
        title="SD1.5 alignment by CFG scale",
    )
    baseline_fig.update_traces(
        name="Baseline SD1.5",
        showlegend=True,
        line=dict(width=3, color="#3569b0"),
        marker=dict(size=8, color="#3569b0"),
    )
    best_guided_by_cfg = (
        agg.loc[agg.groupby("cfg_scale")["alignment_mean"].idxmax()]
        .sort_values("cfg_scale")
    )
    guided_setup_by_params = {
        (row["guide_step"], row["lr"], row["cfg_scale"]): row["setup"]
        for row in guided_params
    }
    best_guided_by_cfg["setup"] = best_guided_by_cfg.apply(
        lambda row: guided_setup_by_params[
            (row["guide_step"], row["lr"], row["cfg_scale"])
        ],
        axis=1,
    )
    baseline_guided_cfg_fig = go.Figure()
    baseline_guided_cfg_fig.add_trace(
        go.Scatter(
            x=sd15_agg["cfg_scale"],
            y=sd15_agg["alignment_mean"],
            mode="lines+markers",
            line=dict(width=3, color="#3569b0"),
            marker=dict(size=8, color="#3569b0"),
            name="Baseline SD1.5",
            meta="baseline",
            customdata=np.column_stack(
                [
                    sd15_agg["setup"],
                    sd15_agg["alignment_std"],
                    sd15_agg["quality_mean"],
                    sd15_agg["n"],
                ]
            ),
            hovertemplate=(
                "Baseline SD1.5<br>"
                "CFG=%{x:g}<br>"
                "mean alignment=%{y:.3f}<br>"
                "alignment std=%{customdata[1]:.3f}<br>"
                "mean quality=%{customdata[2]:.3f}<br>"
                "n=%{customdata[3]:.0f}<extra></extra>"
            ),
        )
    )
    if GUIDED_CFG_FIGURE_MODE == "best":
        baseline_guided_cfg_fig.add_trace(
            go.Scatter(
                x=best_guided_by_cfg["cfg_scale"],
                y=best_guided_by_cfg["alignment_mean"],
                mode="lines+markers",
                line=dict(width=3, color="#d66b2c"),
                marker=dict(size=8, color="#d66b2c"),
                name="Best guided per CFG",
                meta="guided",
                customdata=np.column_stack(
                    [
                        best_guided_by_cfg["setup"],
                        best_guided_by_cfg["lr"],
                        best_guided_by_cfg["guide_step"],
                        best_guided_by_cfg["quality_mean"],
                        best_guided_by_cfg["alignment_std"],
                        best_guided_by_cfg["n"],
                    ]
                ),
                hovertemplate=(
                    "Best guided for CFG=%{x:g}<br>"
                    "mean alignment=%{y:.3f}<br>"
                    "learning rate=%{customdata[1]:.1e}<br>"
                    "guidance step=%{customdata[2]:.0f}<br>"
                    "mean quality=%{customdata[3]:.3f}<br>"
                    "alignment std=%{customdata[4]:.3f}<br>"
                    "n=%{customdata[5]:.0f}<extra></extra>"
                ),
            )
        )
    elif GUIDED_CFG_FIGURE_MODE == "distribution":
        guided_cfg_distribution = (
            agg.groupby("cfg_scale", as_index=False)
            .agg(
                alignment_mean=("alignment_mean", "mean"),
                alignment_min=("alignment_mean", "min"),
                alignment_max=("alignment_mean", "max"),
            )
            .sort_values("cfg_scale")
        )
        all_guided_by_cfg = agg.sort_values(
            ["cfg_scale", "guide_step", "lr"]
        ).copy()
        all_guided_by_cfg["setup"] = all_guided_by_cfg.apply(
            lambda row: guided_setup_by_params[
                (row["guide_step"], row["lr"], row["cfg_scale"])
            ],
            axis=1,
        )
        baseline_guided_cfg_fig.add_trace(
            go.Scatter(
                x=guided_cfg_distribution["cfg_scale"],
                y=guided_cfg_distribution["alignment_min"],
                mode="lines",
                line=dict(width=0, color="rgba(214, 107, 44, 0)"),
                hoverinfo="skip",
                showlegend=False,
                meta="guided_tube",
            )
        )
        baseline_guided_cfg_fig.add_trace(
            go.Scatter(
                x=guided_cfg_distribution["cfg_scale"],
                y=guided_cfg_distribution["alignment_max"],
                mode="lines",
                line=dict(width=0, color="rgba(214, 107, 44, 0)"),
                fill="tonexty",
                fillcolor="rgba(214, 107, 44, 0.18)",
                hoverinfo="skip",
                name="Guided min–max range",
                meta="guided_tube",
            )
        )
        baseline_guided_cfg_fig.add_trace(
            go.Scatter(
                x=guided_cfg_distribution["cfg_scale"],
                y=guided_cfg_distribution["alignment_mean"],
                mode="lines",
                line=dict(width=3, color="#d66b2c"),
                name="Guided mean",
                meta="guided_summary",
                customdata=np.column_stack(
                    [
                        guided_cfg_distribution["alignment_min"],
                        guided_cfg_distribution["alignment_max"],
                    ]
                ),
                hoverinfo="none",
            )
        )
        baseline_guided_cfg_fig.add_trace(
            go.Scatter(
                x=all_guided_by_cfg["cfg_scale"],
                y=all_guided_by_cfg["alignment_mean"],
                mode="markers",
                marker=dict(
                    size=7,
                    color="#d66b2c",
                    opacity=0.62,
                    line=dict(width=0.5, color="#ffffff"),
                ),
                name="Guided runs",
                meta="guided",
                customdata=np.column_stack(
                    [
                        all_guided_by_cfg["setup"],
                        all_guided_by_cfg["lr"],
                        all_guided_by_cfg["guide_step"],
                        all_guided_by_cfg["quality_mean"],
                        all_guided_by_cfg["alignment_std"],
                        all_guided_by_cfg["n"],
                    ]
                ),
                hoverinfo="none",
            )
        )
    else:
        raise ValueError(
            "GUIDED_CFG_FIGURE_MODE must be either 'best' or 'distribution'"
        )
    # Plot the blue baseline last so it remains visible above the guided range,
    # summary line, and individual guided points.
    baseline_guided_cfg_fig.data = (
        baseline_guided_cfg_fig.data[1:] + baseline_guided_cfg_fig.data[:1]
    )
    baseline_fig.add_trace(
        go.Scatter(
            x=[baseline["cfg_scale"]],
            y=[baseline_alignment],
            mode="markers",
            marker=dict(size=16, color="#ffd166", symbol="star"),
            name="Selected baseline",
            customdata=np.array(
                [[
                    baseline_setup,
                    baseline["alignment_std"],
                    baseline_quality,
                    baseline["n"],
                ]],
                dtype=object,
            ),
        )
    )
    baseline_cfg_ticks = sd15_agg["cfg_scale"].tolist()
    baseline_fig.update_xaxes(
        fixedrange=True,
        tickmode="array",
        tickvals=baseline_cfg_ticks,
        ticktext=[f"{value:g}" for value in baseline_cfg_ticks],
        range=linear_padded_range(baseline_cfg_ticks, pad_frac=0.04),
    )
    baseline_fig.update_yaxes(fixedrange=True)
    baseline_fig.update_layout(
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.12,
            yanchor="bottom",
        ),
    )
    baseline_quality_fig = go.Figure()
    baseline_quality_fig.add_trace(
        go.Scatter(
            x=sd15_agg["cfg_scale"],
            y=sd15_agg["quality_mean"],
            mode="lines+markers",
            line=dict(width=3, color="#3569b0"),
            marker=dict(size=8, color="#3569b0"),
            name="Vanilla SD1.5",
            customdata=sd15_agg[
                ["setup", "quality_std", "alignment_mean", "n"]
            ].to_numpy(),
            hovertemplate=(
                "%{customdata[0]}<br>"
                "CFG=%{x:g}<br>"
                "mean quality=%{y:.3f}<br>"
                "quality std=%{customdata[1]:.3f}<br>"
                "mean alignment=%{customdata[2]:.3f}<br>"
                "n=%{customdata[3]:.0f}<extra></extra>"
            ),
        )
    )
    baseline_quality_fig.add_trace(
        go.Scatter(
            x=[baseline["cfg_scale"]],
            y=[baseline_quality],
            mode="markers",
            marker=dict(
                size=16,
                color="#ffd166",
                symbol="star",
                line=dict(width=1, color="#172033"),
            ),
            name="Selected baseline",
            customdata=np.array(
                [[
                    baseline_setup,
                    baseline["quality_std"],
                    baseline_alignment,
                    baseline["n"],
                ]],
                dtype=object,
            ),
            hovertemplate=(
                f"{baseline_setup}<br>"
                f"CFG={baseline['cfg_scale']:g}<br>"
                f"mean quality={baseline_quality:.3f}<extra></extra>"
            ),
        )
    )
    baseline_quality_fig.update_layout(
        xaxis=dict(
            title="CFG scale",
            fixedrange=True,
            tickmode="array",
            tickvals=baseline_cfg_ticks,
            ticktext=[f"{value:g}" for value in baseline_cfg_ticks],
            range=linear_padded_range(baseline_cfg_ticks, pad_frac=0.04),
        ),
        yaxis=dict(title="Mean quality", fixedrange=True),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.12,
            yanchor="bottom",
        ),
    )
    comparison_cfg_ticks = sorted(
        set(sd15_agg["cfg_scale"]) | set(best_guided_by_cfg["cfg_scale"])
    )
    baseline_guided_cfg_fig.update_layout(
        hovermode="closest",
        hoverdistance=20,
        xaxis=dict(
            title="CFG scale",
            fixedrange=True,
            tickmode="array",
            tickvals=comparison_cfg_ticks,
            ticktext=[f"{value:g}" for value in comparison_cfg_ticks],
            range=linear_padded_range(comparison_cfg_ticks, pad_frac=0.04),
        ),
        yaxis=dict(title="Mean alignment", fixedrange=True),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.12,
            yanchor="bottom",
        ),
    )

    alignment_min = float(agg["alignment_mean"].min())
    alignment_max = float(agg["alignment_mean"].max())
    alignment_color_min = 3.0
    alignment_color_top = max(alignment_max, baseline_alignment)
    alignment_color_max = alignment_color_top + 0.03 * (
        alignment_color_top - alignment_color_min
    )
    alignment_colorbar_ticks = [
        alignment_color_min,
        baseline_alignment,
        alignment_max,
    ]
    alignment_colorbar_ticktexts = [
        f"≤ {alignment_color_min:.1f}",
        f"SD1.5 baseline {baseline_alignment:.2f}",
        f"max {alignment_max:.2f}",
    ]

    alignment_fig = px.scatter_3d(
        agg,
        x="lr",
        y="cfg_scale",
        z="guide_step",
        color="alignment_mean",
        size="n",
        hover_data=["alignment_std", "quality_mean", "n"],
        color_continuous_scale="Turbo",
        range_color=[alignment_color_min, alignment_color_max],
        labels={
            "lr": "Learning rate",
            "cfg_scale": "CFG scale",
            "guide_step": "Guidance step",
            "alignment_mean": "Mean alignment",
        },
    )
    alignment_fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="Learning rate",
                type="log",
                range=log10_padded_range(agg["lr"]),
                dtick=1,
                exponentformat="e",
                showexponent="all",
            ),
            yaxis=dict(
                title="CFG scale",
                range=linear_padded_range(agg["cfg_scale"]),
            ),
            zaxis=dict(
                title="Guidance step",
                range=linear_padded_range(agg["guide_step"]),
            ),
            camera=dict(
                projection=dict(type="perspective"),
                eye=dict(x=1.6, y=1.6, z=1.1),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=1.0),
        ),
        coloraxis_colorbar=dict(
            title="Alignment",
            tickmode="array",
            tickvals=alignment_colorbar_ticks,
            ticktext=alignment_colorbar_ticktexts,
        ),
    )
    alignment_fig.update_traces(
        marker=dict(opacity=1.0),
        customdata=np.column_stack(
            [agg["alignment_mean"], agg["quality_mean"]]
        ),
    )

    quality_max = float(agg["quality_mean"].max())
    quality_color_min = 3.0
    quality_color_max = max(quality_max, baseline_quality)
    quality_colorbar_ticks = [
        quality_color_min,
        baseline_quality,
        quality_max,
    ]
    quality_colorbar_ticktexts = [
        f"≤ {quality_color_min:.1f}",
        f"SD1.5 baseline {baseline_quality:.2f}",
        f"max {quality_max:.2f}",
    ]

    quality_fig = px.scatter_3d(
        agg,
        x="lr",
        y="cfg_scale",
        z="guide_step",
        color="quality_mean",
        size="n",
        hover_data=["quality_std", "alignment_mean", "n"],
        color_continuous_scale="Turbo",
        range_color=[quality_color_min, quality_color_max],
        labels={
            "lr": "Learning rate",
            "cfg_scale": "CFG scale",
            "guide_step": "Guidance step",
            "quality_mean": "Mean quality",
        },
    )
    quality_fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="Learning rate",
                type="log",
                range=log10_padded_range(agg["lr"]),
                dtick=1,
                exponentformat="e",
                showexponent="all",
            ),
            yaxis=dict(
                title="CFG scale",
                range=linear_padded_range(agg["cfg_scale"]),
            ),
            zaxis=dict(
                title="Guidance step",
                range=linear_padded_range(agg["guide_step"]),
            ),
            camera=dict(
                projection=dict(type="perspective"),
                eye=dict(x=1.6, y=1.6, z=1.1),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=1.0),
        ),
        coloraxis_colorbar=dict(
            title="Quality",
            tickmode="array",
            tickvals=quality_colorbar_ticks,
            ticktext=quality_colorbar_ticktexts,
        ),
    )
    quality_fig.update_traces(
        marker=dict(opacity=1.0),
        customdata=np.column_stack(
            [agg["quality_mean"], agg["alignment_mean"]]
        ),
    )

    guided_quality_alignment = (
        guided.groupby("setup", as_index=False)
        .agg(
            quality_mean=("quality", "mean"),
            quality_std=("quality", "std"),
            alignment_mean=("alignment", "mean"),
            alignment_std=("alignment", "std"),
            n=("alignment", "size"),
            lr=("lr", "first"),
            cfg_scale=("cfg_scale", "first"),
            guide_step=("guide_step", "first"),
        )
    )
    sd15_quality_alignment = sd15_agg[
        [
            "setup",
            "quality_mean",
            "alignment_mean",
            "alignment_std",
            "n",
            "cfg_scale",
        ]
    ].copy()
    sd15_quality_std = (
        judgment[judgment["setup"].isin(sd15_quality_alignment["setup"])]
        .groupby("setup")["quality"]
        .std()
    )
    sd15_quality_alignment["quality_std"] = sd15_quality_alignment["setup"].map(
        sd15_quality_std
    )
    cfg_color_min = min(
        guided_quality_alignment["cfg_scale"].min(),
        sd15_quality_alignment["cfg_scale"].min(),
    )
    cfg_color_max = max(
        guided_quality_alignment["cfg_scale"].max(),
        sd15_quality_alignment["cfg_scale"].max(),
    )

    quality_alignment_fig = go.Figure()
    # Draw regular experiment points first so selected stars remain on top.
    guided_quality_customdata = np.column_stack(
        [
            guided_quality_alignment["setup"],
            np.full(len(guided_quality_alignment), "Guided SD1.5"),
            guided_quality_alignment["lr"],
            guided_quality_alignment["cfg_scale"],
            guided_quality_alignment["guide_step"],
            guided_quality_alignment["quality_std"],
            guided_quality_alignment["alignment_std"],
            guided_quality_alignment["n"],
        ]
    )
    sd15_quality_customdata = np.column_stack(
        [
            sd15_quality_alignment["setup"],
            np.full(len(sd15_quality_alignment), "Vanilla SD1.5"),
            np.full(len(sd15_quality_alignment), "—"),
            sd15_quality_alignment["cfg_scale"],
            np.full(len(sd15_quality_alignment), "—"),
            sd15_quality_alignment["quality_std"],
            sd15_quality_alignment["alignment_std"],
            sd15_quality_alignment["n"],
        ]
    )
    quality_alignment_fig.add_trace(
        go.Scattergl(
            x=pd.concat(
                [
                    guided_quality_alignment["quality_mean"],
                    sd15_quality_alignment["quality_mean"],
                ],
                ignore_index=True,
            ),
            y=pd.concat(
                [
                    guided_quality_alignment["alignment_mean"],
                    sd15_quality_alignment["alignment_mean"],
                ],
                ignore_index=True,
            ),
            mode="markers",
            name="Experiments",
            marker=dict(
                size=9,
                color=pd.concat(
                    [
                        guided_quality_alignment["cfg_scale"],
                        sd15_quality_alignment["cfg_scale"],
                    ],
                    ignore_index=True,
                ),
                coloraxis="coloraxis",
                symbol="circle",
                opacity=0.75,
            ),
            customdata=np.concatenate(
                [guided_quality_customdata, sd15_quality_customdata],
                axis=0,
            ),
            hoverinfo="none",
        )
    )

    selected_guided_point = guided_quality_alignment[
        guided_quality_alignment["setup"] == best_setup
    ]
    selected_baseline_point = sd15_quality_alignment[
        sd15_quality_alignment["setup"] == baseline_setup
    ]
    selected_guided_cfg = float(selected_guided_point["cfg_scale"].iloc[0])
    selected_baseline_cfg = float(selected_baseline_point["cfg_scale"].iloc[0])

    def cfg_marker_color(cfg_value: float) -> str:
        normalized = (
            (cfg_value - cfg_color_min) / (cfg_color_max - cfg_color_min)
            if cfg_color_max != cfg_color_min
            else 0.5
        )
        return px.colors.sample_colorscale("Turbo", [normalized])[0]

    quality_alignment_fig.add_trace(
        go.Scatter(
            x=selected_guided_point["quality_mean"],
            y=selected_guided_point["alignment_mean"],
            mode="markers",
            name="Selected guided pipeline",
            marker=dict(
                size=18,
                color=cfg_marker_color(selected_guided_cfg),
                symbol="star",
                line=dict(width=1.5, color="#172033"),
            ),
            customdata=np.column_stack(
                [
                    selected_guided_point["setup"],
                    np.full(len(selected_guided_point), "Guided SD1.5 · selected"),
                    selected_guided_point["lr"],
                    selected_guided_point["cfg_scale"],
                    selected_guided_point["guide_step"],
                    selected_guided_point["quality_std"],
                    selected_guided_point["alignment_std"],
                    selected_guided_point["n"],
                ]
            ),
            hoverinfo="none",
        )
    )
    quality_alignment_fig.add_trace(
        go.Scatter(
            x=selected_baseline_point["quality_mean"],
            y=selected_baseline_point["alignment_mean"],
            mode="markers",
            name="Selected baseline",
            marker=dict(
                size=18,
                color=cfg_marker_color(selected_baseline_cfg),
                symbol="star",
                line=dict(width=1.5, color="#172033"),
            ),
            customdata=np.column_stack(
                [
                    selected_baseline_point["setup"],
                    np.full(len(selected_baseline_point), "Vanilla SD1.5 · selected"),
                    np.full(len(selected_baseline_point), "—"),
                    selected_baseline_point["cfg_scale"],
                    np.full(len(selected_baseline_point), "—"),
                    selected_baseline_point["quality_std"],
                    selected_baseline_point["alignment_std"],
                    selected_baseline_point["n"],
                ]
            ),
            hoverinfo="none",
        )
    )
    quality_alignment_fig.update_layout(
        title=None,
        xaxis=dict(title="Mean quality", fixedrange=True),
        yaxis=dict(
            title="Mean alignment",
            fixedrange=True,
            scaleanchor="x",
            scaleratio=1,
        ),
        coloraxis=dict(
            colorscale="Turbo",
            cmin=cfg_color_min,
            cmax=cfg_color_max,
            colorbar=dict(title="CFG scale"),
        ),
        hovermode="closest",
        hoverdistance=25,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.02,
            yanchor="bottom",
        ),
        height=590,
    )
    quality_alignment_plot_id = "quality-alignment-plot"
    quality_alignment_plot_html = plot_div(
        quality_alignment_fig,
        include_plotlyjs=PLOTLY_JS_URL,
        div_id=quality_alignment_plot_id,
        margin=dict(l=65, r=35, t=55, b=60),
        scroll_zoom=False,
    )
    quality_alignment_content = f"""
    <div id="quality-alignment-info" style="min-height:44px;padding:6px 80px 2px;font-size:14px;line-height:1.3;color:#334155;">
      Наведите курсор на точку, чтобы увидеть параметры эксперимента.
    </div>
    {quality_alignment_plot_html}
    <script>
      (() => {{
        const plot = document.getElementById({json.dumps(quality_alignment_plot_id)});
        const info = document.getElementById("quality-alignment-info");
        let resetTimer = null;

        function escapeHtml(value) {{
          return String(value).replace(/[&<>"']/g, character => ({{
            "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
          }})[character]);
        }}

        plot.on("plotly_hover", event => {{
          if (resetTimer !== null) {{
            clearTimeout(resetTimer);
            resetTimer = null;
          }}
          const point = event.points[0];
          const values = point.customdata;
          const optimizer = values[2] === "—"
            ? ""
            : `, lr = ${{Number(values[2]).toExponential(1)}}, guidance step = ${{values[4]}}`;
          info.innerHTML =
            `mean quality = ${{Number(point.x).toFixed(2)}}, ` +
            `mean alignment = ${{Number(point.y).toFixed(2)}}, ` +
            `CFG = ${{values[3]}}${{optimizer}}`;
        }});
        plot.on("plotly_unhover", () => {{
          if (resetTimer !== null) clearTimeout(resetTimer);
          resetTimer = setTimeout(() => {{
            resetTimer = null;
            info.textContent =
              "Наведите курсор на точку, чтобы увидеть параметры эксперимента.";
          }}, 140);
        }});
      }})();
    </script>
    """

    curves = []
    for sample_dir in sorted(best_run_dir.iterdir()):
        if sample_dir.is_dir() and (curve := load_yes_no_curve(sample_dir)) is not None:
            curves.append(add_vlm_loss(curve))
    curves_df = pd.concat(curves, ignore_index=True)
    mean_curve = (
        curves_df.groupby("guidance_iteration", as_index=False)
        .agg(vlm_loss_mean=("vlm_loss", "mean"))
        .fillna(0)
    )
    final_loss_by_sample = (
        curves_df.sort_values("guidance_iteration")
        .groupby("sample")["vlm_loss"]
        .last()
    )
    log_final_loss_by_sample = np.log10(final_loss_by_sample)
    final_loss_color_min = float(log_final_loss_by_sample.min())
    final_loss_color_max = float(log_final_loss_by_sample.max())
    trajectory_colorscale = [
        "#5b2387",
        "#3949ab",
        "#1976a3",
        "#128277",
        "#2e7d4f",
        "#b45f06",
        "#b3263e",
    ]

    def sample_trajectory_color(sample: str) -> str:
        normalized = (
            (float(log_final_loss_by_sample[sample]) - final_loss_color_min)
            / (final_loss_color_max - final_loss_color_min)
            if final_loss_color_max != final_loss_color_min
            else 0.5
        )
        return px.colors.sample_colorscale(trajectory_colorscale, [normalized])[0]

    trajectory_fig = go.Figure()
    for sample, sample_df in curves_df.groupby("sample"):
        sample_df = sample_df.sort_values("guidance_iteration")
        trajectory_color = sample_trajectory_color(sample)
        trajectory_fig.add_trace(
            go.Scatter(
                x=sample_df["guidance_iteration"],
                y=sample_df["vlm_loss"],
                customdata=np.array([[sample] for _ in range(len(sample_df))]),
                mode="lines+markers",
                line=dict(width=1.5, color=trajectory_color),
                marker=dict(size=6, color=trajectory_color),
                opacity=0.65,
                name=sample,
                showlegend=False,
                hoverinfo="none",
            )
        )
        hover_x, hover_y = make_line_hover_points(
            sample_df[["guidance_iteration", "vlm_loss"]].rename(
                columns={"vlm_loss": "p_yes"}
            )
        )
        trajectory_fig.add_trace(
            go.Scatter(
                x=hover_x,
                y=hover_y,
                mode="markers",
                customdata=np.array([[sample] for _ in hover_x]),
                hoverinfo="none",
                marker=dict(size=14, color="rgba(0, 0, 0, 0)"),
                showlegend=False,
            )
        )
    trajectory_fig.add_trace(
        go.Scatter(
            x=mean_curve["guidance_iteration"],
            y=mean_curve["vlm_loss_mean"],
            mode="lines+markers",
            line=dict(width=5, color="#151515"),
            marker=dict(size=9, color="#151515"),
            hoverinfo="skip",
            name="Mean VLM loss",
            showlegend=True,
        )
    )
    trajectory_fig.update_layout(
        title="VLM loss during guidance",
        xaxis=dict(
            title="Guidance iteration",
            range=linear_padded_range(curves_df["guidance_iteration"], pad_frac=0.04),
            fixedrange=True,
        ),
        yaxis=dict(
            title="VLM loss",
            type="log",
            exponentformat="e",
            showexponent="all",
            fixedrange=True,
        ),
        hovermode="closest",
        hoverdistance=20,
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.02,
            yanchor="bottom",
        ),
        height=350,
    )

    all_mean_curve_rows = []
    final_loss_rows = []
    params_by_setup = {row["setup"]: row for row in guided_params}
    for setup, params in params_by_setup.items():
        run_dir = GEN_DIR / setup
        run_curves = []
        sample_final_losses = []
        for sample_dir in sorted(run_dir.iterdir()):
            if sample_dir.is_dir():
                curve = load_yes_no_curve(sample_dir)
                if curve is not None:
                    curve = add_vlm_loss(curve)
                    run_curves.append(curve)
                    final_point = curve.sort_values("guidance_iteration").iloc[-1]
                    sample_final_losses.append(float(final_point["vlm_loss"]))
        if not run_curves:
            continue
        run_mean = (
            pd.concat(run_curves, ignore_index=True)
            .groupby("guidance_iteration", as_index=False)
            .agg(vlm_loss_mean=("vlm_loss", "mean"), n=("vlm_loss", "size"))
        )
        run_mean["setup"] = setup
        run_mean["lr"] = params["lr"]
        run_mean["cfg_scale"] = params["cfg_scale"]
        run_mean["guide_step"] = params["guide_step"]
        all_mean_curve_rows.append(run_mean)
        final_loss_rows.append(
            {
                "setup": setup,
                "final_vlm_loss_mean": np.mean(sample_final_losses),
                "final_vlm_loss_std": np.std(sample_final_losses),
                "n_loss_samples": len(sample_final_losses),
                "lr": params["lr"],
                "cfg_scale": params["cfg_scale"],
                "guide_step": params["guide_step"],
            }
        )

    all_mean_curves = pd.concat(all_mean_curve_rows, ignore_index=True)
    guided_judge_by_setup = (
        guided.groupby("setup", as_index=False)
        .agg(
            alignment_mean=("alignment", "mean"),
            alignment_std=("alignment", "std"),
            quality_mean=("quality", "mean"),
            n_judge_samples=("alignment", "size"),
        )
    )
    final_loss_alignment = pd.DataFrame(final_loss_rows).merge(
        guided_judge_by_setup,
        on="setup",
        how="inner",
    )
    final_loss_alignment_fig = go.Figure(
        go.Scattergl(
            x=final_loss_alignment["final_vlm_loss_mean"],
            y=final_loss_alignment["alignment_mean"],
            mode="markers",
            marker=dict(
                size=9,
                color="#3569b0",
                opacity=0.78,
                line=dict(width=0.5, color="#ffffff"),
            ),
            name="Guided pipelines",
            customdata=final_loss_alignment[
                [
                    "setup",
                    "lr",
                    "cfg_scale",
                    "guide_step",
                    "final_vlm_loss_std",
                    "alignment_std",
                    "quality_mean",
                    "n_loss_samples",
                    "n_judge_samples",
                ]
            ].to_numpy(),
            hovertemplate=(
                "%{customdata[0]}<br>"
                "mean final VLM loss=%{x:.6f}<br>"
                "mean alignment=%{y:.3f}<br>"
                "learning rate=%{customdata[1]:.1e}<br>"
                "CFG=%{customdata[2]:g}<br>"
                "guidance step=%{customdata[3]:.0f}<br>"
                "final loss std=%{customdata[4]:.6f}<br>"
                "alignment std=%{customdata[5]:.3f}<br>"
                "mean quality=%{customdata[6]:.3f}<br>"
                "loss samples=%{customdata[7]:.0f}<br>"
                "judge samples=%{customdata[8]:.0f}<extra></extra>"
            ),
        )
    )
    selected_loss_point = final_loss_alignment[
        final_loss_alignment["setup"] == best_setup
    ]
    if not selected_loss_point.empty:
        final_loss_alignment_fig.add_trace(
            go.Scattergl(
                x=selected_loss_point["final_vlm_loss_mean"],
                y=selected_loss_point["alignment_mean"],
                mode="markers",
                marker=dict(
                    size=18,
                    color="#ffd166",
                    symbol="star",
                    line=dict(width=1, color="#172033"),
                ),
                name="Selected pipeline",
                customdata=selected_loss_point[
                    [
                        "setup",
                        "lr",
                        "cfg_scale",
                        "guide_step",
                        "final_vlm_loss_std",
                        "alignment_std",
                        "quality_mean",
                        "n_loss_samples",
                        "n_judge_samples",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    "mean final VLM loss=%{x:.6f}<br>"
                    "mean alignment=%{y:.3f}<extra></extra>"
                ),
            )
        )
    final_loss_alignment_fig.update_layout(
        title=None,
        xaxis=dict(title="Mean final VLM loss", fixedrange=True),
        yaxis=dict(title="Mean alignment", fixedrange=True),
        hovermode="closest",
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.02,
            yanchor="bottom",
        ),
        height=590,
    )
    final_vlm_loss_by_cfg = (
        final_loss_alignment.groupby("cfg_scale", as_index=False)
        .agg(mean_final_vlm_loss=("final_vlm_loss_mean", "mean"))
        .sort_values("cfg_scale")
    )
    final_vlm_loss_by_cfg_fig = go.Figure(
        go.Scatter(
            x=final_vlm_loss_by_cfg["cfg_scale"],
            y=final_vlm_loss_by_cfg["mean_final_vlm_loss"],
            mode="lines+markers",
            line=dict(width=3, color="#3569b0"),
            marker=dict(size=8, color="#3569b0"),
            hoverinfo="none",
        )
    )
    final_vlm_loss_cfg_ticks = final_vlm_loss_by_cfg["cfg_scale"].tolist()
    final_vlm_loss_by_cfg_fig.update_layout(
        xaxis=dict(
            title="CFG scale",
            tickmode="array",
            tickvals=final_vlm_loss_cfg_ticks,
            ticktext=[f"{value:g}" for value in final_vlm_loss_cfg_ticks],
            range=linear_padded_range(final_vlm_loss_cfg_ticks, pad_frac=0.04),
            fixedrange=True,
        ),
        yaxis=dict(
            title="Mean final VLM loss",
            type="log",
            exponentformat="e",
            showexponent="all",
            fixedrange=True,
        ),
        height=390,
    )
    all_mean_trajectories_fig = go.Figure()
    all_mean_hover_proxies = []
    pipeline_legend_shown = False
    ordered_mean_setups = sorted(
        setup for setup in all_mean_curves["setup"].unique() if setup != best_setup
    )
    if best_setup in set(all_mean_curves["setup"]):
        ordered_mean_setups.append(best_setup)

    def parameter_trace_colors(values, log_scale=False):
        numeric_values = np.asarray(values, dtype=float)
        scale_values = np.log10(numeric_values) if log_scale else numeric_values
        value_min = float(scale_values.min())
        value_max = float(scale_values.max())
        if value_min == value_max:
            normalized = np.full(len(scale_values), 0.5)
        else:
            normalized = (scale_values - value_min) / (value_max - value_min)
        return px.colors.sample_colorscale("Turbo", normalized.tolist())

    mean_guidance_values = [
        float(params_by_setup[setup]["guide_step"]) for setup in ordered_mean_setups
    ]
    mean_lr_values = [
        float(params_by_setup[setup]["lr"]) for setup in ordered_mean_setups
    ]
    mean_cfg_values = [
        float(params_by_setup[setup]["cfg_scale"]) for setup in ordered_mean_setups
    ]
    all_mean_color_modes = {
        "guidance_step": {
            "label": "Guidance step",
            "colors": parameter_trace_colors(mean_guidance_values),
            "min": f"{min(mean_guidance_values):g}",
            "max": f"{max(mean_guidance_values):g}",
        },
        "lr": {
            "label": "Learning rate",
            "colors": parameter_trace_colors(mean_lr_values, log_scale=True),
            "min": f"{min(mean_lr_values):.0e}",
            "max": f"{max(mean_lr_values):.0e}",
        },
        "cfg_scale": {
            "label": "CFG scale",
            "colors": parameter_trace_colors(mean_cfg_values),
            "min": f"{min(mean_cfg_values):g}",
            "max": f"{max(mean_cfg_values):g}",
        },
    }
    initial_mean_colors = all_mean_color_modes["cfg_scale"]["colors"]

    for setup_index, setup in enumerate(ordered_mean_setups):
        setup_df = all_mean_curves[all_mean_curves["setup"] == setup]
        setup_df = setup_df.sort_values("guidance_iteration")
        params = params_by_setup[setup]
        is_selected_pipeline = setup == best_setup
        line_color = "#111827" if is_selected_pipeline else initial_mean_colors[setup_index]
        line_width = 4.5 if is_selected_pipeline else 1.6
        marker_size = 10 if is_selected_pipeline else 5
        trace_opacity = 1.0 if is_selected_pipeline else 0.55
        all_mean_trajectories_fig.add_trace(
            go.Scattergl(
                x=setup_df["guidance_iteration"],
                y=setup_df["vlm_loss_mean"],
                mode="lines+markers",
                line=dict(
                    width=line_width,
                    color=line_color,
                    dash="dash" if is_selected_pipeline else "solid",
                ),
                marker=dict(
                    size=marker_size,
                    color="#ffffff" if is_selected_pipeline else line_color,
                    line=(
                        dict(width=2, color="#111827")
                        if is_selected_pipeline
                        else dict(width=0)
                    ),
                ),
                opacity=trace_opacity,
                name=(
                    "Selected best pipeline"
                    if is_selected_pipeline
                    else "Guided pipelines"
                ),
                legendgroup=(
                    "selected-pipeline" if is_selected_pipeline else "pipelines"
                ),
                showlegend=is_selected_pipeline or not pipeline_legend_shown,
                customdata=np.column_stack(
                    [
                        np.full(len(setup_df), setup),
                        np.full(len(setup_df), params["lr"]),
                        np.full(len(setup_df), params["cfg_scale"]),
                        np.full(len(setup_df), params["guide_step"]),
                        setup_df["n"],
                    ]
                ),
                hoverinfo="none",
            )
        )
        if not is_selected_pipeline:
            pipeline_legend_shown = True
        hover_input = setup_df[["guidance_iteration", "vlm_loss_mean"]].rename(
            columns={"vlm_loss_mean": "p_yes"}
        )
        hover_x, hover_y = make_line_hover_points(hover_input)
        all_mean_hover_proxies.append(
            {
                "setup": setup,
                "x": hover_x,
                "y": hover_y,
                "customdata": np.column_stack(
                    [
                        np.full(len(hover_x), setup),
                        np.full(len(hover_x), params["lr"]),
                        np.full(len(hover_x), params["cfg_scale"]),
                        np.full(len(hover_x), params["guide_step"]),
                        np.full(len(hover_x), setup_df["n"].iloc[0]),
                    ]
                ),
            }
        )

    overall_mean_curve = (
        all_mean_curves.groupby("guidance_iteration", as_index=False)
        .agg(vlm_loss_mean=("vlm_loss_mean", "mean"))
        .sort_values("guidance_iteration")
    )
    all_mean_trajectories_fig.add_trace(
        go.Scattergl(
            x=overall_mean_curve["guidance_iteration"],
            y=overall_mean_curve["vlm_loss_mean"],
            mode="lines+markers",
            line=dict(width=5, color="#151515"),
            marker=dict(size=9, color="#151515"),
            opacity=1.0,
            name="Mean across pipelines",
            hoverinfo="none",
        )
    )
    all_mean_visible_trace_count = len(all_mean_trajectories_fig.data)
    all_mean_base_opacities = [
        float(trace.opacity) for trace in all_mean_trajectories_fig.data
    ]
    all_mean_base_widths = [
        float(trace.line.width) for trace in all_mean_trajectories_fig.data
    ]
    for proxy in all_mean_hover_proxies:
        all_mean_trajectories_fig.add_trace(
            go.Scattergl(
                x=proxy["x"],
                y=proxy["y"],
                mode="markers",
                marker=dict(size=14, color="rgba(0, 0, 0, 0)"),
                name=proxy["setup"],
                showlegend=False,
                customdata=proxy["customdata"],
                hoverinfo="none",
            )
        )
    all_mean_trajectories_fig.update_layout(
        xaxis=dict(
            title="Guidance iteration",
            tickmode="linear",
            dtick=1,
            range=linear_padded_range(
                all_mean_curves["guidance_iteration"], pad_frac=0.04
            ),
            fixedrange=True,
        ),
        yaxis=dict(
            title="Mean VLM loss",
            type="log",
            exponentformat="e",
            showexponent="all",
            fixedrange=True,
        ),
        hovermode="closest",
        hoverdistance=20,
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.02,
            yanchor="bottom",
            itemwidth=70,
        ),
        height=590,
    )
    all_mean_plot_id = "all-mean-p-yes-plot"
    all_mean_plot_html = plot_div(
        all_mean_trajectories_fig,
        include_plotlyjs=PLOTLY_JS_URL,
        div_id=all_mean_plot_id,
        margin=dict(l=65, r=35, t=55, b=55),
        scroll_zoom=False,
    )
    turbo_gradient = ", ".join(
        px.colors.sample_colorscale("Turbo", np.linspace(0, 1, 9).tolist())
    )
    all_mean_trajectory_content = f"""
    <style>
      .trajectory-color-controls {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px 14px;
        padding: 10px 65px 2px;
        color: #334155;
        font-size: 14px;
      }}
      .trajectory-color-legend {{
        display: grid;
        grid-template-columns: auto 132px;
        align-items: center;
        gap: 5px 9px;
        min-width: 230px;
      }}
      .trajectory-color-gradient {{
        height: 8px;
        border-radius: 999px;
        background: linear-gradient(90deg, {turbo_gradient});
      }}
      .trajectory-color-range {{
        grid-column: 2;
        display: flex;
        justify-content: space-between;
        margin-top: -3px;
        font-size: 12px;
        color: #667085;
      }}
    </style>
    <div class="trajectory-color-controls">
      <div class="trajectory-color-legend">
        <span>CFG scale</span>
        <span class="trajectory-color-gradient"></span>
        <span class="trajectory-color-range">
          <span>{min(mean_cfg_values):g}</span>
          <span>{max(mean_cfg_values):g}</span>
        </span>
      </div>
    </div>
    <div id="all-mean-p-yes-info" style="min-height:44px;padding:6px 80px 2px;font-size:14px;line-height:1.3;color:#334155;">
      Наведите курсор на траекторию, чтобы увидеть параметры pipeline.
    </div>
    {all_mean_plot_html}
    <script>
      (() => {{
        const plot = document.getElementById({json.dumps(all_mean_plot_id)});
        const info = document.getElementById("all-mean-p-yes-info");
        const visibleCount = {all_mean_visible_trace_count};
        const pipelineCount = {all_mean_visible_trace_count - 1};
        const baseOpacities = {json.dumps(all_mean_base_opacities)};
        const baseWidths = {json.dumps(all_mean_base_widths)};
        let selectedCurve = null;
        let resetTimer = null;

        function focus(curve) {{
          if (resetTimer !== null) {{
            clearTimeout(resetTimer);
            resetTimer = null;
          }}
          if (selectedCurve === curve) return;

          const previousCurve = selectedCurve;
          if (curve === null) {{
            const indices = Array.from({{length: visibleCount}}, (_, index) => index);
            Plotly.restyle(
              plot,
              {{opacity: baseOpacities, "line.width": baseWidths}},
              indices
            );
          }} else {{
            if (previousCurve === null) {{
              const indices = Array.from({{length: visibleCount}}, (_, index) => index);
              Plotly.restyle(
                plot,
                {{
                  opacity: indices.map(() => 0.06),
                  "line.width": baseWidths
                }},
                indices
              );
            }} else {{
              Plotly.restyle(
                plot,
                {{opacity: 0.06, "line.width": baseWidths[previousCurve]}},
                [previousCurve]
              );
            }}
            Plotly.restyle(plot, {{opacity: 1.0, "line.width": 5.0}}, [curve]);
          }}
          selectedCurve = curve;
        }}

        plot.on("plotly_hover", event => {{
          const point = event.points[0];
          const hoveredTrace = point.curveNumber;
          focus(hoveredTrace < visibleCount ? hoveredTrace : hoveredTrace - visibleCount);
          if (hoveredTrace === pipelineCount) {{
            info.innerHTML = `<strong>Mean across pipelines</strong>, iteration ${{point.x}}, mean VLM loss = ${{Number(point.y).toFixed(2)}}`;
          }} else {{
            const values = point.customdata;
            info.innerHTML =
              `iteration ${{Number(point.x).toFixed(2)}}, ` +
              `mean VLM loss = ${{Number(point.y).toFixed(2)}}, ` +
              `lr = ${{Number(values[1]).toExponential(1)}}, CFG = ${{values[2]}}, ` +
              `guidance step = ${{values[3]}}`;
          }}
        }});
        plot.on("plotly_unhover", () => {{
          if (resetTimer !== null) clearTimeout(resetTimer);
          resetTimer = setTimeout(() => {{
            resetTimer = null;
            focus(null);
            info.textContent = "Наведите курсор на траекторию, чтобы увидеть параметры pipeline.";
          }}, 140);
        }});
      }})();
    </script>
    """

    prompt_by_sample = (
        judgment[judgment["setup"] == best_setup]
        .drop_duplicates("sample")
        .set_index("sample")["prompt"]
        .to_dict()
    )
    trajectory_hover_rows = {}
    for sample in sorted(curves_df["sample"].unique()):
        intermediate_dir = best_run_dir / sample / "intermediate_finals"
        image_paths = sorted(
            intermediate_dir.glob("*.png"),
            key=lambda path: int(re.search(r"g(\d+)-", path.name).group(1)),
        )
        trajectory_hover_rows[sample] = {
            "sample": sample,
            "prompt": prompt_by_sample.get(sample, sample),
            "images": [publish_image(path) for path in image_paths],
        }

    trajectory_default_sample = sorted(trajectory_hover_rows)[0]
    trajectory_hover_json = json.dumps(
        trajectory_hover_rows,
        ensure_ascii=False,
    ).replace("</", "<\\/")
    trajectory_hover_content = f"""
    <div class="trajectory-layout">
      <div id="vlm-trajectory-info" style="min-height:44px;padding:6px 75px 2px;font-size:14px;line-height:1.3;color:#334155;"></div>
      <div class="trajectory-plot">{plot_div(trajectory_fig, include_plotlyjs=PLOTLY_JS_URL, div_id="vlm-trajectory-plot", scroll_zoom=False)}</div>
      <aside class="trajectory-panel" id="vlm-trajectory-panel" aria-live="polite"></aside>
    </div>
    <script>
      (() => {{
        const rows = {trajectory_hover_json};
        const info = document.getElementById("vlm-trajectory-info");
        const panel = document.getElementById("vlm-trajectory-panel");
        const plot = document.getElementById("vlm-trajectory-plot");
        const sampleTraceCount = {curves_df["sample"].nunique()};
        const sampleTraceSpan = sampleTraceCount * 2;
        const aggregateTraceIndices = [sampleTraceSpan];
        const sampleTraceIndices = Array.from(
          {{length: sampleTraceCount}},
          (_, index) => 2 * index
        );
        let focusedSampleTrace = null;
        let focusResetTimer = null;

        function escapeHtml(value) {{
          return String(value).replace(/[&<>"']/g, character => ({{
            "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
          }})[character]);
        }}

        function visibleSampleTrace(curveNumber) {{
          if (curveNumber >= sampleTraceSpan) return null;
          return 2 * Math.floor(curveNumber / 2);
        }}

        function focusSampleTrace(traceIndex) {{
          if (focusResetTimer !== null) {{
            clearTimeout(focusResetTimer);
            focusResetTimer = null;
          }}
          if (focusedSampleTrace === traceIndex) return;

          const previousTrace = focusedSampleTrace;
          if (traceIndex === null) {{
            Plotly.restyle(
              plot,
              {{
                opacity: sampleTraceIndices.map(() => 0.65),
                "line.width": sampleTraceIndices.map(() => 1.5)
              }},
              sampleTraceIndices
            );
            Plotly.restyle(plot, {{opacity: 1.0}}, aggregateTraceIndices);
          }} else {{
            if (previousTrace === null) {{
              Plotly.restyle(
                plot,
                {{
                  opacity: sampleTraceIndices.map(() => 0.08),
                  "line.width": sampleTraceIndices.map(() => 1.5)
                }},
                sampleTraceIndices
              );
              Plotly.restyle(plot, {{opacity: 0.28}}, aggregateTraceIndices);
            }} else {{
              Plotly.restyle(
                plot,
                {{opacity: 0.08, "line.width": 1.5}},
                [previousTrace]
              );
            }}
            Plotly.restyle(
              plot,
              {{opacity: 1.0, "line.width": 4.0}},
              [traceIndex]
            );
          }}
          focusedSampleTrace = traceIndex;
        }}

        function render(sample, point = null) {{
          const row = rows[sample];
          if (!row) return;
          const images = row.images.map((src, index) => `
            <figure>
              <img src="${{src}}" alt="Trajectory image ${{index}}">
              <figcaption>${{index === 0 ? "Initial rollout" :
                index === row.images.length - 1 ? "Final output" :
              `Update ${{index}}`}}</figcaption>
            </figure>
          `).join("");
          const pointDetails = point
            ? `, iteration ${{Number(point.x).toFixed(2)}}, ` +
              `VLM loss = ${{Number(point.y).toFixed(2)}}`
            : "";
          info.innerHTML =
            `<strong>${{escapeHtml(row.prompt)}}</strong>${{pointDetails}}`;
          panel.innerHTML = `<div class="trajectory-images">${{images}}</div>`;
        }}

        render({json.dumps(trajectory_default_sample, ensure_ascii=False)});
        plot.on("plotly_hover", event => {{
          const point = event.points[0];
          const customdata = point.customdata;
          const sample = Array.isArray(customdata) ? customdata[0] : customdata;
          if (sample) {{
            focusSampleTrace(visibleSampleTrace(point.curveNumber));
            render(sample, point);
          }}
        }});
        plot.on("plotly_unhover", () => {{
          if (focusResetTimer !== null) clearTimeout(focusResetTimer);
          focusResetTimer = setTimeout(() => {{
            focusResetTimer = null;
            focusSampleTrace(null);
          }}, 140);
        }});
        plot.on("plotly_click", event => {{
          const customdata = event.points[0].customdata;
          const sample = Array.isArray(customdata) ? customdata[0] : customdata;
          if (sample) render(sample);
        }});
      }})();
    </script>
    """

    baseline_scores = judgment[judgment["setup"] == baseline_setup][
        ["sample", "prompt", "alignment", "quality"]
    ].rename(
        columns={
            "alignment": "alignment_baseline",
            "quality": "quality_baseline",
        }
    )
    guided_scores = judgment[judgment["setup"] == best_setup][
        ["sample", "prompt", "alignment", "quality"]
    ].rename(
        columns={
            "alignment": "alignment_guided",
            "quality": "quality_guided",
        }
    )
    comparison = guided_scores.merge(
        baseline_scores, on=["sample", "prompt"], how="inner"
    )
    comparison["delta_alignment"] = (
        comparison["alignment_guided"] - comparison["alignment_baseline"]
    )
    comparison["delta_quality"] = (
        comparison["quality_guided"] - comparison["quality_baseline"]
    )
    comparison = comparison.sort_values("delta_alignment")
    comparison_fig = go.Figure(
        go.Bar(
            x=comparison["delta_alignment"],
            y=comparison["sample"],
            orientation="h",
            marker_color=np.where(
                comparison["delta_alignment"] >= 0, "#21845d", "#b83b4a"
            ),
            showlegend=False,
            customdata=np.stack(
                [
                    comparison["sample"],
                    comparison["prompt"],
                    comparison["alignment_baseline"],
                    comparison["alignment_guided"],
                    comparison["quality_baseline"],
                    comparison["quality_guided"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "%{customdata[1]}<br>Δ alignment=%{x:+.0f}<br>"
                "alignment: %{customdata[2]} → %{customdata[3]}<br>"
                "quality: %{customdata[4]} → %{customdata[5]}<extra></extra>"
            ),
        )
    )
    comparison_fig.add_vline(x=0, line_dash="dash", line_color="#93a9c5")
    comparison_fig.add_trace(
        go.Scatter(
            x=np.zeros(len(comparison)),
            y=comparison["sample"],
            mode="markers",
            marker=dict(size=18, color="rgba(0, 0, 0, 0)"),
            customdata=np.stack(
                [
                    comparison["sample"],
                    comparison["prompt"],
                    comparison["alignment_baseline"],
                    comparison["alignment_guided"],
                    comparison["quality_baseline"],
                    comparison["quality_guided"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "%{customdata[1]}<br>Δ alignment=0 or near zero<br>"
                "alignment: %{customdata[2]} → %{customdata[3]}<br>"
                "quality: %{customdata[4]} → %{customdata[5]}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    comparison_fig.update_layout(
        title="Per-prompt alignment change",
        xaxis=dict(
            title="Guided − baseline alignment",
            fixedrange=True,
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            fixedrange=True,
            range=[-0.5, len(comparison) - 0.5],
        ),
        height=820,
        hovermode="closest",
        hoverdistance=30,
        showlegend=False,
    )
    comparison_fig.update_traces(hoverinfo="none", hovertemplate=None)

    comparison_hover_rows = {}
    for _, row in comparison.iterrows():
        sample = row["sample"]
        comparison_hover_rows[sample] = {
            "sample": sample,
            "prompt": row["prompt"],
            "delta_alignment": float(row["delta_alignment"]),
            "alignment_baseline": float(row["alignment_baseline"]),
            "alignment_guided": float(row["alignment_guided"]),
            "quality_baseline": float(row["quality_baseline"]),
            "quality_guided": float(row["quality_guided"]),
            "baseline_image": publish_image(
                GEN_DIR / baseline_setup / sample / "sd15.png"
            ),
            "guided_image": publish_image(
                GEN_DIR / best_setup / sample / "guided_sd15.png"
            ),
        }

    comparison_default_sample = comparison.sort_values(
        "delta_alignment", ascending=False
    ).iloc[0]["sample"]
    comparison_hover_json = json.dumps(
        comparison_hover_rows,
        ensure_ascii=False,
    ).replace("</", "<\\/")
    comparison_hover_content = f"""
    <div class="hover-info-bar" id="alignment-delta-info"></div>
    <div class="comparison-layout">
      <div class="comparison-plot">{plot_div(comparison_fig, include_plotlyjs=PLOTLY_JS_URL, div_id="alignment-delta-plot", scroll_zoom=False)}</div>
      <aside class="comparison-panel" id="alignment-comparison-panel" aria-live="polite"></aside>
    </div>
    <script>
      (() => {{
        const rows = {comparison_hover_json};
        const info = document.getElementById("alignment-delta-info");
        const panel = document.getElementById("alignment-comparison-panel");
        const plot = document.getElementById("alignment-delta-plot");

        function escapeHtml(value) {{
          return String(value).replace(/[&<>"']/g, character => ({{
            "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
          }})[character]);
        }}

        function render(sample) {{
          const row = rows[sample];
          if (!row) return;
          const delta = row.delta_alignment >= 0
            ? `+${{row.delta_alignment.toFixed(0)}}`
            : row.delta_alignment.toFixed(0);
          info.innerHTML = `<strong>${{escapeHtml(row.prompt)}}</strong>, ` +
            `Δ alignment = ${{delta}}, ` +
            `alignment: ${{row.alignment_baseline.toFixed(0)}} → ${{row.alignment_guided.toFixed(0)}}, ` +
            `quality: ${{row.quality_baseline.toFixed(0)}} → ${{row.quality_guided.toFixed(0)}}`;
          panel.innerHTML = `
            <div class="comparison-images">
              <figure class="baseline-example">
                <div class="comparison-image-label">BASELINE · SD1.5</div>
                <img src="${{row.baseline_image}}" alt="Baseline SD1.5">
                <figcaption>
                  alignment ${{row.alignment_baseline.toFixed(0)}},
                  quality ${{row.quality_baseline.toFixed(0)}}
                </figcaption>
              </figure>
              <figure class="guided-example">
                <div class="comparison-image-label">GUIDED · VLM guidance</div>
                <img src="${{row.guided_image}}" alt="Guided SD1.5">
                <figcaption>
                  alignment ${{row.alignment_guided.toFixed(0)}},
                  quality ${{row.quality_guided.toFixed(0)}}
                </figcaption>
              </figure>
            </div>`;
        }}

        render({json.dumps(comparison_default_sample, ensure_ascii=False)});
        plot.on("plotly_hover", event => {{
          const customdata = event.points[0].customdata;
          if (customdata) render(customdata[0]);
        }});
        plot.on("plotly_click", event => {{
          const customdata = event.points[0].customdata;
          if (customdata) render(customdata[0]);
        }});
      }})();
    </script>
    """

    slices_uri = make_parameter_slices(agg, baseline_alignment, best)

    intermediate_cards = []
    sample_dirs = [path for path in sorted(best_run_dir.iterdir()) if path.is_dir()]
    example_dir = next(
        (
            path
            for path in sample_dirs
            if path.name.startswith("024_young_children_marching")
        ),
        sample_dirs[0],
    )
    intermediate_paths = sorted(
        (example_dir / "intermediate_finals").glob("*.png"),
        key=lambda path: int(re.search(r"g(\d+)-", path.name).group(1)),
    )
    for index, path in enumerate(intermediate_paths):
        label = "Initial rollout" if index == 0 else (
            "Final output" if index == len(intermediate_paths) - 1 else f"Guidance iteration {index}"
        )
        intermediate_cards.append(image_card(path, label))

    pair_rows = pd.concat(
        [comparison.head(2), comparison.tail(2)]
    ).drop_duplicates("sample")
    pair_cards = []
    for _, row in pair_rows.iterrows():
        baseline_path = GEN_DIR / baseline_setup / row["sample"] / "sd15.png"
        guided_path = GEN_DIR / best_setup / row["sample"] / "guided_sd15.png"
        pair_cards.append(
            f"""
            <article class="pair-card">
              <h4>{escape(row['prompt'])}</h4>
              <p class="delta">Δ alignment = {row['delta_alignment']:+.0f}</p>
              <div class="image-grid two">
                {image_card(baseline_path, "Baseline", f"alignment {row['alignment_baseline']:.0f}")}
                {image_card(guided_path, "Guided", f"alignment {row['alignment_guided']:.0f}")}
              </div>
            </article>
            """
        )

    pipeline_uri = publish_image(
        REPO_ROOT / "figures" / "guided_generation_pipeline.png"
    )
    baseline_content = hover_info_plot(
        baseline_fig,
        div_id="baseline-cfg-plot",
        info_id="baseline-cfg-info",
        formatter_js="""point => {
          const data = point.customdata || [];
          return `CFG = ${fixed(point.x, 0)}, mean alignment = ${fixed(point.y)}, ` +
            `mean quality = ${fixed(data[2])}`;
        }""",
        margin=dict(l=55, r=35, t=70, b=55),
    )
    baseline_quality_content = hover_info_plot(
        baseline_quality_fig,
        div_id="baseline-quality-cfg-plot",
        info_id="baseline-quality-cfg-info",
        formatter_js="""point => {
          const data = point.customdata || [];
          return `CFG = ${fixed(point.x, 0)}, mean quality = ${fixed(point.y)}, ` +
            `mean alignment = ${fixed(data[2])}`;
        }""",
        margin=dict(l=55, r=35, t=70, b=55),
    )
    baseline_guided_cfg_content = hover_info_plot(
        baseline_guided_cfg_fig,
        div_id="baseline-guided-cfg-plot",
        info_id="baseline-guided-cfg-info",
        formatter_js="""point => {
          const data = point.customdata || [];
          const kind = point.data.meta;
          if (kind === "baseline") {
            return `CFG = ${fixed(point.x, 0)}, mean alignment = ${fixed(point.y)}, ` +
              `mean quality = ${fixed(data[2])}`;
          }
          if (kind === "guided_summary") {
            return `CFG = ${fixed(point.x, 0)}, guided mean alignment = ${fixed(point.y)}, ` +
              `range = ${fixed(data[0])}–${fixed(data[1])}`;
          }
          return `CFG = ${fixed(point.x, 0)}, mean alignment = ${fixed(point.y)}, ` +
            `learning rate = ${scientific(data[1])}, guidance step = ${fixed(data[2], 0)}, ` +
            `mean quality = ${fixed(data[3])}`;
        }""",
        margin=dict(l=55, r=35, t=70, b=55),
    )
    alignment_content = hover_info_plot(
        alignment_fig,
        div_id="alignment-parameter-space-plot",
        info_id="alignment-parameter-space-info",
        formatter_js="""point => {
          const data = point.customdata || [];
          return `learning rate = ${scientific(point.x)}, ` +
            `CFG = ${fixed(point.y, 0)}, guidance step = ${fixed(point.z, 0)}, ` +
            `mean alignment = ${fixed(data[0])}, mean quality = ${fixed(data[1])}`;
        }""",
        scroll_zoom=True,
    )
    quality_content = hover_info_plot(
        quality_fig,
        div_id="quality-parameter-space-plot",
        info_id="quality-parameter-space-info",
        formatter_js="""point => {
          const data = point.customdata || [];
          return `learning rate = ${scientific(point.x)}, ` +
            `CFG = ${fixed(point.y, 0)}, guidance step = ${fixed(point.z, 0)}, ` +
            `mean quality = ${fixed(data[0])}, mean alignment = ${fixed(data[1])}`;
        }""",
        scroll_zoom=True,
    )
    final_loss_alignment_content = hover_info_plot(
        final_loss_alignment_fig,
        div_id="final-loss-alignment-plot",
        info_id="final-loss-alignment-info",
        formatter_js="""point => {
          const data = point.customdata || [];
          return `mean final VLM loss = ${fixed(point.x, 2)}, mean alignment = ${fixed(point.y)}, ` +
            `learning rate = ${scientific(data[1])}, CFG = ${fixed(data[2], 0)}, ` +
            `guidance step = ${fixed(data[3], 0)}, mean quality = ${fixed(data[6])}`;
        }""",
        margin=dict(l=65, r=35, t=55, b=60),
    )
    final_vlm_loss_by_cfg_content = hover_info_plot(
        final_vlm_loss_by_cfg_fig,
        div_id="final-vlm-loss-by-cfg-plot",
        info_id="final-vlm-loss-by-cfg-info",
        formatter_js="""point =>
          `CFG = ${fixed(point.x, 0)}, mean final VLM loss = ${fixed(point.y, 2)}`
        """,
        margin=dict(l=65, r=35, t=55, b=60),
    )
    baseline_chart = write_chart(
        "baseline_cfg.html",
        baseline_content,
        "Baseline CFG sweep",
        forward_wheel_to_parent=True,
    )
    baseline_quality_chart = write_chart(
        "baseline_quality_cfg.html",
        baseline_quality_content,
        "SD1.5 quality by CFG",
        forward_wheel_to_parent=True,
    )
    baseline_guided_cfg_chart = write_chart(
        "baseline_vs_guided_cfg.html",
        baseline_guided_cfg_content,
        "Baseline vs guided distribution by CFG",
        forward_wheel_to_parent=True,
    )
    alignment_chart = write_chart(
        "alignment_parameter_space.html",
        alignment_content,
        "Alignment parameter space",
    )
    quality_chart = write_chart(
        "quality_parameter_space.html",
        quality_content,
        "Quality parameter space",
    )
    quality_alignment_chart = write_chart(
        "quality_vs_alignment.html",
        quality_alignment_content,
        "Quality vs alignment",
        forward_wheel_to_parent=True,
    )
    trajectories_chart = write_chart(
        "vlm_loss_trajectories.html",
        trajectory_hover_content,
        "VLM loss trajectories",
        forward_wheel_to_parent=True,
    )
    all_mean_trajectories_chart = write_chart(
        "all_mean_vlm_loss_trajectories.html",
        all_mean_trajectory_content,
        "Mean VLM loss trajectories for all guided pipelines",
        forward_wheel_to_parent=True,
    )
    final_loss_alignment_chart = write_chart(
        "final_loss_vs_alignment.html",
        final_loss_alignment_content,
        "Final VLM loss vs alignment",
        forward_wheel_to_parent=True,
    )
    final_vlm_loss_by_cfg_chart = write_chart(
        "final_vlm_loss_by_cfg.html",
        final_vlm_loss_by_cfg_content,
        "Mean final VLM loss by CFG",
        forward_wheel_to_parent=True,
    )
    comparison_chart = write_chart(
        "alignment_delta.html",
        comparison_hover_content,
        "Per-prompt alignment change",
        forward_wheel_to_parent=True,
    )
    figures = [
        figure(
            1,
            "fig-guided-pipeline",
            f'<img class="full-image pipeline-image" src="{pipeline_uri}?v=wave5-cfg11" alt="Схема VLM-guided генерации">',
            "Схема VLM-guided генерации. Латент обновляется по градиенту VLM loss, после чего изображение повторно генерируется с тем же закэшированным DDPM noise.",
        ),
        figure(
            2,
            "fig-baseline-cfg",
            chart_embed(baseline_chart, "Baseline CFG sweep", 415),
            "Зависимость средней семантической согласованности стандартного SD1.5 от CFG scale. Звездой отмечена выбранная baseline-конфигурация.",
        ),
        figure(
            3,
            "fig-alignment-space",
            chart_iframe(alignment_chart, "Alignment parameter space", 650),
            "Средний alignment guided SD1.5 в пространстве learning rate, CFG scale и номера шага guidance.",
        ),
        figure(
            4,
            "fig-alignment-slices",
            f'<img class="full-image" src="{slices_uri}" alt="Срезы пространства параметров">',
            "Двумерные срезы пространства параметров. Белая линия соответствует baseline, звезда — выбранной для отчёта guided-конфигурации.",
        ),
        figure(
            5,
            "fig-quality-space",
            chart_iframe(quality_chart, "Quality parameter space", 650),
            "Среднее визуальное качество guided SD1.5 в исследованном пространстве параметров.",
        ),
        figure(
            6,
            "fig-vlm-trajectories",
            chart_embed(trajectories_chart, "VLM loss trajectories", 650),
            "Изменение VLM loss по итерациям guidance. Тонкие линии соответствуют отдельным prompts, жирная линия — среднему значению и отрисована поверх отдельных траекторий. Наведение на траекторию показывает соответствующую последовательность изображений. Ось VLM loss логарифмическая.",
        ),
        figure(
            7,
            "fig-intermediate-images",
            f'<div class="image-grid sequence">{"".join(intermediate_cards)}</div>',
            "Промежуточные изображения одного sample при последовательной оптимизации латента. Все повторные rollout используют закэшированный DDPM noise.",
        ),
        figure(
            8,
            "fig-alignment-delta",
            chart_embed(comparison_chart, "Per-prompt alignment change", 900),
            "Попарное изменение alignment относительно выбранного baseline для каждого prompt из набора WHOOPS. Наведение на столбец обновляет пару baseline/guided изображений справа; щелчок фиксирует выбранный sample.",
        ),
        figure(
            9,
            "fig-paired-examples",
            f'<div class="pair-grid">{"".join(pair_cards)}</div>',
            "Примеры наибольшего снижения и роста alignment. Для каждого prompt показаны baseline и guided изображения.",
        ),
    ]
    all_mean_trajectories_figure = figure(
        "6б",
        "fig-all-mean-vlm-trajectories",
        chart_embed(
            all_mean_trajectories_chart,
            "Mean VLM loss trajectories for all guided pipelines",
            650,
        ),
        "Средние траектории VLM loss для всех исследованных guided-конфигураций. Каждая линия усреднена по prompts одного pipeline; цвет кодирует CFG scale. Выбранный лучший запуск показан тёмной пунктирной линией с белыми маркерами и нарисован поверх остальных. Чёрная сплошная линия показывает среднее значение по всем pipeline. Ось VLM loss логарифмическая.",
    )
    quality_alignment_figure = figure(
        "5б",
        "fig-quality-alignment",
        chart_embed(
            quality_alignment_chart,
            "Quality vs alignment",
            620,
        ),
        "Соотношение среднего quality и среднего alignment для всех исследованных запусков. Все обычные запуски показаны кругами, а цвет кодирует CFG scale для guided и vanilla SD1.5. Звёздами соответствующих CFG-цветов отмечены выбранные guided- и baseline-конфигурации.",
    )
    final_loss_alignment_figure = figure(
        "6в",
        "fig-final-loss-alignment",
        chart_embed(
            final_loss_alignment_chart,
            "Final VLM loss vs alignment",
            620,
        ),
        "Связь среднего финального VLM loss со средним alignment для всех исследованных guided-конфигураций. Каждая точка соответствует одному pipeline; звездой отмечена выбранная для подробного анализа конфигурация.",
    )

    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Wave 5 — VLM-guided image generation</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #ffffff;
      --bg-soft: #f6f6f6;
      --panel: #ffffff;
      --panel-2: #f5f7fa;
      --line: #b8b8b8;
      --text: #111111;
      --muted: #444444;
      --heading: #000000;
      --blue: #164f91;
      --cyan: #176b75;
      --green: #21845d;
      --orange: #a76600;
      --red: #b83b4a;
      --max-width: 760px;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 16px/1.28 "Times New Roman", Times, serif;
    }}
    a {{ color: var(--blue); text-decoration: none; }}
    a:hover {{ color: #9ccaff; text-decoration: underline; }}
    .layout {{
      width: min(var(--max-width), calc(100% - 40px));
      margin: 0 auto;
    }}
    header {{
      padding: 36px 0 28px;
    }}
    .eyebrow {{
      padding-bottom: 4px;
      border-bottom: 1px solid #111;
      color: #111;
      letter-spacing: 0;
      font-size: 1rem;
      font-weight: 400;
    }}
    h1, h2, h3, h4 {{ color: var(--heading); font-family: "Times New Roman", Times, serif; }}
    h1 {{
      max-width: 720px;
      margin: 70px 0 34px;
      font-size: 2rem;
      font-weight: 400;
      line-height: 1.08;
      letter-spacing: .035em;
      text-transform: uppercase;
    }}
    h2 {{
      margin: 2.1rem 0 .7rem;
      font-size: 1rem;
      font-weight: 400;
      letter-spacing: .035em;
      text-transform: uppercase;
    }}
    h3 {{ margin: 1.35rem 0 .45rem; font-size: 1rem; }}
    .authors {{ margin: 0 0 10px; text-align: center; font-weight: 700; line-height: 1.55; }}
    .affiliations {{ margin: 0; text-align: center; }}
    main section {{ padding: 0; border: 0; }}
    p {{ margin: .55rem 0; text-align: justify; }}
    #abstract {{ margin: 48px auto 36px; max-width: 650px; }}
    #abstract h2 {{ text-align: center; }}
    .abstract, .note {{
      padding: 0;
      border: 0;
      border-radius: 0;
      background: transparent;
    }}
    .code-link {{
      margin-top: 1rem;
      padding: .7rem .85rem;
      border-left: 4px solid var(--blue);
      background: linear-gradient(135deg, #fbfcfe 0%, #f4f7fb 100%);
      color: #26364d;
      text-align: left;
    }}
    .code-link a {{
      display: inline-flex;
      align-items: center;
      gap: .35rem;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    .code-link svg {{
      width: 1.05rem;
      height: 1.05rem;
      flex: 0 0 auto;
      fill: currentColor;
    }}
    .note {{ margin: 1rem 0; font-size: .93rem; }}
    .report-figure {{
      width: min(1080px, calc(100vw - 36px));
      margin: 30px 50% 38px;
      transform: translateX(-50%);
      scroll-margin-top: 20px;
    }}
    .figure-frame {{
      overflow: visible;
      padding: 0;
      border: 0;
      border-radius: 0;
      background: transparent;
      box-shadow: none;
    }}
    .figure-frame > .plotly-graph-div {{ min-height: 590px; }}
    .chart-frame {{
      display: block;
      width: 100%;
      border: 1.5px solid {CHART_BORDER};
      border-radius: 18px;
      background: {CHART_BACKGROUND};
      box-shadow: 0 5px 16px rgba(27, 39, 64, .10);
    }}
    .embedded-chart {{
      width: 100%;
      overflow: hidden;
      border: 1.5px solid {CHART_BORDER};
      border-radius: 18px;
      background: {CHART_BACKGROUND};
      box-shadow: 0 5px 16px rgba(27, 39, 64, .10);
    }}
    .embedded-chart .hover-info-bar {{
      min-height: 44px;
      padding: 6px 75px 2px;
      color: #334155;
      font-size: 14px;
      line-height: 1.3;
    }}
    .embedded-chart-error {{
      padding: 20px;
      color: #8a1f2d;
      font-family: Arial, sans-serif;
    }}
    .report-figure > figcaption {{
      max-width: 900px;
      margin: 8px auto 0;
      color: #111;
      font-size: .88rem;
      line-height: 1.25;
      text-align: justify;
    }}
    .report-figure > figcaption strong {{ color: #111; }}
    .full-image {{ display: block; width: 100%; border-radius: 0; }}
    .pipeline-image {{ max-width: 1000px; margin: 0 auto; }}
    .image-grid {{ display: grid; gap: 12px; }}
    .image-grid.sequence {{ grid-template-columns: repeat(5, minmax(0, 1fr)); }}
    .image-grid.two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .image-card {{ min-width: 0; margin: 0; }}
    .image-card img {{
      display: block;
      width: 100%;
      aspect-ratio: 1;
      object-fit: cover;
      border: 0;
      border-radius: 0;
    }}
    .image-card figcaption {{
      display: flex;
      flex-direction: column;
      margin-top: 8px;
      color: var(--muted);
      font-size: .78rem;
      line-height: 1.4;
    }}
    .image-card figcaption strong {{ color: var(--text); }}
    .pair-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .pair-card {{ padding: 8px; border: 0; border-radius: 0; background: transparent; }}
    .pair-card h4 {{ min-height: 3em; margin: 0 0 4px; font-size: .95rem; }}
    .pair-card .delta {{ margin: 0 0 12px; color: var(--orange); font-size: .86rem; }}
    .trajectory-layout {{
      display: block;
    }}
    .trajectory-plot .plotly-graph-div {{ min-height: 350px; }}
    .trajectory-panel {{
      position: static;
      padding-top: 8px;
      min-width: 0;
      font-size: .78rem;
      line-height: 1.25;
    }}
    .trajectory-panel > strong {{
      display: block;
      min-height: 0;
      margin-bottom: 8px;
      overflow-wrap: anywhere;
    }}
    .trajectory-images {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
    }}
    .trajectory-images figure {{ margin: 0; min-width: 0; }}
    .trajectory-images img {{
      display: block;
      width: 100%;
      aspect-ratio: 1;
      object-fit: cover;
    }}
    .trajectory-images figcaption {{ margin-top: 3px; text-align: left; }}
    .comparison-layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 350px;
      gap: 18px;
      align-items: start;
    }}
    .comparison-panel {{
      position: sticky;
      top: 12px;
      padding-top: 22px;
    }}
    .comparison-panel-heading {{
      min-height: 92px;
      margin-bottom: 10px;
      font-size: .88rem;
      line-height: 1.25;
    }}
    .comparison-panel-heading strong {{
      display: block;
      margin-bottom: 7px;
      font-size: 1rem;
    }}
    .comparison-panel-heading span {{ display: block; }}
    .comparison-images {{ display: grid; gap: 15px; }}
    .comparison-images figure {{ margin: 0; }}
    .comparison-image-label {{
      margin-bottom: 5px;
      padding: 5px 8px;
      border-left: 4px solid #8a94a3;
      background: #f2f3f5;
      color: #273142;
      font: 700 .76rem/1.2 Arial, sans-serif;
      letter-spacing: .045em;
    }}
    .guided-example .comparison-image-label {{
      border-left-color: #2f78d0;
      background: #edf5ff;
      color: #174f91;
    }}
    .comparison-images img {{
      display: block;
      width: 100%;
      aspect-ratio: 1;
      object-fit: cover;
    }}
    .comparison-images figcaption {{
      margin-top: 5px;
      font-size: .78rem;
      line-height: 1.2;
      text-align: left;
    }}
    code, pre {{ font-family: "SFMono-Regular", Consolas, monospace; }}
    code {{ color: #174f91; }}
    pre {{
      overflow-x: auto;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 0;
      background: #f5f7fa;
      color: #26364d;
      font-size: .78rem;
      line-height: 1.35;
    }}
    .todo-list li {{ margin: .65rem 0; padding-left: .3rem; }}
    footer {{ padding: 30px 0 50px; color: var(--muted); font-size: .8rem; text-align: center; }}
    .footer-content {{ display: flex; flex-direction: column; align-items: center; gap: 8px; }}
    .telegram-link {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      color: #246b9e;
      font-weight: 700;
      text-decoration: none;
    }}
    .telegram-link:hover {{ text-decoration: underline; }}
    .telegram-link svg {{ width: 17px; height: 17px; fill: currentColor; }}
    @media (max-width: 850px) {{
      .image-grid.sequence {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .figure-frame > .plotly-graph-div {{ min-height: 520px; }}
      .trajectory-images {{ grid-template-columns: repeat(5, minmax(0, 1fr)); }}
      .comparison-layout {{ grid-template-columns: 1fr; }}
      .comparison-panel {{ position: static; padding-top: 0; }}
      .comparison-images {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 560px) {{
      .layout {{ width: min(100% - 24px, var(--max-width)); }}
      header {{ padding-top: 24px; }}
      .trajectory-images {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .pair-grid {{ grid-template-columns: 1fr; }}
    }}
    @media print {{
      .report-figure, .pair-card {{ break-inside: avoid; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="layout">
      <div class="eyebrow">Wave 5 research report</div>
      <h1>VLM-guided generation with Stable Diffusion 1.5</h1>
      <p class="authors">[Имя автора]<sup>1</sup> &nbsp;&nbsp; [Имя соавтора]<sup>2</sup></p>
      <p class="affiliations"><sup>1</sup>[Организация] &nbsp;&nbsp; <sup>2</sup>[Организация]</p>
    </div>
  </header>

  <main class="layout">
    <section id="abstract">
      <h2>Аннотация</h2>
      <div class="abstract">
        <p><strong>Цель работы.</strong> [Сформулируйте исследовательскую цель и проверяемую гипотезу.]</p>
        <p><strong>Метод.</strong> [Кратко опишите экспериментальный протокол, модели и данные.]</p>
        <p><strong>Основной результат.</strong> [Сформулируйте главный количественный вывод и сошлитесь на соответствующую фигуру.]</p>
        <p class="code-link"><strong>Код и воспроизводимость:</strong> <a href="https://github.com/AlexKarachun/through_vlm_guidance_research" target="_blank" rel="noopener noreferrer"><svg viewBox="0 0 24 24" role="img" aria-label="GitHub"><path d="M12 .7a11.5 11.5 0 0 0-3.64 22.41c.58.11.79-.25.79-.56v-2.23c-3.22.7-3.9-1.37-3.9-1.37-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.78 2.72 1.27 3.38.97.1-.75.41-1.27.74-1.56-2.57-.29-5.27-1.29-5.27-5.68 0-1.25.45-2.28 1.19-3.08-.12-.29-.52-1.46.11-3.04 0 0 .97-.31 3.16 1.18a10.95 10.95 0 0 1 5.76 0c2.19-1.49 3.16-1.18 3.16-1.18.63 1.58.23 2.75.11 3.04.74.8 1.19 1.83 1.19 3.08 0 4.4-2.71 5.38-5.29 5.67.42.36.79 1.06.79 2.14v3.18c0 .31.21.68.8.56A11.5 11.5 0 0 0 12 .7Z"/></svg><span>AlexKarachun/through_vlm_guidance_research</span></a></p>
      </div>
    </section>

    <section id="method">
      <h2>1. Метод</h2>
      <p>В прошлых отчётах мы уже подробно разбирали общую идею VLM guidance, поэтому здесь напомню её в двух словах. Мы хотим подкрутить промежуточный латент так, чтобы итоговая картинка лучше соответствовала промпту. Для этого показываем результат генерации VLM, получаем вероятности ответов <i>p</i><sub>yes</sub> и <i>p</i><sub>no</sub> и считаем loss: ℒ<sub>VLM</sub> = softplus[−(<i>p</i><sub>yes</sub> − <i>p</i><sub>no</sub>)]. После этого делаем градиентный шаг по латенту <i>X</i><sub>t</sub>.</p>
      <p>Главное отличие текущего варианта пайплайна в том, как именно мы получаем картинку для VLM. Раньше мы строили приближённую оценку финального изображения напрямую из текущего латента. Теперь из <i>X</i><sub>t</sub> выполняем весь оставшийся denoise-проход, получаем полноценную картинку и протягиваем градиент назад через всю эту траекторию. Обновление имеет обычный вид: <i>X</i>′<sub>t</sub> = <i>X</i><sub>t</sub> − η∇<sub><i>X</i><sub>t</sub></sub>ℒ<sub>VLM</sub>.</p>
      <p>После обновления ещё раз запускаем denoise из <i>X</i>′<sub>t</sub>. Чтобы сравнение было честным, во всех повторных проходах используем тот же закэшированный DDPM noise: тогда изменения в картинке связаны именно с guidance, а не с новой случайностью. Этот цикл можно повторить несколько раз. Общая схема показана на <a href="#fig-guided-pipeline">рис. 1</a>.</p>
      {figures[0]}
      <div class="note"><strong>Пример перекрёстной ссылки.</strong> Поведение VLM-сигнала во время последовательных обновлений показано на <a href="#fig-vlm-trajectories">рис. 6</a>, а соответствующие изображения — на <a href="#fig-intermediate-images">рис. 7</a>.</div>
    </section>

    <section id="results">
      <h2>2. Экспериментальные результаты</h2>
      <p>[Здесь приведите краткую сводку основных численных результатов перед подробным анализом фигур.]</p>
      <h3>2.1. Выбор baseline</h3>
      <p>Сначала была исследована зависимость alignment стандартного SD1.5 от CFG scale (<a href="#fig-baseline-cfg">рис. 2</a>). Конфигурация с наибольшим средним alignment использована далее как baseline.</p>
      {figures[1]}
      <h3>2.2. Поиск параметров guided generation</h3>
      <p>Guided-конфигурации сравнивались по learning rate, CFG scale и позиции guidance. Полное пространство результатов приведено на <a href="#fig-alignment-space">рис. 3</a>; его двумерные срезы представлены на <a href="#fig-alignment-slices">рис. 4</a>.</p>
      {figures[2]}
      {figures[3]}
      <h3>2.3. Визуальное качество</h3>
      <p>Поскольку рост alignment может сопровождаться деградацией изображения, quality анализировался как отдельная метрика (<a href="#fig-quality-space">рис. 5</a>). Для подробного анализа зафиксирована конфигурация <code>{best_setup}</code>.</p>
      {figures[4]}
      {quality_alignment_figure}
      <h3>2.4. Динамика VLM guidance</h3>
      <p>Траектории на <a href="#fig-vlm-trajectories">рис. 6</a> характеризуют изменение VLM loss, восстановленного из сохранённых суммарных вероятностей Yes- и No-токенов.</p>
      {figures[5]}
      {all_mean_trajectories_figure}
      {final_loss_alignment_figure}
    </section>

    <section id="examples">
      <h2>3. Качественный анализ</h2>
      <p>Последовательность промежуточных изображений позволяет визуально проверить, какие элементы сцены меняются вследствие обновления латента (<a href="#fig-intermediate-images">рис. 7</a>). Попарное распределение выигрышей и проигрышей относительно baseline показано на <a href="#fig-alignment-delta">рис. 8</a>.</p>
      {figures[6]}
      {figures[7]}
      {figures[8]}
    </section>

    <section id="summary">
      <h2>4. Итог</h2>
      <p>В этой работе мы продемонстрировали trade-off между quality и alignment, показали информативность выбранного VLM loss и обнаружили связь quality, alignment и финального VLM loss с CFG scale.</p>
      <p>В то же время динамика оптимизации латентов в guided-пайплайне оказалась неудовлетворительной. VLM loss снижается слабо и немонотонно, траектории осциллируют. Мы рассматриваем четыре возможные причины:</p>
      <ol>
        <li>Для оптимизации недостаточно четырёх guidance-итераций.</li>
        <li>Learning rate оказался слишком большим.</li>
        <li>При backward через траекторию генерации происходит взрыв градиентов.</li>
        <li>VLM-guided генерация в нашей реализации в принципе нежизнеспособна.</li>
      </ol>
      <p>Первые три гипотезы мы проверим в следующем спринте: увеличим число итераций, уменьшим learning rate и отдельно проследим за градиентами.</p>
      <p>Несмотря на слабую динамику оптимизации, для каждого рассмотренного CFG scale мы нашли guided-конфигурации с alignment не ниже, чем у vanilla SD1.5. В целом нам удалось получить некоторое улучшение верности генерации, однако идея пока не раскрылась полностью и требует дальнейшей доработки.</p>
    </section>

    <section id="authoring">
      <h2>5. Как дополнять отчёт</h2>
      <p>Обычный научный абзац оформляется тегом <code>&lt;p&gt;</code>. Заголовки разделов используют <code>&lt;h2&gt;</code>, подразделов — <code>&lt;h3&gt;</code>. Каждой фигуре присваивается стабильный <code>id</code>, на который ведёт ссылка.</p>
      <pre>&lt;h2&gt;5. Новый раздел&lt;/h2&gt;
&lt;h3&gt;5.1. Новый эксперимент&lt;/h3&gt;
&lt;p&gt;
  Изменение метрики представлено на
  &lt;a href="#fig-new-result"&gt;рис. 9&lt;/a&gt;.
&lt;/p&gt;

&lt;figure class="report-figure" id="fig-new-result"&gt;
  &lt;div class="figure-frame"&gt;
    &lt;!-- Plotly div, SVG, canvas или img --&gt;
  &lt;/div&gt;
  &lt;figcaption&gt;
    &lt;strong&gt;Рисунок 9.&lt;/strong&gt;
    Полная содержательная подпись к рисунку.
  &lt;/figcaption&gt;
&lt;/figure&gt;</pre>
      <p>Чтобы сменить анализируемый guided pipeline, измените <code>REPORT_GUIDED_SETUP</code> в <code>generate_wave_5_report.py</code> и повторно запустите генератор. HTML и публикационная папка с изображениями будут обновлены автоматически.</p>
    </section>
  </main>

  <footer><div class="layout footer-content">
    <a class="telegram-link" href="https://t.me/Alex_Karachun" target="_blank" rel="noopener noreferrer">
      <svg viewBox="0 0 24 24" role="img" aria-label="Telegram"><path d="M23.91 3.79 20.3 20.84c-.27 1.2-.98 1.49-1.99.93l-5.5-4.05-2.65 2.55c-.29.29-.54.54-1.11.54l.39-5.6L19.65 6c.45-.4-.1-.62-.7-.22L6.33 13.73.9 12.03c-1.18-.37-1.2-1.18.25-1.75L22.4 2.09c.98-.36 1.84.22 1.51 1.7Z"/></svg>
      <span>t.me/Alex_Karachun</span>
    </a>
    <span>При создании графиков использовалась LLM.</span>
  </div></footer>
  <script src="wave_5_report_assets/js/plotly.min.js"></script>
  <script>
    async function loadEmbeddedChart(container) {{
      const source = container.dataset.chartSrc;
      try {{
        const response = await fetch(source);
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
        const chartDocument = new DOMParser().parseFromString(
          await response.text(),
          "text/html"
        );
        const scripts = Array.from(chartDocument.body.querySelectorAll("script"))
          .map(script => ({{ src: script.getAttribute("src"), code: script.textContent }}));
        chartDocument.body.querySelectorAll("script").forEach(script => script.remove());
        container.replaceChildren(
          ...Array.from(chartDocument.body.childNodes).map(node =>
            document.importNode(node, true)
          )
        );
        for (const scriptData of scripts) {{
          if (scriptData.src || scriptData.code.includes("wave-1-report-scroll")) continue;
          const script = document.createElement("script");
          script.textContent = scriptData.code;
          container.appendChild(script);
        }}
      }} catch (error) {{
        container.innerHTML =
          `<div class="embedded-chart-error">Не удалось загрузить график: ${{source}}</div>`;
        console.error(`Failed to load chart ${{source}}`, error);
      }}
    }}

    document.querySelectorAll(".embedded-chart[data-chart-src]")
      .forEach(container => loadEmbeddedChart(container));
  </script>
</body>
</html>
"""
    if "--preserve-report" in sys.argv:
        print(f"Preserved {OUTPUT_PATH}; regenerated charts and assets only")
    else:
        OUTPUT_PATH.write_text(html, encoding="utf-8")
        print(f"Wrote {OUTPUT_PATH}")
    print(f"Baseline: {baseline_setup} (alignment={baseline_alignment:.3f})")
    print(
        f"Report guided setup: {best_setup} "
        f"(alignment={best['alignment_mean']:.3f}, quality={best['quality_mean']:.3f})"
    )


if __name__ == "__main__":
    main()
