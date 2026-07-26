# Handoff for VLM-guided image generation research

This is the primary handoff for future Codex conversations in this repository.
Read it completely before changing files. Communicate with the user in Russian.

## Start every new task here

Before making changes:

1. Read this file completely.
2. Read the following files in order:
   1. `README.md`
   2. `run.py`
   3. `scripts/wave_5_run.sh`
   4. `configs/config.yaml`
   5. `configs/pipeline/guided_sd15.yaml`
   6. `configs/pipeline/qwen3_judge.yaml`
   7. `configs/vlm_loss/qwen3_loss.yaml`
   8. `modules/guided_sd15/main.py`
   9. `modules/guided_sd15/pipeline.py`
   10. `modules/qwen3_loss/main.py`
   11. `modules/qwen3_judge/main.py`
   12. `modules/utils.py`
   13. `experiments/wave_5/external/plots.ipynb`
   14. `experiments/wave_5/external/generate_wave_5_report.py`
   15. `experiments/wave_5/external/wave_5_report.html`
3. Run `git status --short` and inspect relevant diffs before editing.
4. Assume uncommitted and untracked files are intentional user work. Never
   reset, discard, overwrite, or broadly clean the worktree.

For a small, clearly isolated request, still inspect the relevant files and
status; do not blindly regenerate artifacts.

## What the user is working on

The project studies whether a differentiable vision-language-model signal can
improve prompt faithfulness during Stable Diffusion 1.5 generation. The current
finished experimental/reporting milestone is Wave 5.

The user is simultaneously:

- developing and evaluating the guided diffusion pipeline;
- running large GPU experiments on a server;
- comparing 8B and 32B Qwen3-VL judges;
- analysing results in a Plotly/Dash notebook;
- maintaining a hand-edited scientific HTML report for GitHub Pages;
- maintaining a concise repository README and a pipeline diagram.

The next research direction recorded in the report is to test more guidance
iterations, smaller learning rates, and possible gradient explosion.

## Guided pipeline

The current guided pipeline:

1. Starts from a diffusion latent and denoises it.
2. Stores the selected intermediate latent `X_t`.
3. Completes the remaining denoising trajectory to obtain an image.
4. Sends the image and prompt to Qwen3-VL with a Yes/No question.
5. Computes a differentiable loss:

   `L_VLM = softplus(-(p_yes - p_no))`

6. Backpropagates through the VLM signal, decoder, and remaining diffusion
   trajectory to obtain the gradient with respect to `X_t`.
7. Updates `X_t` and regenerates from the updated latent.
8. Reuses cached DDPM noise, so differences are caused by guidance rather than
   a newly sampled stochastic trajectory.

Important distinctions:

- Differentiable guidance uses `modules.qwen3_loss.Qwen3Loss` with
  `Qwen/Qwen3-VL-2B-Instruct`.
- Judge scores are independent evaluation results, not the differentiable VLM
  loss.
- The final report uses judgments from
  `experiments/wave_5/generations/judgment-32b.csv`, produced with
  `Qwen/Qwen3-VL-32B-Instruct`.
- The notebook also loads `judgment-8b.csv` for judge comparison.
- Gradient checkpointing is enabled for guided generation.
- Wave 5 uses 50 denoising steps and four guidance iterations at the same
  selected timestep.

## Important completed data fix

Originally `modules/guided_sd15/pipeline.py` did not evaluate the
Yes/No distribution after the final latent update. It now evaluates the final
decoded float tensor before conversion to `uint8`/NumPy and stores a key such
as:

```text
g004-final-after_update
```

A temporary backfill was already run successfully for all 9,250 guided samples.
Do not rerun it unless the user explicitly requests it: it is expensive.
Backfilled values were computed from saved PNGs, which are necessarily uint8
reconstructions of the original float outputs.

## Wave 5 data and selected runs

Generated data is under:

```text
experiments/wave_5/generations/
```

The WHOOPS subset contains 50 prompts.

Selected vanilla baseline:

```text
multi_prompt_sd15-whoops50-cfg13
```

Selected guided run used by the final report:

```text
multi_prompt_guided_sd15-whoops50-guide_steps0000-lr2e-4-cfg11
```

Current 32B-judge aggregates used in the report:

- baseline: mean alignment `3.76`, mean quality `3.62`;
- selected guided run: mean alignment `3.84`, mean quality `3.56`.

The report generator pins the selected guided run through
`REPORT_GUIDED_SETUP`. Do not silently replace it with a newly observed run.
Confirm data and user intent first.

## Notebook

`experiments/wave_5/external/plots.ipynb` is exploratory analysis, not the
canonical website.

It contains Plotly and Dash applications for:

- baseline and guided parameter sweeps;
- `p_yes`, `p_no`, and VLM-loss trajectories;
- final loss/alignment and final `p_yes`/alignment plots;
- quality/alignment analysis;
- gradient-value distributions from `nablas`;
- comparison of Qwen3-VL-8B and Qwen3-VL-32B judge scores and images.

Notebook conventions and pitfalls:

- The main dataframe name is `params_df`, not `guided_params`.
- Dash applications use `find_available_port(...)`; preferred ports start at
  8050 through 8055 but automatically fall back when busy.
- Do not hard-code a port or kill an unrelated notebook server.
- `judge_comparison_app` must update both plotted values and images on hover.
- Avoid executing the entire notebook during validation because its Dash
  servers block and may conflict with existing sessions.
- Parse notebook JSON and execute only affected computational cells in a
  prepared namespace when practical.

## Execution environments

Local editing currently occurs on macOS at:

```text
/Users/alexkarachun/Documents/DEV/through_vlm_guidance_research_dev
```

The server-side conda environment used for notebook/report Python is:

```text
/opt/miniconda3/envs/through_guidance/bin/python
```

Typical setup:

```bash
conda env create -f environment.yml
conda activate through_guidance
```

Runs use Hydra through `run.py`, usually with:

```bash
HYDRA_FULL_ERROR=1 python run.py ...
```

Approximate measured VRAM:

- SD1.5: 11 GB;
- guided SD1.5 with Qwen3-VL-2B loss: 23 GB;
- Qwen3-VL-8B judge: 18 GB;
- Qwen3-VL-32B judge: 64 GB.

Hardware used by the author includes RTX 3090 24 GB, RTX 4090 24 GB,
RTX 5090 32 GB, and RTX PRO 6000 96 GB. CUDA/server details can differ from
the local macOS environment; do not assume local GPU availability.

## Final HTML report: preserve manual work

The canonical report is:

```text
experiments/wave_5/external/wave_5_report.html
```

The user edits this file manually. Never overwrite or fully regenerate it
without explicit permission. To rebuild chart files and assets safely, use:

```bash
/opt/miniconda3/envs/through_guidance/bin/python \
  experiments/wave_5/external/generate_wave_5_report.py \
  --preserve-report
```

This must print that the report was preserved. If it does not, stop and inspect
before proceeding.

When report charts change:

- update `generate_wave_5_report.py`;
- rebuild with `--preserve-report`;
- verify the generated chart file;
- make only narrowly scoped manual edits to `wave_5_report.html`;
- never assume the generator's full-report template is the canonical prose.

The report style is a simple one-column scientific article for a friendly
scientific supervisor. It should remain clear and scientific but not
over-formal, corporate, or marketing-like.

## Final report structure

The report currently has:

- unnumbered abstract;
- section 1, introduction/method/setup, subsections 1.1 through 1.5;
- section 2, observations, subsections 2.1 through 2.5;
- section 3, generation examples;
- section 4, conclusion;
- figures numbered consecutively from 1 through 12.

Figure mapping:

1. guided-generation pipeline diagram;
2. vanilla alignment by CFG;
3. guided alignment parameter space;
4. guided quality parameter space;
5. vanilla quality by CFG;
6. quality versus alignment;
7. final VLM loss versus alignment;
8. mean final VLM loss by CFG;
9. per-prompt VLM-loss trajectories for the selected guided run;
10. mean VLM-loss trajectories for all guided pipelines;
11. vanilla alignment and guided alignment range by CFG;
12. per-prompt baseline/guided alignment difference with image comparison.

The conclusion intentionally keeps the user's direct wording and includes a
numbered list of four hypotheses. There is no separate plans section.

The footer contains only:

- a Telegram link to `https://t.me/Alex_Karachun` with a small logo;
- a note that an LLM was used to create the graphs.

## Chart architecture

Charts live in:

```text
experiments/wave_5/external/wave_5_report_charts/
```

Assets and the local Plotly bundle live in:

```text
experiments/wave_5/external/wave_5_report_assets/
```

Published report images use WebP to keep the GitHub Pages bundle compact:

- generated images use WebP quality 92 with method 6;
- diagrams and plot-like assets use lossless WebP;
- original PNG experiment outputs remain untouched under
  `experiments/wave_5/generations/` and `figures/`.

Keep this behavior in `publish_image(...)`; do not revert published copies to
PNG without a specific reason.

All 11 current chart files are used by the website. Do not delete either 3D
chart merely because it is not referenced by `data-chart-src`: the two 3D
charts are iframes.

Embedded 2D charts:

- `baseline_cfg.html`
- `baseline_quality_cfg.html`
- `quality_vs_alignment.html`
- `final_loss_vs_alignment.html`
- `final_vlm_loss_by_cfg.html`
- `vlm_loss_trajectories.html`
- `all_mean_vlm_loss_trajectories.html`
- `baseline_vs_guided_cfg.html`
- `alignment_delta.html`

Iframe-based 3D charts:

- `alignment_parameter_space.html`
- `quality_parameter_space.html`

For 2D charts:

- `scrollZoom: false`;
- normal page/trackpad scrolling must work over the chart;
- do not restore synthetic wheel forwarding;
- hover details belong in a compact information row above the chart, never in
  a Plotly tooltip plaque;
- omit pipeline folder names, `n=...`, and standard deviations from
  user-facing information rows;
- display fractional values with no more than two decimal places;
- draw highlighted stars and important mean trajectories above ordinary data.

The report loads Plotly once from:

```text
wave_5_report_assets/js/plotly.min.js
```

`loadEmbeddedChart(...)` fetches each 2D chart, inserts it into the main DOM,
skips duplicate Plotly sources and obsolete wheel-forwarding code, then runs
the inline chart script.

Because charts are loaded with `fetch`, do not inspect the report through
`file://`. Serve it over HTTP:

```bash
cd experiments/wave_5/external
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/wave_5_report.html
```

If port 8000 is occupied, choose another port; do not stop an unrelated
process.

## Important interactive chart behavior

Figure 9:

- plots per-prompt VLM-loss trajectories for the selected guided run;
- uses a dark custom continuous palette that remains visible on white;
- maps final VLM loss to color logarithmically but shows no colorbar;
- draws a thick black mean line above individual trajectories;
- uses a logarithmic y-axis with `e` notation;
- hover highlights one trajectory, dims the others, updates the information
  row, and displays the corresponding image sequence.

Figure 10:

- plots one mean VLM-loss trajectory per guided pipeline;
- defaults to coloring by CFG scale;
- lets the user recolor by guidance step, learning rate, or CFG scale;
- draws the selected best pipeline last with a small dark dashed line and
  visible markers;
- draws the thick black overall mean above the pipeline trajectories;
- uses incremental hover restyling for performance.

Figure 11:

- shows every guided mean-alignment point for each CFG;
- fills the range between minimum and maximum guided alignment values;
- draws the blue vanilla trajectory above the filled guided range;
- hover must work for every point, including multiple points at the same CFG.

Figure 12:

- has no Plotly tooltip or technical `trace 0` legend;
- updates baseline/guided images and the information row on hover;
- uses a 350 px image column so the stacked images approximately match the
  bar plot height.

## Pipeline diagram and README

The pipeline diagram source and exports are:

```text
figures/guided_generation_pipeline.svg
figures/guided_generation_pipeline.png
figures/guided_generation_pipeline.pdf
figures/candidate_image_before_guidance.png
figures/guided_output_example.png
```

The two chef images now come from the selected best guided configuration:

```text
multi_prompt_guided_sd15-whoops50-guide_steps0000-lr2e-4-cfg11/
047_a_chef_prepares_a_painting/
```

The upper image is the initial rollout
`g000-guide_s0-image_before_update.png`; the lower image is the final
`guided_sd15.png`.

Preserve the diagram's existing typography, box geometry, uppercase `X`,
prompt placement, VLM question, and overall appearance unless the user asks for
a specific visual change. The README and report use a cache-busting query
`?v=wave5-cfg11` for this image.

The current GitHub username is `AlexKarachun`, not `AlexKrachun`.

## Validation

For Python changes:

```bash
/opt/miniconda3/envs/through_guidance/bin/python -m py_compile PATH
git diff --check -- PATH
```

For notebooks:

- parse the notebook as JSON;
- execute only changed cells where possible;
- do not automatically launch every Dash server.

For reports:

- verify unique IDs;
- verify every `href="#fig-..."` target exists;
- verify displayed figure numbers match their targets;
- verify figure numbers are consecutive;
- verify every local `src`, iframe, and `data-chart-src` exists;
- count both embedded charts and iframe charts before declaring files unused;
- inspect through HTTP in a real browser when possible.

The final report audit on 2026-07-26 confirmed:

- sections 1 through 4 and subsections are sequential;
- figures 1 through 12 are sequential;
- all figure cross-references match;
- all 11 chart files are used;
- all local resources exist;
- all chart requests returned HTTP 200; only the optional browser
  `/favicon.ico` request returned 404.

## How to help this user

- Communicate in Russian.
- Lead with the result and keep explanations compact.
- Prefer acting on a clear request over asking unnecessary questions.
- Preserve manual comments and edits, especially in HTML and README.
- Do not invent scientific claims or add prose beyond the user's supplied
  meaning. Rephrase and structure when asked, but keep the user's conclusions.
- Use `alignment`, `quality`, `VLM loss`, `CFG scale`, `learning rate`,
  `guidance`, `baseline`, and `guided` consistently; do not translate terms
  merely for stylistic purity.
- Prefer prompts to raw setup/folder names in user-facing labels.
- Keep report graphs interactive, visually consistent, and readable on white.
- Use information rows above graphs instead of hover plaques.
- Preserve normal page scrolling over 2D plots and useful interaction in 3D.
- When a change may be reverted later, retain the previous implementation or
  make the change easy to undo rather than deleting it permanently.
- When cleaning generated files, prove they are unused before deleting them.
- Mention exactly what was regenerated and whether the main HTML was preserved.

## Git and publication notes

The current local branch is `master`. At the time of this handoff, `origin`
points to:

```text
git@github.com:AlexKarachun/through_vlm_guidance_research_dev.git
```

The public repository used by README and the report is:

```text
https://github.com/AlexKarachun/through_vlm_guidance_research_dev
```

The user explicitly decided not to maintain
`through_vlm_guidance_research`; use only the `_dev` repository and do not
restore links to the repository without that suffix.

The GitHub Pages report path is:

```text
experiments/wave_5/external/wave_5_report.html
```

The deployment bundle needs the main HTML, all report chart files, the local
Plotly bundle, referenced generation images, and the pipeline diagram asset.
Do not add the enormous raw `experiments/wave_5/generations/` directory merely
to publish the report.
