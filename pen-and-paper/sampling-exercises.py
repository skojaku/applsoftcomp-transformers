# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.21.1",
#     "numpy",
#     "pandas",
#     "altair",
#     "pyarrow",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # GPT Sampling Exercises

    Use this notebook to implement and test your solutions for the pen-and-paper exercises.
    Each exercise has a starter cell with `TODO` comments where you write your code,
    followed by test cells that compute the answers you need to fill in on the paper.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import altair as alt
    import marimo as mo

    return alt, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## Exercise 1: Softmax with Temperature

    Softmax converts logits $z = [z_1, \dots, z_V]$ into probabilities:

    $$P(i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

    **Your task:** implement `softmax_with_temperature(z, T)` below, then run the
    test cell to compute and plot the distributions for $T \in \{0.5, 1, 2, 5\}$.
    """)
    return


@app.cell
def _(np):
    def softmax_with_temperature(z, T):
        """Compute softmax of logits z with temperature T.

        Args:
            z: array of logits
            T: temperature (positive float)

        Returns:
            array of probabilities (same shape as z)
        """
        z = np.asarray(z, dtype=float)
        # TODO: implement softmax with temperature
        # Hint: divide logits by T before applying softmax.
        # For numerical stability, subtract the max of (z/T) before exponentiating.
        raise NotImplementedError("Implement softmax_with_temperature")

    return (softmax_with_temperature,)


@app.cell(hide_code=True)
def _(mo, softmax_with_temperature):
    # Interactive temperature slider for live exploration
    try:
        softmax_with_temperature([0], 1.0)
        _ex1_ready = True
    except NotImplementedError:
        _ex1_ready = False

    if _ex1_ready:
        ex1_T_slider = mo.ui.slider(
            0.1, 5.0, value=1.0, step=0.1, label="Temperature T", show_value=True
        )
        mo.vstack([mo.md("### Interactive Explorer"), ex1_T_slider])
    else:
        ex1_T_slider = None
        mo.callout(
            mo.md("Implement `softmax_with_temperature` in the cell above, then re-run."),
            kind="warn",
        )
    return (ex1_T_slider,)


@app.cell(hide_code=True)
def _(alt, ex1_T_slider, mo, np, pd, softmax_with_temperature):
    if ex1_T_slider is None:
        mo.output.replace(mo.md(""))
    else:
        _z = np.array([1.0, 0.0, -1.0])
        _tokens = ["a", "b", "c"]
        _colors = ["#4dabf7", "#51cf66", "#ff6b6b"]

        # --- Live bar chart at current slider T ---
        _T_cur = ex1_T_slider.value
        _probs_cur = softmax_with_temperature(_z, _T_cur)
        _df_bar = pd.DataFrame({"Token": _tokens, "Probability": _probs_cur})
        _bar = (
            alt.Chart(_df_bar)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=60)
            .encode(
                x=alt.X("Token:N", axis=alt.Axis(labelFontSize=14, titleFontSize=14)),
                y=alt.Y(
                    "Probability:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(labelFontSize=12, titleFontSize=14),
                ),
                color=alt.Color(
                    "Token:N",
                    scale=alt.Scale(domain=_tokens, range=_colors),
                    legend=None,
                ),
                tooltip=["Token", alt.Tooltip("Probability:Q", format=".4f")],
            )
            .properties(width=300, height=250, title=f"Distribution at T = {_T_cur:.1f}")
        )
        _bar_text = (
            alt.Chart(_df_bar)
            .mark_text(dy=-12, fontSize=13, fontWeight="bold")
            .encode(
                x="Token:N",
                y="Probability:Q",
                text=alt.Text("Probability:Q", format=".3f"),
            )
        )

        # --- Line chart across all required temperatures ---
        _temperatures = [0.5, 1.0, 2.0, 5.0]
        _rows = []
        for _T in _temperatures:
            _probs = softmax_with_temperature(_z, _T)
            for _tok, _p in zip(_tokens, _probs):
                _rows.append({"Token": _tok, "T": _T, "P(token)": _p})
        _df_line = pd.DataFrame(_rows)

        _line = (
            alt.Chart(_df_line)
            .mark_line(point=alt.OverlayMarkDef(size=80), strokeWidth=2.5)
            .encode(
                x=alt.X("T:Q", title="Temperature T", axis=alt.Axis(labelFontSize=12)),
                y=alt.Y(
                    "P(token):Q",
                    scale=alt.Scale(domain=[0, 1]),
                    title="P(token)",
                    axis=alt.Axis(labelFontSize=12),
                ),
                color=alt.Color(
                    "Token:N",
                    scale=alt.Scale(domain=_tokens, range=_colors),
                ),
                tooltip=[
                    "Token",
                    alt.Tooltip("T:Q", format=".1f"),
                    alt.Tooltip("P(token):Q", format=".4f"),
                ],
            )
            .properties(width=350, height=250, title="P(token) vs Temperature")
        )

        # Vertical rule at current slider position
        _rule = (
            alt.Chart(pd.DataFrame({"T": [_T_cur]}))
            .mark_rule(strokeDash=[4, 4], color="gray", strokeWidth=1.5)
            .encode(x="T:Q")
        )

        # --- Answer table ---
        _ans_rows = []
        for _T in _temperatures:
            _probs = softmax_with_temperature(_z, _T)
            _ans_rows.append(
                {
                    "T": _T,
                    "P(a)": f"{_probs[0]:.4f}",
                    "P(b)": f"{_probs[1]:.4f}",
                    "P(c)": f"{_probs[2]:.4f}",
                }
            )
        _ans_df = pd.DataFrame(_ans_rows)

        mo.vstack(
            [
                mo.hstack(
                    [_bar + _bar_text, _line + _rule],
                    justify="center",
                    gap=1.5,
                ),
                mo.md("**Probability table** (copy these to your paper):"),
                mo.ui.table(_ans_df, selection=None),
            ]
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## Exercise 2: Top-$k$ and Top-$p$ Sampling

    **Top-$k$** keeps the $k$ highest-probability tokens and zeroes out the rest, then renormalizes.

    **Top-$p$** (nucleus) keeps the fewest tokens whose cumulative probability reaches $p$, then renormalizes.

    Logits: $z = [3.0, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0]$

    **Your task:** implement `top_k_sampling` and `top_p_sampling` below.
    """)
    return


@app.cell
def _(np):
    def top_k_sampling(probs, k):
        """Apply top-k filtering to a probability distribution.

        Keep only the k highest-probability tokens. Set the rest to zero.
        Renormalize so the result sums to 1.

        Args:
            probs: array of probabilities (already normalized, sums to 1)
            k: number of top tokens to keep

        Returns:
            array of renormalized probabilities (same shape as probs)
        """
        probs = np.asarray(probs, dtype=float)
        # TODO: implement top-k sampling
        # 1. Find the k-th largest probability (threshold).
        # 2. Zero out all tokens with probability below that threshold.
        # 3. Renormalize.
        raise NotImplementedError("Implement top_k_sampling")

    return (top_k_sampling,)


@app.cell
def _(np):
    def top_p_sampling(probs, p):
        """Apply top-p (nucleus) filtering to a probability distribution.

        Sort tokens by probability (descending). Keep the fewest tokens
        whose cumulative probability is >= p. Zero out the rest. Renormalize.

        Args:
            probs: array of probabilities (already normalized, sums to 1)
            p: cumulative probability threshold

        Returns:
            array of renormalized probabilities (same shape as probs)
        """
        probs = np.asarray(probs, dtype=float)
        # TODO: implement top-p (nucleus) sampling
        # 1. Sort probabilities in descending order.
        # 2. Compute the cumulative sum.
        # 3. Find the smallest set of tokens whose cumulative prob >= p.
        # 4. Zero out all other tokens. Renormalize.
        raise NotImplementedError("Implement top_p_sampling")

    return (top_p_sampling,)


@app.cell(hide_code=True)
def _(mo, softmax_with_temperature, top_k_sampling, top_p_sampling):
    # Check if all functions are implemented
    try:
        softmax_with_temperature([0], 1.0)
        top_k_sampling([0.5, 0.5], 1)
        top_p_sampling([0.5, 0.5], 0.5)
        _ex2_ready = True
    except NotImplementedError:
        _ex2_ready = False

    if _ex2_ready:
        ex2_k_slider = mo.ui.slider(1, 10, value=3, step=1, label="k", show_value=True)
        ex2_p_slider = mo.ui.slider(
            0.1, 1.0, value=0.8, step=0.05, label="p", show_value=True
        )
        ex2_T_slider = mo.ui.slider(
            0.5, 5.0, value=1.0, step=0.5, label="Temperature T", show_value=True
        )
        mo.vstack(
            [
                mo.md("### Interactive Explorer"),
                mo.md("Drag the sliders to see how top-k, top-p, and temperature interact:"),
                mo.hstack([ex2_k_slider, ex2_p_slider, ex2_T_slider]),
            ]
        )
    else:
        ex2_k_slider = None
        ex2_p_slider = None
        ex2_T_slider = None
        mo.callout(
            mo.md(
                "Implement `softmax_with_temperature`, `top_k_sampling`, and `top_p_sampling` above, then re-run."
            ),
            kind="warn",
        )
    return ex2_T_slider, ex2_k_slider, ex2_p_slider


@app.cell(hide_code=True)
def _(
    alt,
    ex2_T_slider,
    ex2_k_slider,
    ex2_p_slider,
    mo,
    np,
    pd,
    softmax_with_temperature,
    top_k_sampling,
    top_p_sampling,
):
    if ex2_k_slider is None:
        mo.output.replace(mo.md(""))
    else:
        _z = np.array([3.0, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
        _labels = [f"t{i}" for i in range(10)]
        _k = ex2_k_slider.value
        _p = ex2_p_slider.value
        _T = ex2_T_slider.value

        _probs = softmax_with_temperature(_z, _T)
        _topk = top_k_sampling(_probs, _k)
        _topp = top_p_sampling(_probs, _p)

        def _make_bar_chart(orig, filtered, title, color_inc, labels):
            rows = []
            for i, (lab, o, f) in enumerate(zip(labels, orig, filtered)):
                rows.append(
                    {
                        "Token": lab,
                        "Original": o,
                        "Filtered": f,
                        "Status": "Included" if f > 0 else "Excluded",
                        "order": i,
                    }
                )
            df = pd.DataFrame(rows)

            base = alt.Chart(df).encode(
                x=alt.X("Token:N", sort=labels, axis=alt.Axis(labelFontSize=12)),
            )

            # Ghost bars for original distribution
            ghost = base.mark_bar(
                color="#e0e0e0", cornerRadiusTopLeft=3, cornerRadiusTopRight=3, size=28
            ).encode(
                y=alt.Y("Original:Q", scale=alt.Scale(domain=[0, max(orig) * 1.15])),
                tooltip=[
                    "Token",
                    alt.Tooltip("Original:Q", format=".4f", title="Original P"),
                ],
            )

            # Colored bars for filtered distribution
            bars = base.mark_bar(
                cornerRadiusTopLeft=3, cornerRadiusTopRight=3, size=18
            ).encode(
                y=alt.Y("Filtered:Q"),
                color=alt.Color(
                    "Status:N",
                    scale=alt.Scale(
                        domain=["Included", "Excluded"],
                        range=[color_inc, "#cccccc"],
                    ),
                    legend=alt.Legend(title=""),
                ),
                tooltip=[
                    "Token",
                    alt.Tooltip("Original:Q", format=".4f", title="Original P"),
                    alt.Tooltip("Filtered:Q", format=".4f", title="Filtered P"),
                    "Status",
                ],
            )

            # Probability labels on included bars
            text = (
                base.transform_filter(alt.datum.Filtered > 0)
                .mark_text(dy=-10, fontSize=10, fontWeight="bold")
                .encode(
                    y="Filtered:Q",
                    text=alt.Text("Filtered:Q", format=".3f"),
                )
            )

            n_inc = int(np.sum(np.array([r["Filtered"] for r in rows]) > 0))
            return (ghost + bars + text).properties(
                width=320, height=220, title=f"{title} ({n_inc} tokens included)"
            )

        _chart_k = _make_bar_chart(
            _probs, _topk, f"Top-k (k={_k})", "#4dabf7", _labels
        )
        _chart_p = _make_bar_chart(
            _probs, _topp, f"Top-p (p={_p:.2f})", "#51cf66", _labels
        )

        mo.vstack(
            [
                mo.hstack([_chart_k, _chart_p], justify="center", gap=1.5),
                mo.md(
                    f"Gray bars show the original distribution at T={_T}. "
                    f"Colored bars show the renormalized probabilities after filtering."
                ),
            ]
        )
    return


@app.cell(hide_code=True)
def _(mo, np, pd, softmax_with_temperature, top_k_sampling, top_p_sampling):
    # --- Fixed answer tables for the paper ---
    _z = np.array([3.0, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
    _labels = [f"t{i}" for i in range(10)]

    try:
        _probs_t1 = softmax_with_temperature(_z, 1.0)
        _topk3 = top_k_sampling(_probs_t1, 3)
        _topp80 = top_p_sampling(_probs_t1, 0.8)
        _n_q2 = int(np.sum(_topp80 > 0))

        _probs_t2 = softmax_with_temperature(_z, 2.0)
        _topp80_t2 = top_p_sampling(_probs_t2, 0.8)
        _n_q3 = int(np.sum(_topp80_t2 > 0))

        _q1_df = pd.DataFrame(
            {"Token": _labels, "Original P": [f"{p:.4f}" for p in _probs_t1], "Top-k=3 P": [f"{p:.4f}" for p in _topk3]}
        )
        _q2_df = pd.DataFrame(
            {"Token": _labels, "Original P": [f"{p:.4f}" for p in _probs_t1], "Top-p=0.8 P": [f"{p:.4f}" for p in _topp80]}
        )
        _q3_df = pd.DataFrame(
            {"Token": _labels, "P (T=2)": [f"{p:.4f}" for p in _probs_t2], "Top-p=0.8 (T=2) P": [f"{p:.4f}" for p in _topp80_t2]}
        )

        mo.vstack(
            [
                mo.md("### Answers for the Paper"),
                mo.md("**Q1: Top-k with k=3** (sampling probabilities):"),
                mo.ui.table(_q1_df, selection=None),
                mo.md(f"**Q2: Top-p with p=0.8 at T=1.** Number of tokens included: **{_n_q2}**"),
                mo.ui.table(_q2_df, selection=None),
                mo.md(
                    f"**Q3: Top-p with p=0.8 at T=2.** Number of tokens included: **{_n_q3}**"
                    + (f" (changed from {_n_q2})" if _n_q2 != _n_q3 else " (same as Q2)")
                ),
                mo.ui.table(_q3_df, selection=None),
            ]
        )
    except NotImplementedError:
        mo.output.replace(mo.md(""))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## Exercise 3: Beam Search

    This exercise is best done by hand, but you can use the code below to verify your answers.

    Vocab: $\{a, b\}$, three steps. Step 3 depends on both previous tokens.

    The log-probability tables are pre-loaded below. Run the greedy and beam search
    cells to check your pen-and-paper work.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Log-probability tables
    step1_logprobs = {"a": -0.51, "b": -0.92}
    step2_logprobs = {
        "a": {"a": -1.20, "b": -0.36},
        "b": {"a": -0.22, "b": -1.61},
    }
    step3_logprobs = {
        ("a", "a"): {"a": -1.61, "b": -0.22},
        ("a", "b"): {"a": -0.92, "b": -0.51},
        ("b", "a"): {"a": -0.11, "b": -2.30},
        ("b", "b"): {"a": -0.92, "b": -0.51},
    }

    mo.md(
        """
        **Log-probability tables (pre-loaded):**

        | Step 1 | ln P |
        |--------|------|
        | a | -0.51 |
        | b | -0.92 |

        | Step 2 (from) | ln P(a) | ln P(b) |
        |---------------|---------|---------|
        | a | -1.20 | -0.36 |
        | b | -0.22 | -1.61 |

        | Step 3 (from) | ln P(a) | ln P(b) |
        |---------------|---------|---------|
        | (a,a) | -1.61 | -0.22 |
        | (a,b) | -0.92 | -0.51 |
        | (b,a) | -0.11 | -2.30 |
        | (b,b) | -0.92 | -0.51 |
        """
    )
    return step1_logprobs, step2_logprobs, step3_logprobs


@app.function
def greedy_search(step1_lp, step2_lp, step3_lp):
    """Perform greedy decoding (B=1).

    At each step, pick the single token with the highest log-probability.

    Args:
        step1_lp: dict mapping token -> log-prob for step 1
        step2_lp: dict mapping prev_token -> {token: log-prob} for step 2
        step3_lp: dict mapping (prev2, prev1) -> {token: log-prob} for step 3

    Returns:
        (sequence_list, cumulative_log_prob)
        e.g. (["a", "b", "b"], -1.38)
    """
    # TODO: implement greedy decoding
    # Step 1: pick the token with the highest log-prob from step1_lp
    # Step 2: given step 1's choice, pick the best token from step2_lp
    # Step 3: given steps 1 & 2, pick the best token from step3_lp
    # Track cumulative log-probability throughout.
    raise NotImplementedError("Implement greedy_search")


@app.function
def beam_search(step1_lp, step2_lp, step3_lp, B):
    """Perform beam search with beam width B.

    At each step, expand all beams, score candidates, and keep the top B.

    Args:
        step1_lp: dict mapping token -> log-prob for step 1
        step2_lp: dict mapping prev_token -> {token: log-prob} for step 2
        step3_lp: dict mapping (prev2, prev1) -> {token: log-prob} for step 3
        B: beam width

    Returns:
        list of (cumulative_log_prob, sequence_list) for the top B beams,
        sorted best-first.
    """
    vocab = ["a", "b"]
    # TODO: implement beam search
    # Start with a single beam: (0.0, [])
    # At each step:
    #   1. Expand each beam by appending every vocab token.
    #   2. Score each candidate (add the new token's log-prob).
    #   3. Sort candidates by cumulative log-prob (descending).
    #   4. Keep only the top B candidates.
    # After 3 steps, return the final beams.
    raise NotImplementedError("Implement beam_search")


@app.cell(hide_code=True)
def _(mo, step1_logprobs, step2_logprobs, step3_logprobs):
    # Check readiness
    try:
        greedy_search(step1_logprobs, step2_logprobs, step3_logprobs)
        _greedy_ok = True
    except NotImplementedError:
        _greedy_ok = False
    try:
        beam_search(step1_logprobs, step2_logprobs, step3_logprobs, 3)
        _beam_ok = True
    except NotImplementedError:
        _beam_ok = False

    if _greedy_ok:
        _seq, _lp = greedy_search(step1_logprobs, step2_logprobs, step3_logprobs)
        _greedy_md = mo.md(
            f"### Q1: Greedy Decoding\n\n"
            f"**Sequence:** {' '.join(_seq)}\n\n"
            f"**Cumulative log-probability:** {_lp:.2f}"
        )
    else:
        _greedy_md = mo.callout(
            mo.md("Implement `greedy_search` above, then re-run."), kind="warn"
        )

    if _beam_ok:
        ex3_B_slider = mo.ui.slider(1, 4, value=3, step=1, label="Beam width B", show_value=True)
        mo.vstack([_greedy_md, mo.md("---"), mo.md("### Q2: Beam Search"), ex3_B_slider])
    else:
        ex3_B_slider = None
        mo.vstack([
            _greedy_md,
            mo.md("---"),
            mo.callout(mo.md("Implement `beam_search` above, then re-run."), kind="warn"),
        ])
    return (ex3_B_slider,)


@app.cell(hide_code=True)
def _(
    alt,
    ex3_B_slider,
    mo,
    pd,
    step1_logprobs,
    step2_logprobs,
    step3_logprobs,
):
    if ex3_B_slider is None:
        mo.output.replace(mo.md(""))
    else:
        _B = ex3_B_slider.value
        _vocab = ["a", "b"]

        # Run beam search step-by-step, recording all candidates and survivors
        _beams = [(0.0, [])]
        _all_steps = []  # list of dicts per step: candidates + survivors

        for _step in range(3):
            _cands = []
            for _lp, _seq in _beams:
                for _t in _vocab:
                    if _step == 0:
                        _nlp = _lp + step1_logprobs[_t]
                    elif _step == 1:
                        _nlp = _lp + step2_logprobs[_seq[-1]][_t]
                    else:
                        _nlp = _lp + step3_logprobs[(_seq[-2], _seq[-1])][_t]
                    _cands.append((_nlp, _seq + [_t]))
            _cands.sort(key=lambda x: x[0], reverse=True)
            _survivors = _cands[:_B]
            _all_steps.append({"candidates": _cands, "survivors": _survivors})
            _beams = _survivors

        # Also run greedy for comparison
        try:
            _greedy_seq, _greedy_lp = greedy_search(
                step1_logprobs, step2_logprobs, step3_logprobs
            )
        except NotImplementedError:
            _greedy_seq, _greedy_lp = None, None

        # Build heatmap data: show all candidates at each step with survive status
        _rows = []
        for _si, _step_data in enumerate(_all_steps):
            _surv_seqs = {tuple(s) for _, s in _step_data["survivors"]}
            for _ci, (_lp, _seq) in enumerate(_step_data["candidates"]):
                _is_surv = tuple(_seq) in _surv_seqs
                _label = " ".join(_seq)
                _rows.append(
                    {
                        "Step": _si + 1,
                        "Sequence": _label,
                        "Log-prob": _lp,
                        "Status": "Kept" if _is_surv else "Pruned",
                        "rank": _ci,
                    }
                )
        _df = pd.DataFrame(_rows)

        # Heatmap of all candidates at each step
        _hm = (
            alt.Chart(_df)
            .mark_rect(cornerRadius=4, stroke="white", strokeWidth=2)
            .encode(
                x=alt.X("Step:O", title="Step", axis=alt.Axis(labelFontSize=14)),
                y=alt.Y(
                    "rank:O",
                    title="Candidate rank",
                    axis=alt.Axis(labelFontSize=12),
                    sort="ascending",
                ),
                color=alt.Color(
                    "Log-prob:Q",
                    scale=alt.Scale(scheme="viridis"),
                    legend=alt.Legend(format=".2f", title="Cumul. log-prob"),
                ),
                stroke=alt.condition(
                    alt.datum.Status == "Kept",
                    alt.value("#ff6b6b"),
                    alt.value("white"),
                ),
                strokeWidth=alt.condition(
                    alt.datum.Status == "Kept",
                    alt.value(3),
                    alt.value(1),
                ),
                opacity=alt.condition(
                    alt.datum.Status == "Kept",
                    alt.value(1.0),
                    alt.value(0.35),
                ),
                tooltip=[
                    "Step:O",
                    "Sequence:N",
                    alt.Tooltip("Log-prob:Q", format=".2f"),
                    "Status:N",
                ],
            )
            .properties(width=250, height=max(_B * 2 * 55, 160))
        )

        _txt = (
            alt.Chart(_df)
            .mark_text(fontSize=12, fontWeight="bold")
            .encode(
                x="Step:O",
                y=alt.Y("rank:O", sort="ascending"),
                text="Sequence:N",
                opacity=alt.condition(
                    alt.datum.Status == "Kept",
                    alt.value(1.0),
                    alt.value(0.4),
                ),
            )
        )

        _chart = (_hm + _txt).properties(title=f"Beam Search (B={_B})")

        # Final results table
        _final_beams = _all_steps[-1]["survivors"]
        _res_rows = []
        for _i, (_lp, _seq) in enumerate(_final_beams):
            _is_greedy = (
                _greedy_seq is not None and " ".join(_seq) == " ".join(_greedy_seq)
            )
            _res_rows.append(
                {
                    "Rank": _i + 1,
                    "Sequence": " ".join(_seq),
                    "Log-prob": f"{_lp:.2f}",
                    "Note": "= greedy" if _is_greedy else "",
                }
            )
        _res_df = pd.DataFrame(_res_rows)

        # Step-by-step detail in accordion
        _detail_lines = []
        for _si, _step_data in enumerate(_all_steps):
            _detail_lines.append(f"**Step {_si + 1} candidates:**\n")
            _surv_seqs = {tuple(s) for _, s in _step_data["survivors"]}
            for _lp, _seq in _step_data["candidates"]:
                _mark = " **[kept]**" if tuple(_seq) in _surv_seqs else ""
                _detail_lines.append(
                    f"- `{' '.join(_seq)}` : cumul. log-prob = {_lp:.2f}{_mark}"
                )
            _detail_lines.append("")

        mo.vstack(
            [
                _chart,
                mo.md(
                    "Red borders mark the beams that survive each step. "
                    "Pruned candidates are faded."
                ),
                mo.md("**Final beams:**"),
                mo.ui.table(_res_df, selection=None),
                mo.md(
                    f"**Top sequence:** {' '.join(_final_beams[0][1])}  |  "
                    f"**Log-probability:** {_final_beams[0][0]:.2f}"
                ),
                mo.accordion(
                    {"Step-by-step candidate details": mo.md("\n".join(_detail_lines))}
                ),
            ]
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    Once you have all answers, fill in the paper exercise sheet and submit it as your exit ticket.
    """)
    return


if __name__ == "__main__":
    app.run()
