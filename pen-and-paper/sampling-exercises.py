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
    mo.md(
        """
        # GPT Sampling Exercises

        Use this notebook to implement and test your solutions for the pen-and-paper exercises.
        Each exercise has a starter cell with `TODO` comments where you write your code,
        followed by test cells that compute the answers you need to fill in on the paper.
        """
    )
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
    mo.md(
        """
        ---

        ## Exercise 1: Softmax with Temperature

        Softmax converts logits $z = [z_1, \\dots, z_V]$ into probabilities:

        $$P(i) = \\frac{e^{z_i / T}}{\\sum_j e^{z_j / T}}$$

        **Your task:** implement `softmax_with_temperature(z, T)` below, then run the
        test cell to compute and plot the distributions for $T \\in \\{0.5, 1, 2, 5\\}$.
        """
    )
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
def _(alt, mo, np, pd, softmax_with_temperature):
    # --- Test cell for Exercise 1 ---
    _z = np.array([1.0, 0.0, -1.0])
    _tokens = ["a", "b", "c"]
    _temperatures = [0.5, 1.0, 2.0, 5.0]

    try:
        _rows = []
        for _T in _temperatures:
            _probs = softmax_with_temperature(_z, _T)
            for _tok, _p in zip(_tokens, _probs):
                _rows.append({"Token": _tok, "T": _T, "P(token)": _p})

        _df = pd.DataFrame(_rows)

        _chart = (
            alt.Chart(_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("T:Q", title="Temperature T"),
                y=alt.Y("P(token):Q", scale=alt.Scale(domain=[0, 1]), title="P(token)"),
                color=alt.Color("Token:N"),
                tooltip=[
                    "Token",
                    alt.Tooltip("T:Q", format=".1f"),
                    alt.Tooltip("P(token):Q", format=".4f"),
                ],
            )
            .properties(width=500, height=300, title="P(token) vs Temperature")
        )

        # Build answer table
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
                mo.md("### Results for Exercise 1"),
                _chart,
                mo.md("**Probability table** (copy these to your paper):"),
                mo.ui.table(_ans_df, selection=None),
            ]
        )
    except NotImplementedError:
        mo.callout(
            mo.md("Implement `softmax_with_temperature` in the cell above, then re-run."),
            kind="warn",
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ---

        ## Exercise 2: Top-$k$ and Top-$p$ Sampling

        **Top-$k$** keeps the $k$ highest-probability tokens and zeroes out the rest, then renormalizes.

        **Top-$p$** (nucleus) keeps the fewest tokens whose cumulative probability reaches $p$, then renormalizes.

        Logits: $z = [3.0, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0]$

        **Your task:** implement `top_k_sampling` and `top_p_sampling` below.
        """
    )
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
def _(mo, np, pd, softmax_with_temperature, top_k_sampling, top_p_sampling):
    _z = np.array([3.0, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
    _labels = [f"t{i}" for i in range(10)]

    try:
        # Q1: top-k with k=3 at T=1
        _probs_t1 = softmax_with_temperature(_z, 1.0)
        _topk3 = top_k_sampling(_probs_t1, 3)

        _q1_df = pd.DataFrame(
            {"Token": _labels, "Original P": [f"{p:.4f}" for p in _probs_t1], "Top-k=3 P": [f"{p:.4f}" for p in _topk3]}
        )

        # Q2: top-p with p=0.8 at T=1
        _topp80 = top_p_sampling(_probs_t1, 0.8)
        _n_included_q2 = int(np.sum(_topp80 > 0))

        _q2_df = pd.DataFrame(
            {"Token": _labels, "Original P": [f"{p:.4f}" for p in _probs_t1], "Top-p=0.8 P": [f"{p:.4f}" for p in _topp80]}
        )

        # Q3: top-p with p=0.8 at T=2
        _probs_t2 = softmax_with_temperature(_z, 2.0)
        _topp80_t2 = top_p_sampling(_probs_t2, 0.8)
        _n_included_q3 = int(np.sum(_topp80_t2 > 0))

        _q3_df = pd.DataFrame(
            {
                "Token": _labels,
                "P (T=2)": [f"{p:.4f}" for p in _probs_t2],
                "Top-p=0.8 (T=2) P": [f"{p:.4f}" for p in _topp80_t2],
            }
        )

        mo.vstack(
            [
                mo.md("### Results for Exercise 2"),
                mo.md("**Q1: Top-k with k=3** (sampling probabilities):"),
                mo.ui.table(_q1_df, selection=None),
                mo.md(f"**Q2: Top-p with p=0.8 at T=1.** Number of tokens included: **{_n_included_q2}**"),
                mo.ui.table(_q2_df, selection=None),
                mo.md(
                    f"**Q3: Top-p with p=0.8 at T=2.** Number of tokens included: **{_n_included_q3}**"
                    + (
                        f" (changed from {_n_included_q2})"
                        if _n_included_q2 != _n_included_q3
                        else f" (same as Q2)"
                    )
                ),
                mo.ui.table(_q3_df, selection=None),
            ]
        )
    except NotImplementedError:
        mo.callout(
            mo.md(
                "Implement `softmax_with_temperature`, `top_k_sampling`, and `top_p_sampling` above, then re-run."
            ),
            kind="warn",
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ---

        ## Exercise 3: Beam Search

        This exercise is best done by hand, but you can use the code below to verify your answers.

        Vocab: $\\{a, b\\}$, three steps. Step 3 depends on both previous tokens.

        The log-probability tables are pre-loaded below. Run the greedy and beam search
        cells to check your pen-and-paper work.
        """
    )
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


@app.cell
def _(mo, step1_logprobs, step2_logprobs, step3_logprobs):
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

    try:
        _seq, _lp = greedy_search(step1_logprobs, step2_logprobs, step3_logprobs)
        mo.vstack(
            [
                mo.md("### Q1: Greedy Decoding"),
                mo.md(f"**Sequence:** {' '.join(_seq)}"),
                mo.md(f"**Cumulative log-probability:** {_lp:.2f}"),
            ]
        )
    except NotImplementedError:
        mo.callout(mo.md("Implement `greedy_search` above, then re-run."), kind="warn")
    return (greedy_search,)


@app.cell
def _(mo, step1_logprobs, step2_logprobs, step3_logprobs):
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
            Also return a dict of step-by-step candidate details for display.
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

    try:
        _beams = beam_search(step1_logprobs, step2_logprobs, step3_logprobs, B=3)
        _lines = ["### Q2: Beam Search (B=3)", ""]
        _lines.append("**Final beams (sorted best-first):**")
        _lines.append("")
        _lines.append("| Rank | Sequence | Cumulative log-prob |")
        _lines.append("|------|----------|-------------------|")
        for _i, (_lp, _seq) in enumerate(_beams):
            _lines.append(f"| {_i+1} | {' '.join(_seq)} | {_lp:.2f} |")
        _lines.append("")
        _lines.append(f"**Top sequence:** {' '.join(_beams[0][1])}")
        _lines.append(f"**Log-probability:** {_beams[0][0]:.2f}")
        mo.md("\n".join(_lines))
    except NotImplementedError:
        mo.callout(mo.md("Implement `beam_search` above, then re-run."), kind="warn")
    return (beam_search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ---

        Once you have all answers, fill in the paper exercise sheet and submit it as your exit ticket.
        """
    )
    return


if __name__ == "__main__":
    app.run()
