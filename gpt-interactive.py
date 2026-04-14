# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch==2.6.0",
#     "transformers==4.49.0",
#     "numpy",
#     "plotly",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Generative Pre-trained Transformer (GPT)

    GPT is an **autoregressive language model** that generates text one token at a time. Given tokens $(x_1, \ldots, x_n)$, the model maximizes:

    $$P(x_1, \ldots, x_n) = \prod_{i=1}^n P(x_i \mid x_1, \ldots, x_{i-1})$$

    The model gives us a conditional probability distribution over the next token, but finding the *best* full sequence is intractable. Instead, we use **sampling strategies** to generate text.

    How do we go from raw model outputs to a chosen token? Let's build the intuition step by step.
    """)
    return


# ──────────────────────────────────────────────
# Section 1: Load model & get raw logits
# ──────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Step 0: Load a Language Model

    We use **SmolLM2-135M**, a compact language model that runs comfortably on CPU. Let's load it and peek at its raw output.
    """)
    return


@app.cell
def _():
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM

    _device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(_device)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M").to(device)
    model.eval()
    return AutoModelForCausalLM, AutoTokenizer, device, model, np, tokenizer, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    When we feed a prompt into the model, it doesn't directly output words. It outputs a vector of **logits**, one number per token in the vocabulary. These are raw, unnormalized scores. Positive logits mean the model thinks that token is a plausible continuation. Negative logits mean it's unlikely.

    Let's see what the model produces for a simple prompt.
    """)
    return


@app.cell
def _(device, model, mo, np, tokenizer, torch):
    _prompt = "I wish I could be"
    _inputs = tokenizer(_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        _out = model(**_inputs)
    # Logits for the last token position = prediction for the next token
    raw_logits = _out.logits[0, -1, :].cpu().numpy()

    # Show top 15 tokens by logit value
    _top_idx = np.argsort(raw_logits)[::-1][:15]
    _rows = []
    for _i in _top_idx:
        _tok = tokenizer.decode([_i])
        _rows.append(f"| `{_tok}` | {raw_logits[_i]:.2f} |")

    mo.md(
        f'**Prompt:** "{_prompt}"\n\n'
        "The model produces one logit per vocabulary token "
        f"({len(raw_logits):,} tokens total). Here are the top 15:\n\n"
        "| Token | Logit |\n|---|---|\n" + "\n".join(_rows)
    )
    return (raw_logits,)


# ──────────────────────────────────────────────
# Section 2: Softmax
# ──────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: Softmax — From Logits to Probabilities

    Raw logits are hard to interpret. We need to convert them into a proper probability distribution that sums to 1. That's what the **softmax** function does:

    $$p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$$

    Softmax does two things. First, it exponentiates, which makes all values positive and amplifies differences. Second, it normalizes, so the values sum to 1.

    Let's see it in action on our model's logits.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, raw_logits, tokenizer):
    # Compute softmax
    _exp = np.exp(raw_logits - np.max(raw_logits))  # subtract max for numerical stability
    _probs = _exp / _exp.sum()

    _top_idx = np.argsort(_probs)[::-1][:15]
    _rows = []
    for _i in _top_idx:
        _tok = tokenizer.decode([_i])
        _rows.append(f"| `{_tok}` | {raw_logits[_i]:.2f} | {_probs[_i]:.4f} |")

    mo.md(
        "After softmax, the logits become probabilities:\n\n"
        "| Token | Logit | Probability |\n|---|---|---|\n" + "\n".join(_rows) + "\n\n"
        f"The top token has probability **{_probs[_top_idx[0]]:.2%}**. "
        "Notice how softmax concentrates most of the probability mass on a handful of tokens. "
        "If we always pick the top one, that's **greedy sampling**. Simple, but repetitive."
    )
    return


# ──────────────────────────────────────────────
# Section 3: Temperature
# ──────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: Temperature — Controlling Randomness

    What if the distribution is too peaked (always picking the same token) or too flat (picking nonsense)? We can tune the **sharpness** of the distribution by dividing the logits by a temperature parameter $\tau$ before applying softmax:

    $$p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$

    When $\tau < 1$, the distribution becomes *sharper* (more confident). When $\tau > 1$, it becomes *flatter* (more random). At $\tau = 1$, it's the standard softmax.

    Use the slider below to see how temperature reshapes the probability distribution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    temp_slider = mo.ui.slider(
        start=0.1, stop=3.0, step=0.1, value=1.0,
        label="Temperature (τ)",
        show_value=True,
    )
    temp_slider
    return (temp_slider,)


@app.cell(hide_code=True)
def _(mo, np, raw_logits, temp_slider, tokenizer):
    _tau = temp_slider.value
    _scaled = raw_logits / _tau
    _exp = np.exp(_scaled - np.max(_scaled))
    _probs = _exp / _exp.sum()

    _top_idx = np.argsort(_probs)[::-1][:20]
    _tokens = [tokenizer.decode([i]) for i in _top_idx]
    _values = [_probs[i] for i in _top_idx]

    # Build a horizontal bar chart using plain text/HTML
    _bars = []
    _max_val = max(_values)
    for _tok, _val in zip(_tokens, _values):
        _pct = _val / _max_val * 100
        _bar_html = (
            f'<div style="display:flex;align-items:center;margin:2px 0;">'
            f'<code style="width:80px;text-align:right;margin-right:8px;white-space:pre">{_tok}</code>'
            f'<div style="background:#4a90d9;height:18px;width:{_pct:.1f}%;border-radius:3px;"></div>'
            f'<span style="margin-left:6px;font-size:0.85em">{_val:.4f}</span>'
            f'</div>'
        )
        _bars.append(_bar_html)

    _entropy = -np.sum(_probs[_probs > 0] * np.log2(_probs[_probs > 0]))

    mo.md(
        f"### Token probabilities at τ = {_tau:.1f}\n\n"
        + "".join(_bars) + "\n\n"
        f"**Entropy:** {_entropy:.2f} bits. "
        + ("The distribution is very peaked. The model is highly confident in one token. "
           if _tau < 0.5 else
           "The distribution is sharp. A few tokens dominate. "
           if _tau < 1.0 else
           "Standard softmax. The model's natural confidence level. "
           if _tau == 1.0 else
           "The distribution is getting flatter. More tokens become viable candidates. "
           if _tau < 2.0 else
           "The distribution is very flat. Almost any token could be sampled, leading to incoherent text. ")
    )
    return


# ──────────────────────────────────────────────
# Section 4: Top-k Sampling
# ──────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: Top-k Sampling — Limiting Candidates

    Even after softmax, the model assigns *some* probability to every token in the vocabulary. Most of these are near zero, but sampling from all of them can occasionally produce garbage tokens.

    **Top-k sampling** solves this by keeping only the $k$ most likely tokens and zeroing out everything else. We then renormalize the remaining probabilities and sample from that smaller set.

    Use the slider to see how $k$ affects which tokens remain in the candidate pool.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    topk_k_slider = mo.ui.slider(
        start=1, stop=50, step=1, value=10,
        label="k (number of candidates)",
        show_value=True,
    )
    topk_temp_slider = mo.ui.slider(
        start=0.1, stop=3.0, step=0.1, value=1.0,
        label="Temperature (τ)",
        show_value=True,
    )
    mo.vstack([topk_k_slider, topk_temp_slider])
    return topk_k_slider, topk_temp_slider


@app.cell(hide_code=True)
def _(mo, np, raw_logits, tokenizer, topk_k_slider, topk_temp_slider):
    _k = topk_k_slider.value
    _tau = topk_temp_slider.value

    # Apply temperature
    _scaled = raw_logits / _tau
    _exp = np.exp(_scaled - np.max(_scaled))
    _full_probs = _exp / _exp.sum()

    # Top-k filtering
    _sorted_idx = np.argsort(_full_probs)[::-1]
    _top_k_idx = _sorted_idx[:_k]
    _rejected_idx = _sorted_idx[_k:_k + 10]  # show a few rejected tokens

    # Renormalize
    _top_k_probs_raw = _full_probs[_top_k_idx]
    _top_k_probs = _top_k_probs_raw / _top_k_probs_raw.sum()

    _max_val = max(_top_k_probs)
    _bars = []
    for _i, _idx in enumerate(_top_k_idx):
        _tok = tokenizer.decode([_idx])
        _val = _top_k_probs[_i]
        _pct = _val / _max_val * 100
        _bar_html = (
            f'<div style="display:flex;align-items:center;margin:2px 0;">'
            f'<code style="width:80px;text-align:right;margin-right:8px;white-space:pre">{_tok}</code>'
            f'<div style="background:#4a90d9;height:18px;width:{_pct:.1f}%;border-radius:3px;"></div>'
            f'<span style="margin-left:6px;font-size:0.85em">{_val:.4f}</span>'
            f'</div>'
        )
        _bars.append(_bar_html)

    # Show rejected tokens
    _rejected = ", ".join([f"`{tokenizer.decode([_idx])}`" for _idx in _rejected_idx[:5]])
    _mass_kept = _full_probs[_top_k_idx].sum()

    mo.md(
        f"### Top-{_k} candidates (τ = {_tau:.1f})\n\n"
        + "".join(_bars) + "\n\n"
        f"These {_k} tokens capture **{_mass_kept:.1%}** of the original probability mass. "
        f"Tokens just outside the cutoff: {_rejected}.\n\n"
        + ("With k=1 this is equivalent to **greedy sampling** — always picking the top token."
           if _k == 1 else
           f"We sample from these {_k} tokens proportionally to their (renormalized) probabilities."
           if _k <= 10 else
           f"With k={_k}, we allow quite a bit of diversity while still filtering out low-probability noise.")
    )
    return


# ──────────────────────────────────────────────
# Section 5: Top-p (Nucleus) Sampling
# ──────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4: Nucleus (Top-p) Sampling — Adaptive Candidate Pool

    Top-k uses a fixed number of candidates regardless of how the probability is distributed. But sometimes the model is very confident (one token has 90% probability) and sometimes it's uncertain (probability spread across many tokens).

    **Nucleus sampling** adapts to this. Instead of a fixed $k$, we select the smallest set of tokens whose cumulative probability exceeds a threshold $p$. When the model is confident, only a few tokens make the cut. When uncertain, more do.

    Watch how the number of selected tokens changes as you adjust $p$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    topp_p_slider = mo.ui.slider(
        start=0.1, stop=1.0, step=0.05, value=0.9,
        label="p (cumulative probability threshold)",
        show_value=True,
    )
    topp_temp_slider = mo.ui.slider(
        start=0.1, stop=3.0, step=0.1, value=1.0,
        label="Temperature (τ)",
        show_value=True,
    )
    mo.vstack([topp_p_slider, topp_temp_slider])
    return topp_p_slider, topp_temp_slider


@app.cell(hide_code=True)
def _(mo, np, raw_logits, tokenizer, topp_p_slider, topp_temp_slider):
    _p = topp_p_slider.value
    _tau = topp_temp_slider.value

    # Apply temperature
    _scaled = raw_logits / _tau
    _exp = np.exp(_scaled - np.max(_scaled))
    _full_probs = _exp / _exp.sum()

    # Sort by probability descending
    _sorted_idx = np.argsort(_full_probs)[::-1]
    _sorted_probs = _full_probs[_sorted_idx]
    _cumsum = np.cumsum(_sorted_probs)

    # Find the nucleus: smallest set with cumulative prob >= p
    _nucleus_size = int(np.searchsorted(_cumsum, _p) + 1)
    _nucleus_idx = _sorted_idx[:_nucleus_size]

    # Renormalize
    _nucleus_probs_raw = _full_probs[_nucleus_idx]
    _nucleus_probs = _nucleus_probs_raw / _nucleus_probs_raw.sum()

    # Show up to 30 tokens, indicate cutoff
    _show_n = min(_nucleus_size + 5, 30)
    _show_idx = _sorted_idx[:_show_n]
    _max_val = max(_nucleus_probs) if len(_nucleus_probs) > 0 else 1.0

    _bars = []
    _cum = 0.0
    for _i, _idx in enumerate(_show_idx):
        _tok = tokenizer.decode([_idx])
        _prob = _full_probs[_idx]
        _cum += _prob
        _in_nucleus = _i < _nucleus_size
        _color = "#4a90d9" if _in_nucleus else "#cccccc"
        _val = _nucleus_probs[_i] if _in_nucleus else _prob
        _pct = (_val / _max_val * 100) if _in_nucleus else (_prob / _max_val * 100)
        _label = f"{_val:.4f}" if _in_nucleus else f"<span style='color:#999'>{_prob:.4f} (excluded)</span>"
        _bar_html = (
            f'<div style="display:flex;align-items:center;margin:2px 0;">'
            f'<code style="width:80px;text-align:right;margin-right:8px;white-space:pre">{_tok}</code>'
            f'<div style="background:{_color};height:18px;width:{_pct:.1f}%;border-radius:3px;"></div>'
            f'<span style="margin-left:6px;font-size:0.85em">{_label}</span>'
            f'</div>'
        )
        _bars.append(_bar_html)

    mo.md(
        f"### Nucleus at p = {_p:.2f} (τ = {_tau:.1f})\n\n"
        + "".join(_bars) + "\n\n"
        f"**{_nucleus_size} tokens** are in the nucleus "
        f"(covering {_full_probs[_nucleus_idx].sum():.1%} of original mass). "
        "Blue bars are included, gray bars are excluded.\n\n"
        + ("The nucleus is very tight. The model is confident here, so only a few tokens pass the threshold."
           if _nucleus_size <= 3 else
           "The nucleus includes a moderate number of tokens, balancing diversity and quality."
           if _nucleus_size <= 15 else
           "The nucleus is large. The model is uncertain, so many tokens share the probability mass.")
    )
    return


# ──────────────────────────────────────────────
# Section 6: Beam Search
# ──────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5: Beam Search — Exploring Multiple Paths

    All the methods above make decisions one token at a time. Greedy picks the single best token, and top-k/top-p sample from a filtered set. But what if the best *sequence* doesn't start with the best *first token*?

    Consider a toy example. Token A has 60% probability at step 1, and token B has 40%. Greedy always picks A. But what if every continuation of A is mediocre, while B leads to a highly probable sequence? Greedy would never discover that.

    **Beam search** addresses this by keeping track of $k$ candidate sequences (called *beams*) in parallel. At each step, it expands every beam by considering the top-$B$ next tokens, then keeps only the $k$ highest-scoring full sequences. This lets it explore multiple paths without the exponential cost of checking every possibility.

    The algorithm works as follows. Start with $k$ copies of the prompt. For each beam, compute the next-token probabilities and expand to the top-$B$ candidates. Now there are $k \times B$ candidate sequences. Score each by its total log-probability (product of all token probabilities). Keep the top-$k$ and discard the rest. Repeat until done.

    Beam search is **deterministic** (no randomness) but can find higher-probability sequences than greedy. The tradeoff is computational cost: more beams means better exploration but slower generation.

    Let's visualize how beams branch and get pruned at each step.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    beam_k_slider = mo.ui.slider(
        start=2, stop=6, step=1, value=3,
        label="Number of beams (k)",
        show_value=True,
    )
    beam_steps_slider = mo.ui.slider(
        start=1, stop=6, step=1, value=3,
        label="Steps to visualize",
        show_value=True,
    )
    mo.vstack([beam_k_slider, beam_steps_slider])
    return beam_k_slider, beam_steps_slider


@app.cell(hide_code=True)
def _(beam_k_slider, beam_steps_slider, device, model, np, tokenizer, torch):
    import plotly.graph_objects as go

    _k = beam_k_slider.value
    _n_steps = beam_steps_slider.value
    _prompt = "I wish I could be"

    _input_ids = tokenizer(_prompt, return_tensors="pt").input_ids.to(device)

    # ── Run beam search, recording the full tree ──
    # Each node: (node_id, parent_id, step, token_str, cumul_score, survived)
    _nodes = []  # list of dicts
    _node_id = 0

    # Root node
    _nodes.append(dict(
        id=_node_id, parent=None, step=0,
        token=_prompt, score=0.0, survived=True, seq=_input_ids[0].tolist(),
    ))
    _node_id += 1

    # Active beams: list of (node_id, seq, cumul_score)
    _active = [(0, _input_ids[0].tolist(), 0.0)]

    for _step in range(1, _n_steps + 1):
        _candidates = []  # (parent_node_id, seq, score, token_str, token_log_prob)

        for _parent_nid, _seq, _parent_score in _active:
            _ids = torch.tensor([_seq], device=device)
            with torch.no_grad():
                _out = model(_ids)
            _logits = _out.logits[0, -1, :]
            _log_probs = torch.nn.functional.log_softmax(_logits, dim=-1)
            _top_vals, _top_ids = torch.topk(_log_probs, _k)

            for _v, _tid in zip(_top_vals.cpu().numpy(), _top_ids.cpu().numpy()):
                _tok_str = tokenizer.decode([int(_tid)])
                _new_score = _parent_score + float(_v)
                _new_seq = _seq + [int(_tid)]
                _candidates.append((_parent_nid, _new_seq, _new_score, _tok_str, float(_v)))

        # Sort and keep top-k
        _candidates.sort(key=lambda x: x[2], reverse=True)
        _survivors = set(range(_k))

        _new_active = []
        for _c_idx, (_par, _seq, _sc, _tok, _lp) in enumerate(_candidates):
            _surv = _c_idx in _survivors
            _nodes.append(dict(
                id=_node_id, parent=_par, step=_step,
                token=_tok, score=_sc, survived=_surv, seq=_seq,
                log_prob=_lp,
            ))
            if _surv:
                _new_active.append((_node_id, _seq, _sc))
            _node_id += 1

        _active = _new_active

    # ── Layout: x = step, y = spread nodes vertically per step ──
    _step_nodes = {}
    for _n in _nodes:
        _s = _n["step"]
        _step_nodes.setdefault(_s, []).append(_n)

    _pos = {}  # node_id -> (x, y)
    for _s, _ns in _step_nodes.items():
        _count = len(_ns)
        for _i, _n in enumerate(_ns):
            _y = (_i - (_count - 1) / 2)  # center around 0
            _pos[_n["id"]] = (_s, _y)

    # ── Build plotly traces ──
    # Edges
    _edge_x, _edge_y = [], []
    _pruned_edge_x, _pruned_edge_y = [], []
    for _n in _nodes:
        if _n["parent"] is not None:
            _px, _py = _pos[_n["parent"]]
            _cx, _cy = _pos[_n["id"]]
            _target = _edge_x if _n["survived"] else _pruned_edge_x
            _target_y = _edge_y if _n["survived"] else _pruned_edge_y
            _target.extend([_px, _cx, None])
            _target_y.extend([_py, _cy, None])

    # Edge traces
    _traces = []
    if _pruned_edge_x:
        _traces.append(go.Scatter(
            x=_pruned_edge_x, y=_pruned_edge_y, mode="lines",
            line=dict(color="#dddddd", width=1.5, dash="dot"),
            hoverinfo="none", showlegend=False,
        ))
    _traces.append(go.Scatter(
        x=_edge_x, y=_edge_y, mode="lines",
        line=dict(color="#4a90d9", width=2.5),
        hoverinfo="none", showlegend=False,
    ))

    # Edge labels (token text at midpoint)
    _label_x, _label_y, _label_text = [], [], []
    for _n in _nodes:
        if _n["parent"] is not None:
            _px, _py = _pos[_n["parent"]]
            _cx, _cy = _pos[_n["id"]]
            _mx = (_px + _cx) / 2
            _my = (_py + _cy) / 2
            _lp = _n.get("log_prob", 0)
            _label_x.append(_mx)
            _label_y.append(_my)
            _color = "#333" if _n["survived"] else "#bbb"
            _label_text.append(
                f'<span style="color:{_color}">{_n["token"]}<br>({_lp:.2f})</span>'
            )

    _traces.append(go.Scatter(
        x=_label_x, y=_label_y, mode="text",
        text=[_n["token"] + f'\n({_n.get("log_prob", 0):.2f})'
              if _n["parent"] is not None else ""
              for _nid in range(len(_label_x))
              for _n in [_nodes[0]]]  # placeholder, replaced below
        if False else _label_text,
        textfont=dict(size=10),
        textposition="middle right",
        hoverinfo="none", showlegend=False,
    ))

    # Survived nodes
    _surv_nodes = [_n for _n in _nodes if _n["survived"]]
    _prune_nodes = [_n for _n in _nodes if not _n["survived"]]

    if _prune_nodes:
        _traces.append(go.Scatter(
            x=[_pos[_n["id"]][0] for _n in _prune_nodes],
            y=[_pos[_n["id"]][1] for _n in _prune_nodes],
            mode="markers",
            marker=dict(size=14, color="#dddddd", line=dict(width=1, color="#ccc")),
            text=[f"<b>{_n['token']}</b><br>Score: {_n['score']:.2f}<br><i>Pruned</i>"
                  for _n in _prune_nodes],
            hoverinfo="text", showlegend=False,
        ))

    # Highlight the best path
    _best_node = max(_active, key=lambda x: x[2])
    _best_nid = _best_node[0]
    _best_path_ids = set()
    _trace_id = _best_nid
    while _trace_id is not None:
        _best_path_ids.add(_trace_id)
        _parent = next((_n["parent"] for _n in _nodes if _n["id"] == _trace_id), None)
        _trace_id = _parent

    _surv_colors = ["#ff6b35" if _n["id"] in _best_path_ids else "#4a90d9" for _n in _surv_nodes]
    _surv_sizes = [20 if _n["id"] in _best_path_ids else 16 for _n in _surv_nodes]

    _traces.append(go.Scatter(
        x=[_pos[_n["id"]][0] for _n in _surv_nodes],
        y=[_pos[_n["id"]][1] for _n in _surv_nodes],
        mode="markers+text",
        marker=dict(size=_surv_sizes, color=_surv_colors,
                    line=dict(width=2, color="white")),
        text=[_n["token"] if _n["step"] == 0 else "" for _n in _surv_nodes],
        textposition="middle left",
        textfont=dict(size=11, color="#333"),
        customdata=[[_n["token"], f"{_n['score']:.2f}"] for _n in _surv_nodes],
        hovertemplate="<b>%{customdata[0]}</b><br>Score: %{customdata[1]}<extra></extra>",
        showlegend=False,
    ))

    # Best path edges highlighted
    _bp_edge_x, _bp_edge_y = [], []
    for _nid in _best_path_ids:
        _n = next((_n for _n in _nodes if _n["id"] == _nid), None)
        if _n and _n["parent"] is not None and _n["parent"] in _best_path_ids:
            _px, _py = _pos[_n["parent"]]
            _cx, _cy = _pos[_n["id"]]
            _bp_edge_x.extend([_px, _cx, None])
            _bp_edge_y.extend([_py, _cy, None])

    if _bp_edge_x:
        _traces.append(go.Scatter(
            x=_bp_edge_x, y=_bp_edge_y, mode="lines",
            line=dict(color="#ff6b35", width=4),
            hoverinfo="none", showlegend=False,
        ))

    # Build the best sequence text
    _best_text = tokenizer.decode(_best_node[1][len(_input_ids[0]):])

    _fig = go.Figure(data=_traces)
    _fig.update_layout(
        title=dict(
            text=f'Beam Search Tree (k={_k})<br>'
                 f'<span style="font-size:13px;color:#666">Best path (orange): '
                 f'"{_prompt}<b>{_best_text}</b>" (score: {_best_node[2]:.2f})</span>',
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Step", showgrid=False, zeroline=False,
            tickmode="array",
            tickvals=list(range(_n_steps + 1)),
            ticktext=["prompt"] + [f"step {i}" for i in range(1, _n_steps + 1)],
        ),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        height=max(350, len(_nodes) * 28),
        margin=dict(l=40, r=40, t=80, b=40),
        hoverlabel=dict(bgcolor="white"),
    )

    _fig
    return


# ──────────────────────────────────────────────
# Section 7: Putting It All Together
# ──────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Putting It All Together

    Now that we understand each piece, let's use them for actual text generation. The controls below let you pick any combination of sampling strategies and see how they affect the generated text.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    prompt_input = mo.ui.text(
        value="I wish I could be",
        label="Prompt",
        full_width=True,
    )
    method_select = mo.ui.dropdown(
        options=["Greedy", "Beam Search", "Top-k", "Top-p (Nucleus)", "Top-k + Top-p + Temperature"],
        value="Greedy",
        label="Sampling method",
    )
    max_tokens_slider = mo.ui.slider(
        start=10, stop=100, step=10, value=30,
        label="Max new tokens",
        show_value=True,
    )
    num_beams_slider = mo.ui.slider(
        start=2, stop=10, step=1, value=5,
        label="Number of beams",
        show_value=True,
    )
    gen_top_k_slider = mo.ui.slider(
        start=1, stop=50, step=1, value=10,
        label="Top-k",
        show_value=True,
    )
    gen_top_p_slider = mo.ui.slider(
        start=0.1, stop=1.0, step=0.05, value=0.95,
        label="Top-p",
        show_value=True,
    )
    gen_temperature_slider = mo.ui.slider(
        start=0.1, stop=3.0, step=0.1, value=1.0,
        label="Temperature",
        show_value=True,
    )
    generate_button = mo.ui.run_button(label="Generate")
    return (
        gen_temperature_slider,
        gen_top_k_slider,
        gen_top_p_slider,
        generate_button,
        max_tokens_slider,
        method_select,
        num_beams_slider,
        prompt_input,
    )


@app.cell(hide_code=True)
def _(
    gen_temperature_slider,
    gen_top_k_slider,
    gen_top_p_slider,
    generate_button,
    max_tokens_slider,
    method_select,
    mo,
    num_beams_slider,
    prompt_input,
):
    _method = method_select.value
    _controls = [prompt_input, method_select, max_tokens_slider]

    if _method == "Beam Search":
        _controls.append(num_beams_slider)
    elif _method == "Top-k":
        _controls.extend([gen_top_k_slider, gen_temperature_slider])
    elif _method == "Top-p (Nucleus)":
        _controls.extend([gen_top_p_slider, gen_temperature_slider])
    elif _method == "Top-k + Top-p + Temperature":
        _controls.extend([gen_top_k_slider, gen_top_p_slider, gen_temperature_slider])

    _controls.append(generate_button)
    mo.vstack(_controls)
    return


@app.cell
def _(
    gen_temperature_slider,
    gen_top_k_slider,
    gen_top_p_slider,
    generate_button,
    max_tokens_slider,
    method_select,
    mo,
    model,
    num_beams_slider,
    prompt_input,
    tokenizer,
):
    mo.stop(not generate_button.value, mo.md("*Click **Generate** to see output.*"))

    _method = method_select.value
    _prompt = prompt_input.value
    _max_tokens = max_tokens_slider.value
    _pad_id = tokenizer.eos_token_id
    _inputs = tokenizer(_prompt, return_tensors="pt").to(model.device)

    _gen_kwargs = dict(
        **_inputs,
        max_new_tokens=_max_tokens,
        pad_token_id=_pad_id,
    )

    if _method == "Greedy":
        _gen_kwargs["do_sample"] = False
    elif _method == "Beam Search":
        _n = num_beams_slider.value
        _gen_kwargs.update(do_sample=False, num_beams=_n, num_return_sequences=_n)
    elif _method == "Top-k":
        _gen_kwargs.update(do_sample=True, top_k=gen_top_k_slider.value, temperature=gen_temperature_slider.value)
    elif _method == "Top-p (Nucleus)":
        _gen_kwargs.update(do_sample=True, top_p=gen_top_p_slider.value, temperature=gen_temperature_slider.value)
    else:
        _gen_kwargs.update(
            do_sample=True, top_k=gen_top_k_slider.value,
            top_p=gen_top_p_slider.value, temperature=gen_temperature_slider.value,
        )

    _output_ids = model.generate(**_gen_kwargs)

    _result_lines = []
    for i, _ids in enumerate(_output_ids):
        _text = tokenizer.decode(_ids, skip_special_tokens=True)
        if len(_output_ids) > 1:
            _result_lines.append(f"**Beam {i+1}:** {_text}")
        else:
            _result_lines.append(_text)

    mo.callout(mo.md("\n\n".join(_result_lines)), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    We built up the text generation pipeline piece by piece:

    **Logits** are the raw model output, one score per vocabulary token. **Softmax** converts them into probabilities. **Temperature** controls how peaked or flat that distribution is. **Top-k** limits candidates to a fixed number of the most likely tokens. **Top-p (nucleus)** adapts the candidate pool size based on the model's confidence. In practice, these are combined (e.g., top-k=50, top-p=0.95, temperature=0.7) to balance coherence and diversity.

    /// tip | Try it yourself
    Go back to the interactive generation section and experiment. Try greedy first to see the repetition problem, then switch to top-k or nucleus sampling. What happens as you crank up the temperature?
    ///
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
