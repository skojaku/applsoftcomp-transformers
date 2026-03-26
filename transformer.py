# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair==6.0.0",
#     "marimo>=0.21.1",
#     "numpy==2.4.3",
#     "pandas==3.0.1",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium", css_file="marimo_lecture_note_theme.css")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Transformers Inside Out

    *An Interactive Guide to Attention, Residual Connections, and Positional Encoding*

    [@SadamoriKojaku](https://skojaku.github.io/)

    ## Attention as Weighted Average

    Consider the word "bank." A static embedding gives it one fixed position, but its meaning shifts depending on context -- is it a financial institution or the side of a river?

    We need a way to let surrounding words influence the meaning of "bank."
    """)
    return


@app.cell(hide_code=True)
def _(embeddings, mo, pd, scatter_plot, words):
    _df = pd.DataFrame({"word": words, "x": embeddings[:, 0], "y": embeddings[:, 1]})
    _chart = scatter_plot(_df, _df, title="Static Word Embeddings", width=400, height=400)

    mo.vstack(
        [
            _chart,
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The simplest idea: compute a weighted average of all word vectors in the sentence.

    $$
    v_{\text{bank}}^{\text{new}} = a_1 \, v_{\text{bank}} + a_2 \, v_{\text{money}} + a_3 \, v_{\text{loan}} + a_4 \, v_{\text{river}} + a_5 \, v_{\text{shore}}
    $$

    If we put large weight $a$ on "money" and "loan," the new "bank" vector shifts toward the financial cluster. Weight "river" and "shore" instead, and it drifts toward geography.

    The weights have constraints so that, when used, they give a "weighted" average.

    1. The weights sum to 1
    2. The weight $a$ is non-negative $(a \geq 0)$.
    """)
    return


@app.cell(hide_code=True)
def _(
    embeddings,
    mo,
    np,
    pd,
    scatter_plot,
    slider_bank,
    slider_loan,
    slider_money,
    slider_river,
    slider_shore,
    words,
):
    _raw = np.array([slider_bank.value, slider_money.value, slider_loan.value, slider_river.value, slider_shore.value])
    _weights = _raw / _raw.sum()
    _new_vec = _weights @ embeddings

    _df_orig = pd.DataFrame({"word": words, "x": embeddings[:, 0], "y": embeddings[:, 1]})
    _df_new = pd.DataFrame({"word": ["bank (new)"], "x": [_new_vec[0]], "y": [_new_vec[1]]})

    _chart = scatter_plot(_df_new, _df_orig, title="Contextualized 'bank'", width=400, height=400)

    _eq = r"$v_{\text{bank}}^{\text{new}} = " + " + ".join(
        f"{_weights[i]:.2f} \\, v_{{\\text{{{words[i]}}}}}$"
        if i == len(words) - 1
        else f"{_weights[i]:.2f} \\, v_{{\\text{{{words[i]}}}}}"
        for i in range(len(words))
    )

    mo.vstack(
        [
            mo.md("Drag the sliders to change the weights and see how 'bank' moves."),
            mo.hstack(
                [mo.vstack([slider_bank, slider_money, slider_loan, slider_river, slider_shore]), _chart],
                align="center",
            ),
            mo.md(_eq),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Computing Weights with Query and Key

    We train two small neural networks.

    1. One produces a *query* vector for each word.  Think of it as asking "what am I looking for?"
    2. The other produces a *key* vector, answering "what do I contain?"

    We compute the query vector $q_i$ for token $i$ and key vector $k_j$ for token $j$ by a linear transformation of the original embeddings:

    $$
    q_i = {\bf W}_{\text{query}} x_i, \quad
    k_i = {\bf W}_{\text{key}} x_i,
    $$

    where $x_i$ is the input token vector, and ${\bf W}_{\text{query}}$ and ${\bf W}_{\text{key}}$ are learnable weight matrices. Denoted by ${\bf Q}$ and ${\bf K}$ the stack of query and key vectors, respectively, i.e.,

    $$
    {\bf Q} = \begin{bmatrix} q_1 ^\top \\ q_2^\top \\ \vdots \\ q_n^\top \end{bmatrix},
    \quad
    {\bf K} = \begin{bmatrix} k_1^\top \\ k_2^\top \\ \vdots \\ k_n^\top \end{bmatrix},
    $$

    The asssociations between tokens $i$ and $j$ is computed by the dot product of their query and key vectors:

    $$
    \text{score}_{ij} = q_i \cdot k_j, \quad S = {\bf Q} {\bf K}^\top.
    $$

    Scores range from $-\infty$ to $\infty$. But we want to ensure that the attention is positive and sum to one.
    We do so by using *soft-max*:

    $$
    \text{attention} = \text{softmax}({\bf Q} {\bf K}^\top / \sqrt{d}),
    $$

    where for a matrix ${\bf X}=(x_{ij})$, the softmax is given by:

    $$
    \text{softmax}({\bf X}) = \dfrac{\exp(x_{ij})}{\sum_\ell \exp(x_{i\ell})}.
    $$

    Here, we divide by $\sqrt{d}$ to keep the dot products from growing too large in high dimensions, which would push softmax into regions with tiny gradients.
    """)
    return


@app.cell(hide_code=True)
def _(embeddings, heatmap, mo, np, words):
    _scores = embeddings @ embeddings.T
    _exp = np.exp(_scores / np.sqrt(embeddings.shape[1]))
    attn = _exp / _exp.sum(axis=1, keepdims=True)

    _chart_raw = heatmap(_scores, tick_labels=words, title="Raw dot products", width=300, height=300)
    _chart_soft = heatmap(attn, tick_labels=words, title="After softmax", width=300, height=300, vmin=0, vmax=1)

    mo.vstack(
        [
            mo.ui.tabs({"Raw scores": _chart_raw, "After softmax": _chart_soft}),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(
    compute_attention,
    emb2df,
    embeddings,
    heatmap,
    k_bias,
    k_rotation,
    k_scale,
    mo,
    q_bias,
    q_rotation,
    q_scale,
    scatter_plot,
    words,
):
    _orig_q, _tf_q, _W_q, _b_q = emb2df(words, embeddings, q_scale.value, q_rotation.value, q_bias.value)
    _orig_k, _tf_k, _W_k, _b_k = emb2df(words, embeddings, k_scale.value, k_rotation.value, k_bias.value)

    _Q = embeddings @ _W_q + _b_q
    _K = embeddings @ _W_k + _b_k
    _attn = compute_attention(_Q, _K)

    _chart_q = scatter_plot(_tf_q, _orig_q, title="Query (Q)", width=200, height=200)
    _chart_k = scatter_plot(_tf_k, _orig_k, title="Key (K)", width=200, height=200)
    _chart_attn = heatmap(_attn, tick_labels=words, title="Attention weights", width=250, height=250, vmin=0, vmax=1)

    mo.vstack(
        [
            mo.md("Explore how different Q and K transformations change the attention pattern."),
            mo.hstack(
                [
                    mo.vstack([q_scale, q_rotation, q_bias]),
                    mo.vstack([k_scale, k_rotation, k_bias]),
                ],
                align="center",
            ),
            mo.hstack(
                [_chart_q, _chart_k, _chart_attn],
                align="center",
            ),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Remind that:

    $$
    v_{\text{bank}}^{\text{new}} = a_1 \, v_{\text{bank}} + a_2 \, v_{\text{money}} + \cdots
    $$

    We still need another vector, $v$. We do so in the same way as we do for $q$ and $k$:

    $$
    v_i = {\bf W}_{\text{value}} x_i,
    $$

    where ${\bf W}$ is another learnable weight matrix. Denoted by ${\bf V}$ the stack of value vectors, we now have the full description of the attention, i.e.,

    $$
    \text{Attention}(Q,K,V) = \underbrace{\text{softmax}\!\left(\frac{{\bf Q} {\bf K}^\top}{\sqrt{d}}\right)}_{\text{weights (from Q and K)}} \cdot \underbrace{{\bf V}}_{\text{what to average}}
    $$

    The output is an weighted average, with the weights being *learned* from the data instead of set by hand.

    Next, let's see how transformers use multiple attention heads in parallel.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-Head Attention

    A single attention head can only capture one type of relationship. For "bank," one head might learn to attend to financial words, but then it cannot simultaneously attend to geographical words.

    Multi-head attention solves this by running several attention heads in parallel. Here is how it works:

    **1. Split the embedding into sub-vectors.** Each token's $d$-dimensional vector is split into $h$ chunks of size $d/h$. For example, with $d=512$ and $h=8$, each head works with 64-dimensional sub-vectors.

    **2. Each head runs its own attention.** Head $i$ has its own weight matrices ${\bf W}^{(i)}_Q$, ${\bf W}^{(i)}_K$, ${\bf W}^{(i)}_V$ and computes attention independently on its sub-vectors:

    $$
    \text{head}_i = \text{Attention}({\bf Q}_i, {\bf K}_i, {\bf V}_i)
    $$

    One head might learn financial associations for "bank," another geographical ones -- each discovers a different pattern without interference.

    **3. Concatenate and project.** The outputs of all heads are concatenated back into a single $d$-dimensional vector and passed through a linear projection:

    $$
    \text{MultiHead}(Q,K,V) = [\text{head}_1; \text{head}_2; \ldots; \text{head}_h] \, {\bf W}_O
    $$

    **4. Feed-forward network.** The result is then passed through a two-layer MLP (multilayer perceptron) applied independently to each token:

    $$
    \text{FFN}(x) = {\bf W}_2 \, \text{ReLU}({\bf W}_1 x + b_1) + b_2
    $$

    The attention layers move information *between* tokens. The feed-forward layers transform each token's representation *individually* -- adding non-linearity and richer feature combinations that attention alone cannot express.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Residual Stream

    Let's step back and look at how attention fits into the bigger picture. In a transformer, there is a central *residual stream* -- a highway that carries information through every layer.

    Each attention layer does not replace the stream. It computes a small correction and *adds* it back:

    $$
    \text{output} = x + \text{Attention}(x)
    $$

    The network only needs to learn what is *missing* -- the residual -- not reconstruct everything from scratch.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    This framing explains why transformers can stack dozens of layers. Each layer makes a small adjustment. Without residual connections, information would degrade after just a few layers.

    It also helps gradients flow during training -- the addition operation creates a direct path for gradients to travel backward through many layers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Masked Attention -- Causal Models

    When generating text, the model produces one word at a time. At each step, it should only attend to words that came *before* -- never peek ahead.

    Consider translating "I love you" to "Je t'aime." When predicting "t'", the model can see "Je" but not "aime." We enforce this with a mask.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We set attention scores for future positions to $-\infty$ before softmax, which zeros them out. This creates a lower-triangular attention matrix -- each token only attends to itself and earlier tokens.
    """)
    return


@app.cell(hide_code=True)
def _(apply_causal_mask, en_embeddings, en_words, heatmap, mo, np):
    np.random.seed(42)
    _W_q = np.random.randn(2, 2) * 0.5
    _W_k = np.random.randn(2, 2) * 0.5

    _Q = en_embeddings @ _W_q
    _K = en_embeddings @ _W_k
    _scores = _Q @ _K.T / np.sqrt(2)

    # Unmasked
    _exp = np.exp(_scores - np.max(_scores, axis=1, keepdims=True))
    _attn_unmasked = _exp / _exp.sum(axis=1, keepdims=True)

    # Masked
    _attn_masked = apply_causal_mask(_scores)

    _chart_unmasked = heatmap(_attn_unmasked, tick_labels=en_words, title="Unmasked", width=250, height=250, vmin=0, vmax=1)
    _chart_masked = heatmap(
        _attn_masked, tick_labels=en_words, title="Masked (causal)", width=250, height=250, vmin=0, vmax=1
    )

    mo.vstack(
        [
            mo.ui.tabs({"Unmasked": _chart_unmasked, "Masked (causal)": _chart_masked}),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cross-Attention

    In self-attention, Q, K, and V all come from the same sequence. In cross-attention, the query comes from one sequence while keys and values come from another.

    This is how a translation model connects its understanding of the source language to the target it is generating.
    """)
    return


@app.cell(hide_code=True)
def _(
    alt,
    en_embeddings,
    en_words,
    fr_embeddings,
    fr_words,
    heatmap,
    mo,
    np,
    pd,
):
    def _():
        # Hand-picked weights so the alignment is clear:
        # Je -> I, t' -> you, aime -> love
        _W_q_cross = np.array([[2.0, -0.8], [0.5, 2.5]])
        _W_k_cross = np.array([[2.2, 0.3], [-0.6, 2.0]])

        _Q_fr = fr_embeddings @ _W_q_cross
        _K_en = en_embeddings @ _W_k_cross

        _scores_cross = _Q_fr @ _K_en.T / np.sqrt(2)
        _exp_cross = np.exp(_scores_cross - np.max(_scores_cross, axis=1, keepdims=True))
        _attn_cross = _exp_cross / _exp_cross.sum(axis=1, keepdims=True)

        _chart_cross = heatmap(
            _attn_cross,
            tick_labels=en_words,
            title="Cross-attention (French -> English)",
            width=280,
            height=280,
            vmin=0,
            vmax=1,
        )

        # Relabel rows for French words
        _data = []
        for i in range(len(fr_words)):
            for j in range(len(en_words)):
                _data.append({"French": fr_words[i], "English": en_words[j], "value": _attn_cross[i, j]})

        _df_cross = pd.DataFrame(_data)
        _base = (
            alt.Chart(_df_cross)
            .mark_rect(strokeWidth=1, stroke="white")
            .encode(
                x=alt.X("English:N", title="English (K)", sort=en_words),
                y=alt.Y("French:N", title="French (Q)", sort=fr_words),
                color=alt.Color("value:Q", scale=alt.Scale(domain=[0, 1], scheme="inferno")),
            )
        )
        _text = (
            alt.Chart(_df_cross)
            .mark_text(baseline="middle")
            .encode(
                x=alt.X("English:N", sort=en_words),
                y=alt.Y("French:N", sort=fr_words),
                text=alt.Text("value:Q", format=".2f"),
                color=alt.condition(alt.datum.value < 0.5, alt.value("white"), alt.value("black")),
            )
        )
        _cross_chart = (_base + _text).properties(width=280, height=200, title="Cross-attention")
        return mo.vstack(
            [
                _cross_chart,
                mo.md(
                    'Each French word "asks" (via Q) which English words are most relevant. The encoder "answers" (via K).'
                ),
            ],
            align="center",
        )


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Putting It Together

    Here is the full transformer architecture. Input embeddings are combined with positional encodings, then passed through repeated blocks of multi-head self-attention, add-and-norm, and feed-forward layers.

    Variants include encoder-only (BERT), decoder-only (GPT), and encoder-decoder (the original Transformer, T5).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Layer Normalization

    Layer normalization rescales each token vector to have zero mean and unit variance, then applies learnable parameters:

    $$
    \text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
    $$

    This prevents signals from growing or shrinking as they pass through many layers.
    """)
    return


@app.cell(hide_code=True)
def _(
    alt,
    d_model_slider,
    get_positional_encoding,
    mo,
    np,
    pd,
    position_slider,
):
    _seq_len = position_slider.value
    _d_model = d_model_slider.value
    _pos_enc = get_positional_encoding(_seq_len, _d_model)

    # Spiral plot (first 2 dims)
    _df_spiral = pd.DataFrame(
        {
            "position": list(range(_seq_len)),
            "x": _pos_enc[:, 0],
            "y": _pos_enc[:, 1] if _d_model >= 2 else np.zeros(_seq_len),
        }
    )

    _scatter = (
        alt.Chart(_df_spiral)
        .mark_circle(size=100)
        .encode(
            x=alt.X("x", scale=alt.Scale(domain=[-1.1, 1.1])),
            y=alt.Y("y", scale=alt.Scale(domain=[-1.1, 1.1])),
            color=alt.Color("position:O", scale=alt.Scale(scheme="viridis")),
            tooltip=["position", "x", "y"],
        )
    )
    _line = alt.Chart(_df_spiral).mark_line(opacity=0.3, color="gray").encode(x="x", y="y", order="position")
    _spiral_chart = (_scatter + _line).properties(width=300, height=300, title="Positional Encoding (first 2 dims)")

    # Similarity heatmap
    _sim = _pos_enc @ _pos_enc.T
    _sim_data = []
    for _i in range(_seq_len):
        for _j in range(_seq_len):
            _sim_data.append({"pos_i": _i, "pos_j": _j, "similarity": _sim[_i, _j]})
    _sim_df = pd.DataFrame(_sim_data)

    _sim_chart = (
        alt.Chart(_sim_df)
        .mark_rect()
        .encode(
            x=alt.X("pos_i:O", title="Position i"),
            y=alt.Y("pos_j:O", title="Position j"),
            color=alt.Color("similarity:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=["pos_i", "pos_j", "similarity"],
        )
        .properties(width=250, height=250, title="Position similarity")
    )

    mo.vstack(
        [
            mo.md(
                r"""
                ### Positional Encoding

                Attention treats input as a set -- it has no notion of word order. Positional encoding adds position information to each token embedding.

                $$
                PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
                $$
                """
            ),
            mo.hstack([position_slider, d_model_slider]),
            mo.hstack([_spiral_chart, _sim_chart], align="center"),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Further Readings

    - [Attention is All You Need](https://arxiv.org/abs/1706.03762)
    - [3Blue1Brown - Visualizing Attention](https://www.3blue1brown.com/lessons/attention)
    - [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
    - [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
    - [You could have designed state of the art positional encoding](https://huggingface.co/blog/designing-positional-encoding)
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt

    return alt, mo, np, pd


@app.cell(hide_code=True)
def _(np):
    # --- Data ---
    words = ["bank", "money", "loan", "river", "shore"]
    embeddings = (
        np.array(
            [
                [0.0, -0.3],
                [-0.8, -0.3],
                [-0.7, -0.6],
                [0.7, -0.5],
                [0.6, -0.7],
            ]
        )
        * 2
    )

    # Translation tokens for later sections
    en_words = ["I", "love", "you"]
    en_embeddings = np.array([[0.5, 0.8], [-0.3, 0.6], [0.7, -0.2]])
    fr_words = ["Je", "t'", "aime"]
    fr_embeddings = np.array([[0.4, 0.9], [-0.5, 0.3], [0.6, -0.4]])
    return embeddings, en_embeddings, en_words, fr_embeddings, fr_words, words


@app.cell(hide_code=True)
def _(alt, np, pd):
    # --- Helper functions ---


    def scatter_plot(
        df,
        df_original,
        color="#ff7f0e",
        width=300,
        height=300,
        size=100,
        title=None,
        vmax=2,
    ):
        if vmax is None:
            vmax = np.maximum(np.max(np.abs(df["x"])), np.max(np.abs(df["y"])))

        base_original = (
            alt.Chart(df_original)
            .mark_circle(size=size, color="#aaaaaa", opacity=0.8)
            .encode(
                x=alt.X("x", scale=alt.Scale(domain=[-vmax, vmax])),
                y=alt.Y("y", scale=alt.Scale(domain=[-vmax, vmax])),
                tooltip=["word"],
            )
        )
        text_original = (
            alt.Chart(df_original)
            .mark_text(align="left", dx=10, dy=-5, fontSize=14, color="#aaaaaa")
            .encode(x="x", y="y", text="word")
        )
        base = (
            alt.Chart(df)
            .mark_circle(size=size, color=color)
            .encode(
                x=alt.X("x", scale=alt.Scale(domain=[-vmax, vmax])),
                y=alt.Y("y", scale=alt.Scale(domain=[-vmax, vmax])),
                tooltip=["word"],
            )
        )
        vectors = (
            alt.Chart(df)
            .mark_line(color=color, opacity=0.5)
            .encode(
                x=alt.X("x0:Q", scale=alt.Scale(domain=[-vmax, vmax])),
                x2=alt.X2("x:Q"),
                y=alt.Y("y0:Q", scale=alt.Scale(domain=[-vmax, vmax])),
                y2=alt.Y2("y:Q"),
                angle=alt.value(0),
            )
            .transform_calculate(x0="0", y0="0")
        )
        text = alt.Chart(df).mark_text(align="left", dx=10, dy=-5, fontSize=14).encode(x="x", y="y", text="word")
        return (base_original + text_original + base + vectors + text).properties(width=width, height=height, title=title)


    def heatmap(
        matrix,
        tick_labels=None,
        title=None,
        width=300,
        height=300,
        vmin=None,
        vmax=None,
    ):
        data = []
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                data.append(
                    {
                        "x": tick_labels[j] if tick_labels else j,
                        "y": tick_labels[i] if tick_labels else i,
                        "value": value,
                    }
                )
        df = pd.DataFrame(data)
        min_val = df["value"].min() if vmin is None else vmin
        max_val = df["value"].max() if vmax is None else vmax
        domain = [min_val, max_val]

        base = (
            alt.Chart(df)
            .mark_rect(strokeWidth=1, stroke="white")
            .encode(
                x=alt.X("x:N", title="", axis=alt.Axis(labelAngle=45), sort=None),
                y=alt.Y("y:N", title="", sort=None),
                color=alt.Color(
                    "value",
                    scale=alt.Scale(domain=domain, scheme="inferno", clamp=True),
                    legend=alt.Legend(title="Value", orient="right"),
                ),
            )
        )
        text_layer = (
            alt.Chart(df)
            .mark_text(baseline="middle", align="center")
            .encode(
                x="x:N",
                y="y:N",
                text=alt.Text("value:Q", format=".2f"),
                color=alt.condition(
                    alt.datum.value < (domain[1] + domain[0]) / 2,
                    alt.value("white"),
                    alt.value("black"),
                ),
            )
        )
        return (base + text_layer).properties(width=width, height=height, title=title)


    def emb2df(words, embeddings, scale, rotation, bias):
        theta = np.radians(rotation)
        scale_matrix = np.array([[scale, 0], [0, scale]])
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        W = scale_matrix @ rotation_matrix
        b = np.array([bias, bias])
        transformed = embeddings @ W + b
        original_df = pd.DataFrame({"word": words, "x": embeddings[:, 0], "y": embeddings[:, 1]})
        transformed_df = pd.DataFrame({"word": words, "x": transformed[:, 0], "y": transformed[:, 1]})
        return original_df, transformed_df, W, b


    def compute_attention(Q, K):
        d = Q.shape[1]
        scores = Q @ K.T / np.sqrt(d)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


    def apply_causal_mask(scores):
        n = scores.shape[0]
        mask = np.triu(np.ones((n, n)), k=1) * (-1e9)
        masked = scores + mask
        exp_scores = np.exp(masked - np.max(masked, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


    def get_positional_encoding(seq_len, d_model):
        pos_enc = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:
                    pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
        return pos_enc

    return (
        apply_causal_mask,
        compute_attention,
        emb2df,
        get_positional_encoding,
        heatmap,
        scatter_plot,
    )


@app.cell(hide_code=True)
def _(mo):
    slider_bank = mo.ui.slider(0, 1, 0.05, value=0.25, label="bank")
    slider_money = mo.ui.slider(0, 1, 0.05, value=0.25, label="money")
    slider_loan = mo.ui.slider(0, 1, 0.05, value=0.25, label="loan")
    slider_river = mo.ui.slider(0, 1, 0.05, value=0.25, label="river")
    slider_shore = mo.ui.slider(0, 1, 0.05, value=0.25, label="shore")
    return slider_bank, slider_loan, slider_money, slider_river, slider_shore


@app.cell(hide_code=True)
def _(mo):
    q_scale = mo.ui.slider(0.1, 2.5, 0.1, value=1.0, label="Q scale")
    q_rotation = mo.ui.slider(-180, 180, 1, value=0, label="Q rotation")
    q_bias = mo.ui.slider(-1, 1, 0.1, value=0, label="Q bias")
    k_scale = mo.ui.slider(0.1, 2.5, 0.1, value=1.0, label="K scale")
    k_rotation = mo.ui.slider(-180, 180, 1, value=0, label="K rotation")
    k_bias = mo.ui.slider(-1, 1, 0.1, value=0, label="K bias")
    return k_bias, k_rotation, k_scale, q_bias, q_rotation, q_scale


@app.cell(hide_code=True)
def _(mo):
    position_slider = mo.ui.slider(2, 30, 1, value=10, label="Number of positions")
    d_model_slider = mo.ui.slider(2, 100, 1, value=2, label="Embedding dimension")
    return d_model_slider, position_slider


if __name__ == "__main__":
    app.run()
