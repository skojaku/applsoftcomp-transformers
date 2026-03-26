# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair==6.0.0",
#     "marimo>=0.21.1",
#     "numpy==2.4.3",
#     "pandas==3.0.1",
#     "scikit-learn==1.6.1",
#     "torch==2.6.0",
#     "tqdm==4.67.1",
#     "transformers==4.49.0",
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

    Consider the word "bank." A static embedding gives it one fixed position, but its meaning shifts depending on context. Is it a financial institution or the side of a river?

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
    mo.md(r"""
    ## Variations of Attention

    So far, every token attends to every other token (self-attention). Two important variations change *who* can attend to *whom*.

    **Masked (causal) attention.** When generating text, the model produces one token at a time and must never peek ahead. We set attention scores for future positions to $-\infty$ before softmax, which zeros them out. This creates a lower-triangular attention matrix where each token only attends to itself and earlier tokens.

    **Cross-attention.** In self-attention, Q, K, and V all come from the same sequence. In cross-attention, the query comes from one sequence while keys and values come from another. This is how a translation model connects its understanding of the source language to the target it is generating.
    """)
    return


@app.cell(hide_code=True)
def _(
    alt,
    apply_causal_mask,
    en_embeddings,
    en_words,
    fr_embeddings,
    fr_words,
    heatmap,
    mo,
    np,
    pd,
):
    # --- Causal masking ---
    np.random.seed(42)
    _W_q = np.random.randn(2, 2) * 0.5
    _W_k = np.random.randn(2, 2) * 0.5

    _Q = en_embeddings @ _W_q
    _K = en_embeddings @ _W_k
    _scores = _Q @ _K.T / np.sqrt(2)

    _exp = np.exp(_scores - np.max(_scores, axis=1, keepdims=True))
    _attn_unmasked = _exp / _exp.sum(axis=1, keepdims=True)
    _attn_masked = apply_causal_mask(_scores)

    _chart_unmasked = heatmap(_attn_unmasked, tick_labels=en_words, title="Self-attention (unmasked)", width=250, height=250, vmin=0, vmax=1)
    _chart_masked = heatmap(_attn_masked, tick_labels=en_words, title="Causal (masked)", width=250, height=250, vmin=0, vmax=1)

    # --- Cross-attention ---
    _W_q_cross = np.array([[2.0, -0.8], [0.5, 2.5]])
    _W_k_cross = np.array([[2.2, 0.3], [-0.6, 2.0]])

    _Q_fr = fr_embeddings @ _W_q_cross
    _K_en = en_embeddings @ _W_k_cross

    _scores_cross = _Q_fr @ _K_en.T / np.sqrt(2)
    _exp_cross = np.exp(_scores_cross - np.max(_scores_cross, axis=1, keepdims=True))
    _attn_cross = _exp_cross / _exp_cross.sum(axis=1, keepdims=True)

    _data = []
    for _i in range(len(fr_words)):
        for _j in range(len(en_words)):
            _data.append({"French": fr_words[_i], "English": en_words[_j], "value": _attn_cross[_i, _j]})

    _df_cross = pd.DataFrame(_data)
    _base = (
        alt.Chart(_df_cross)
        .mark_rect(strokeWidth=1, stroke="white")
        .encode(
            x=alt.X("English:N", title="English (K)", sort=en_words),
            y=alt.Y("French:N", title="French (Q)", sort=fr_words),
            color=alt.Color("value:Q", scale=alt.Scale(domain=[0, 1], scheme="inferno"), legend=None),
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
    _chart_cross = (_base + _text).properties(width=250, height=200, title="Cross-attention (Fr → En)")

    mo.vstack(
        [
            mo.hstack([_chart_masked, _chart_cross], align="center"),
            mo.md(
                "**Left:** Causal mask forces each token to only attend to earlier tokens. "
                "**Right:** Cross-attention lets French tokens query English tokens."
            ),
        ],
        align="center",
    )
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

    One head might learn financial associations for "bank," another geographical ones. Each discovers a different pattern without interference.

    **3. Concatenate and project.** The outputs of all heads are concatenated back into a single $d$-dimensional vector and passed through a linear projection:

    $$
    \text{MultiHead}(Q,K,V) = [\text{head}_1; \text{head}_2; \ldots; \text{head}_h] \, {\bf W}_O
    $$

    **4. Feed-forward network.** The result is then passed through a two-layer MLP (multilayer perceptron) applied independently to each token:

    $$
    \text{FFN}(x) = {\bf W}_2 \, \text{ReLU}({\bf W}_1 x + b_1) + b_2
    $$

    The attention layers move information *between* tokens. The feed-forward layers transform each token's representation *individually*, adding non-linearity and richer feature combinations that attention alone cannot express.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Residual Stream

    Let's step back and look at how attention fits into the bigger picture. In a transformer, there is a central *residual stream*, a highway that carries information through every layer.

    Each attention layer does not replace the stream. It computes a small correction and *adds* it back:

    $$
    \text{output} = x + \text{Attention}(x)
    $$

    The network only needs to learn what is *missing*, the residual, not reconstruct everything from scratch.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    This framing explains why transformers can stack dozens of layers. Each layer makes a small adjustment. Without residual connections, information would degrade after just a few layers.

    It also helps gradients flow during training. The addition operation creates a direct path for gradients to travel backward through many layers.
    """)
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
def _(mo):
    mo.md(r"""
    ### Positional Encoding

    Attention treats input as a set with no notion of word order. The sentence "dog bites man" and "man bites dog" would produce the same attention scores. We need to inject position information into each token.

    The simplest idea: add the position number directly to the embedding.

    $$
    x_i' = x_i + i \quad (i = 0, 1, 2, \ldots)
    $$
    """)
    return


@app.cell(hide_code=True)
def _(heatmap, mo, np, position_slider):
    _n = position_slider.value

    # --- Attempt 1: integer positions ---
    _int_enc = np.arange(_n).reshape(-1, 1).astype(float)
    _int_sim = _int_enc @ _int_enc.T
    _chart_int = heatmap(
        _int_sim,
        tick_labels=[str(_i) for _i in range(_n)],
        title="Integer encoding: dot products",
        width=280,
        height=280,
    )

    mo.vstack(
        [
            mo.hstack([position_slider]),
            _chart_int,
            mo.md(
                r"""
                The dot product $i \cdot j$ grows with position. Token 20 attending to token 19 gets a score of 380, while token 2 attending to token 1 gets only 2. Later positions dominate attention simply because their numbers are bigger, not because they are more relevant.
                """
            ),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Attempt 2: binary encoding.** Represent each position as a binary vector, e.g., position 5 = $(1, 0, 1)$.

    This fixes the magnitude problem since every vector has entries in $\{0, 1\}$, but introduces a new one: the similarity between positions has no smooth structure.
    """)
    return


@app.cell(hide_code=True)
def _(heatmap, mo, np, position_slider):
    _n = position_slider.value
    _bits = max(int(np.ceil(np.log2(_n + 1))), 1)
    _bin_enc = np.array([list(map(int, format(_i, f'0{_bits}b'))) for _i in range(_n)], dtype=float)
    _bin_sim = _bin_enc @ _bin_enc.T

    _chart_bin = heatmap(
        _bin_sim,
        tick_labels=[str(_i) for _i in range(_n)],
        title="Binary encoding: dot products",
        width=280,
        height=280,
    )

    mo.vstack(
        [
            _chart_bin,
            mo.md(
                r"""
                Positions 4 and 7 (binary $100$ and $111$) have a dot product of 1, while positions 6 and 7 ($110$ and $111$) have a dot product of 2. The similarity pattern is jagged and discontinuous. Nearby positions are not necessarily more similar than distant ones.
                """
            ),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What we actually want:**
    1. Bounded values, so position does not dominate the embedding magnitude.
    2. Smooth similarity, so nearby positions have similar encodings.
    3. Unique encoding for every position, even for sequences longer than seen during training.

    The sinusoidal positional encoding achieves all three. Like binary encoding, each dimension oscillates between values, but instead of discrete $\{0,1\}$ it uses continuous sine and cosine waves at different frequencies:

    $$
    PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
    $$

    Each dimension is a wave with a different wavelength, ranging from $2\pi$ (fastest) to $10000 \cdot 2\pi$ (slowest). This creates a smooth, bounded encoding where the dot product between two positions depends only on their distance.
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
    _spiral_chart = (_scatter + _line).properties(width=300, height=300, title="Sinusoidal encoding (first 2 dims)")

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
        .properties(width=250, height=250, title="Sinusoidal: dot products")
    )

    mo.vstack(
        [
            mo.hstack([d_model_slider]),
            mo.hstack([_spiral_chart, _sim_chart], align="center"),
            mo.md(
                "The similarity is highest on the diagonal and decays smoothly with distance. "
                "Values are bounded between $-1$ and $1$. Every position gets a unique encoding."
            ),
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


# ============================================================
# BERT
# ============================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # BERT: Bidirectional Encoder Representations from Transformers

    ELMo addressed polysemy with bidirectional LSTMs. BERT advances this by using transformers and self-attention, significantly improving context understanding.

    Due to its powerful capabilities in tasks like question answering and text classification, BERT has become foundational in NLP, even enhancing Google's search engine.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Architecture of BERT

    BERT consists of 12 stacked encoder transformer layers. No decoder is used.

    Each layer progressively refines token embeddings, making them increasingly context-aware and effective for NLP tasks.

    ## Pre-training BERT

    BERT is pre-trained on massive text datasets like Wikipedia and BooksCorpus. During this phase, BERT learns language patterns, context, and semantic relationships without human supervision.

    BERT is pre-trained on two objectives:

    - **Masked Language Modeling (MLM)**: Some tokens are masked, and the model must predict the masked tokens. Three ways to mask: replace with `[MASK]` (80%), replace with a random word (10%), or keep unchanged (10%).
    - **Next Sentence Prediction (NSP)**: Predict whether two sentences are consecutive. A `[CLS]` token at the start and `[SEP]` tokens between/after sentences structure the input. The `[CLS]` embedding is used for the prediction.

    ## Let's play with BERT

    The best way to understand BERT is to play with it.

    We will use the [transformers](https://huggingface.co/docs/transformers/index) library to load the model and the tokenizer.
    """)
    return


@app.cell(hide_code=True)
def _():
    from transformers import AutoTokenizer, AutoModel

    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    bert_model = bert_model.eval()
    return bert_model, bert_tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    - `transformers` library provides easy access to NLP models
    - `AutoTokenizer` and `AutoModel` automatically load the right tokenizer and model
    - `model.eval()` prepares the model for inference

    ## Tokenizer

    The tokenizer breaks text into tokens that BERT can understand.
    """)
    return


@app.cell
def _(bert_tokenizer):
    text = "Binghamton University"

    tokens = bert_tokenizer.tokenize(text, add_special_tokens=True)
    token_ids = bert_tokenizer.convert_tokens_to_ids(tokens)

    print(f"Text: '{text}'")
    print(f"Tokenized: {tokens}")
    print(f"Token IDs: {token_ids}")
    return (token_ids,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Embedding tokens

    Now the preparation is done. Let us feed the token IDs to the model and get the embeddings of the tokens.
    """)
    return


@app.cell
def _(bert_model, token_ids):
    import torch

    token_ids_tensor = torch.tensor([token_ids])
    bert_outputs = bert_model(token_ids_tensor, output_hidden_states=True)
    return bert_outputs, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The `outputs` contains:

    1. **Input token embeddings** before the first transformer module
    2. **Output token embeddings** after each transformer module
    3. **Attention scores** of the tokens

    There are 13 tensors in `outputs.hidden_states` (1 input embedding + 12 transformer outputs). Each tensor has the shape `(batch_size, sequence_length, hidden_size)`.
    """)
    return


@app.cell(hide_code=True)
def _(bert_outputs):
    bert_outputs.hidden_states
    return


@app.cell
def _(bert_outputs):
    bert_last_hidden_state = bert_outputs.hidden_states[-1]
    print(f"Shape: {bert_last_hidden_state.shape}")
    return (bert_last_hidden_state,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Each token embedding can be retrieved by indexing:
    """)
    return


@app.cell
def _(bert_last_hidden_state):
    token_position = 3
    token_embedding = bert_last_hidden_state[0, token_position, :]
    print(token_embedding[:10])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Batch processing

    Processing multiple sentences one by one is inefficient. We can process them in a batch.

    The challenge: sentences have different lengths, so we pad shorter sentences with `[PAD]` tokens and use an attention mask to ignore padding.
    """)
    return


@app.cell
def _(bert_tokenizer):
    bert_text1 = "Binghamton University"
    bert_text2 = "State University of New York"

    tokens1 = bert_tokenizer.tokenize(bert_text1, add_special_tokens=True)
    tokens2 = bert_tokenizer.tokenize(bert_text2, add_special_tokens=True)

    token_ids1 = bert_tokenizer.convert_tokens_to_ids(tokens1)
    token_ids2 = bert_tokenizer.convert_tokens_to_ids(tokens2)

    print(f"Token IDs of text1: {token_ids1}")
    print(f"Token IDs of text2: {token_ids2}")
    return bert_text1, bert_text2


@app.cell
def _(bert_text1, bert_text2, bert_tokenizer):
    bert_inputs = bert_tokenizer(
        [bert_text1, bert_text2],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    print(bert_inputs)
    return (bert_inputs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Now we feed the padded sequences and the attention mask into the model. Notice the first dimension is `2` because we have two sentences.
    """)
    return


@app.cell
def _(bert_inputs, bert_model):
    bert_outputs_batch = bert_model(**bert_inputs, output_hidden_states=True)
    bert_last_hidden_batch = bert_outputs_batch.hidden_states[-1]
    print(f"Last hidden state batch shape: {bert_last_hidden_batch.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Case study: Polysemy resolution

    Consider two meanings of a word (e.g., "apple" as fruit vs. company). We use BERT to embed sentences containing the polysemous word and visualize whether BERT separates the senses.

    We use [CoarseWSD-20](https://github.com/danlou/bert-disambiguation/tree/master/data/CoarseWSD-20), a dataset with polysemous words and their sense labels.
    """)
    return


@app.cell(hide_code=True)
def _(pd):
    from sklearn.decomposition import PCA

    def load_wsd_data(focal_word, is_train, n_samples=100):
        data_type = "train" if is_train else "test"
        data_file = f"https://raw.githubusercontent.com/danlou/bert-disambiguation/master/data/CoarseWSD-20/{focal_word}/{data_type}.data.txt"
        label_file = f"https://raw.githubusercontent.com/danlou/bert-disambiguation/master/data/CoarseWSD-20/{focal_word}/{data_type}.gold.txt"

        data_table = pd.read_csv(
            data_file, sep="\t", header=None,
            dtype={"word_pos": int, "sentence": str},
            names=["word_pos", "sentence"],
        )
        label_table = pd.read_csv(
            label_file, sep="\t", header=None,
            dtype={"label": int}, names=["label"],
        )
        combined_table = pd.concat([data_table, label_table], axis=1)
        return combined_table.sample(n_samples)

    focal_word = "apple"
    wsd_train_data = load_wsd_data(focal_word, is_train=True)
    return PCA, wsd_train_data


@app.cell(hide_code=True)
def _(wsd_train_data):
    wsd_train_data.head(10)
    return


@app.cell(hide_code=True)
def _(mo, slider_bert_layer):
    mo.vstack([
        mo.md("Choose the layer for the embedding, and see how the embedding changes."),
        slider_bert_layer,
    ])
    return


@app.cell
def _(bert_model, bert_tokenizer, torch, wsd_train_data):
    from collections import defaultdict

    _batch_size = 128
    wsd_all_labels = []
    wsd_all_sentences = []
    wsd_all_embeddings = defaultdict(list)

    for _i in range(0, len(wsd_train_data), _batch_size):
        _batch = wsd_train_data.iloc[_i : _i + _batch_size]
        _batch_sentences = _batch["sentence"].tolist()
        _batch_focal_indices = _batch["word_pos"].tolist()
        _batch_labels = _batch["label"].tolist()

        _encoded = bert_tokenizer(
            _batch_sentences, padding=True, truncation=True,
            return_tensors="pt", add_special_tokens=True,
        )
        _out = bert_model(**_encoded, output_hidden_states=True)

        for _layer_id in range(len(_out.hidden_states)):
            _focal_embs = [
                _out.hidden_states[_layer_id][_idx, _focal_pos, :]
                for _idx, _focal_pos in enumerate(_batch_focal_indices)
            ]
            wsd_all_embeddings[_layer_id] += _focal_embs

        wsd_all_labels = wsd_all_labels + _batch_labels
        wsd_all_sentences = wsd_all_sentences + _batch_sentences

    for _layer_id in wsd_all_embeddings.keys():
        wsd_all_embeddings[_layer_id] = (
            torch.vstack(wsd_all_embeddings[_layer_id]).detach().numpy()
        )
    return wsd_all_embeddings, wsd_all_labels, wsd_all_sentences


@app.cell(hide_code=True)
def _(
    PCA,
    alt,
    pd,
    slider_bert_layer,
    wsd_all_embeddings,
    wsd_all_labels,
    wsd_all_sentences,
):
    _pca = PCA(n_components=2, random_state=42)
    _xy = _pca.fit_transform(wsd_all_embeddings[slider_bert_layer.value])

    _df_chart = pd.DataFrame(
        {"x": _xy[:, 0], "y": _xy[:, 1], "label": wsd_all_labels, "sentence": wsd_all_sentences}
    )

    _chart = (
        alt.Chart(_df_chart)
        .mark_circle(size=120)
        .encode(
            x=alt.X("x:Q", title="PCA 1"),
            y=alt.Y("y:Q", title="PCA 2"),
            color=alt.Color("label:N", legend=alt.Legend(title="Label")),
            tooltip=["label", "sentence"],
        )
        .properties(width=700, height=500, title="Word Embeddings Visualization")
        .interactive()
    )
    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Case Study 2: Implicit associations in language

    Human language contains implicit associations between concepts that reflect cultural norms, stereotypes, and common patterns.

    We set up a MLM task using the following template:

    > "Choose a color from red, blue, green, yellow, brown, black, white, purple, orange, pink to describe the color of {object}. Color: [MASK]."

    BERT will predict the masked token. Since no color word appears explicitly, the prediction reflects BERT's learned understanding of the object.

    We use `BertForMaskedLM`, a version of the model with a language modeling head on top.
    """)
    return


@app.cell(hide_code=True)
def _():
    from transformers import BertForMaskedLM

    bert_masked_lm = BertForMaskedLM.from_pretrained("bert-base-uncased")
    return (bert_masked_lm,)


@app.cell
def _(bert_masked_lm, bert_tokenizer, torch):
    def predict_masked_word(template, object_name, top_k=5):
        _text = template.format(object=object_name)
        _inputs = bert_tokenizer(_text, return_tensors="pt")
        _mask_idx = torch.where(_inputs["input_ids"] == bert_tokenizer.mask_token_id)[1]

        with torch.no_grad():
            _outputs = bert_masked_lm(**_inputs)

        _logits = _outputs.logits
        _mask_logits = _logits[0, _mask_idx, :]
        _top_k_ids = torch.topk(_mask_logits, top_k, dim=1).indices[0].tolist()
        _top_k_words = [bert_tokenizer.convert_ids_to_tokens(_tid) for _tid in _top_k_ids]
        return _top_k_words

    return (predict_masked_word,)


@app.cell(hide_code=True)
def _(mo, noun_placeholder, predict_masked_word):
    _template = "Choose a color from red, blue, green, yellow, brown, black, white, purple, orange, pink to describe the color of {object}. Color: [MASK]."
    _top_k = 5
    _obj = noun_placeholder.value
    _predictions = predict_masked_word(_template, _obj, _top_k)
    _results = f"**{_obj}**: {', '.join(_predictions)}"

    mo.vstack([noun_placeholder, _results])
    return


# --- Logic cells (imports, data, helpers, UI definitions) ---


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


@app.cell(hide_code=True)
def _(mo):
    slider_bert_layer = mo.ui.slider(0, 12, 1, 4, label="Layer to use")
    noun_placeholder = mo.ui.text(
        value="banana",
        label="When asked about its color, {object} is described as [MASK].",
        full_width=True,
    )
    return noun_placeholder, slider_bert_layer


if __name__ == "__main__":
    app.run()
