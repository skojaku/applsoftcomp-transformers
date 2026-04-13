# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.21.1",
#     "numpy==2.4.3",
#     "matplotlib==3.10.1",
#     "torch>=2.0",
#     "transformers>=4.40",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium", css_file="marimo_lecture_note_theme.css")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Steering a Language Model from the Inside

    *An Interactive Guide to Activation Steering*

    [@SadamoriKojaku](https://skojaku.github.io/)

    When you chat with an LLM, you influence its output through your prompt. But what if you could reach inside the model and nudge its internal representations directly? That is the idea behind **activation steering**: instead of asking the model nicely, you add a carefully chosen vector to its hidden states during the forward pass, pushing its behavior in a direction you choose.

    In this module, we will build a steering vector from scratch using SmolLM2 and plain PyTorch. We will see how a simple vector addition can shift the model's personality, making it more positive, more negative, or something else entirely. No special libraries needed, just PyTorch hooks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Core Idea

    A transformer processes text by passing it through a sequence of layers. At each layer, the model maintains a **residual stream**, a vector of numbers that encodes everything the model "knows" about the text so far. The key insight is that directions in this vector space correspond to meaningful concepts.

    Activation steering works in three steps:

    1. **Pick two contrasting prompts.** For example, "Love" and "Hate". These represent the two poles of the behavior you want to control.

    2. **Extract their residual stream activations** at a chosen layer. Each prompt produces a vector in the model's hidden space.

    3. **Compute the difference.** The vector $\mathbf{v}_{\text{steer}} = \mathbf{a}_{\text{positive}} - \mathbf{a}_{\text{negative}}$ points in the direction of the concept you want to amplify.

    During generation, you add $\alpha \cdot \mathbf{v}_{\text{steer}}$ to the residual stream at that layer, where $\alpha$ controls the strength. Positive $\alpha$ pushes toward the positive pole, negative $\alpha$ pushes toward the negative pole, and $\alpha = 0$ leaves the model unchanged.

    $$
    \mathbf{h}'_\ell = \mathbf{h}_\ell + \alpha \cdot \mathbf{v}_{\text{steer}}
    $$

    Let us build this step by step.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: Load the Model

    We will use HuggingFace's SmolLM2-360M-Instruct, a compact language model with 360 million parameters. It is small enough to run comfortably on a CPU (about 720 MB in bfloat16) yet capable enough to produce coherent text that responds visibly to steering. We load it with standard PyTorch and the `transformers` library, no special interpretability frameworks needed.
    """)
    return


@app.cell
def _():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import numpy as np

    torch.set_grad_enabled(False)

    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"Model: {model_name}")
    print(f"Layers: {n_layers}, Hidden dim: {hidden_dim}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    return model, n_layers, np, tokenizer, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: Extract Activations with PyTorch Hooks

    To build a steering vector, we need to peek inside the model and read the hidden states at a specific layer. PyTorch provides a mechanism for this called **forward hooks**. A hook is a function that PyTorch calls every time a module runs its forward pass. We attach a hook to a decoder layer, and it captures the output activations for us.

    The function below extracts the activation of the **last token** at a given layer. Why the last token? In autoregressive models, the last token's hidden state is the one that determines what comes next. It has accumulated the most context from the input.

    Let us look at what the hook mechanism looks like under the hood.

    ```python
    # PyTorch's hook API: register a function on any module
    def my_hook(module, input, output):
        # 'output' is what the layer produced
        captured_activation = output[0].detach()

    handle = model.model.layers[3].register_forward_hook(my_hook)
    model(**inputs)      # hook fires during this forward pass
    handle.remove()      # always clean up!
    ```

    The key idea is that `output[0]` gives us the hidden state tensor of shape `(batch, seq_len, hidden_dim)`. We grab the last token position `[:, -1, :]` to get the vector that will determine the next predicted token.
    """)
    return


@app.cell
def _(model, tokenizer):
    def _get_hidden_state(output):
        """Extract the hidden state tensor from a layer's output.

        Different architectures return different formats:
        some return a plain tensor, others a tuple with the
        hidden state as the first element.
        """
        if isinstance(output, tuple):
            return output[0]
        return output


    def get_activation(prompt, layer):
        """Extract the hidden state of the last token at a given layer."""
        captured = {}

        def hook_fn(module, input, output):
            captured["act"] = _get_hidden_state(output).detach()

        handle = model.model.layers[layer].register_forward_hook(hook_fn)
        inputs = tokenizer(prompt, return_tensors="pt")
        model(**inputs)
        handle.remove()

        # Return last token's activation: shape (hidden_dim,)
        return captured["act"][0, -1, :]


    # Test it
    act_love = get_activation("Love", layer=10)
    act_hate = get_activation("Hate", layer=10)
    print(f"Activation shape: {act_love.shape}")
    print(f"Love activation norm: {act_love.norm().item():.2f}")
    print(f"Hate activation norm: {act_hate.norm().item():.2f}")
    return (get_activation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: Build the Steering Vector

    The steering vector is simply the difference between the two activations. This vector points from the "negative" concept toward the "positive" concept in the model's representation space.

    Try different prompt pairs below. Single words work well ("Love" vs "Hate"), but short phrases can capture more nuanced concepts ("I feel joyful and excited" vs "I feel miserable and bored").
    """)
    return


@app.cell
def _(mo, n_layers):
    layer_slider = mo.ui.slider(0, n_layers - 1, value=n_layers // 2, step=1, label="Layer")
    positive_prompt_input = mo.ui.text(value="Love", label="Positive prompt")
    negative_prompt_input = mo.ui.text(value="Hate", label="Negative prompt")

    mo.hstack([positive_prompt_input, negative_prompt_input, layer_slider])
    return layer_slider, negative_prompt_input, positive_prompt_input


@app.cell
def _(
    get_activation,
    layer_slider,
    mo,
    negative_prompt_input,
    positive_prompt_input,
):
    _layer = layer_slider.value
    _pos = positive_prompt_input.value
    _neg = negative_prompt_input.value

    act_pos = get_activation(_pos, _layer)
    act_neg = get_activation(_neg, _layer)

    # Normalize each activation to unit length before differencing.
    # Without this, the prompt with larger norm dominates the difference
    # and the steering vector carries no information about the other prompt.
    act_pos_normed = act_pos / act_pos.norm()
    act_neg_normed = act_neg / act_neg.norm()
    steering_vector = act_pos_normed - act_neg_normed

    mo.md(
        f"""
    **Steering vector built** from "{_pos}" vs "{_neg}" at layer {_layer}.

    $\\mathbf{{v}} = \\hat{{\\mathbf{{h}}}}_{{\\text{{pos}}}} - \\hat{{\\mathbf{{h}}}}_{{\\text{{neg}}}}$, with norm **{steering_vector.norm().item():.2f}** (activations normalized to unit length before differencing).
    """
    )
    return (steering_vector,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4: Steer the Model

    Now for the fun part. We attach a hook that adds our steering vector to the chosen layer during generation. PyTorch calls this hook on every forward pass, which means the steering is applied at every token generation step.

    We wrap the hook in a Python context manager so it is automatically cleaned up after generation finishes. This is important: a forgotten hook would silently alter every future forward pass.

    ```python
    @contextmanager
    def steering_hook(model, layer_idx, vector, coeff):
        def hook(module, input, output):
            return (output[0] + coeff * vector,) + output[1:]

        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()  # always clean up
    ```

    Adjust the coefficient $\alpha$ below. At $\alpha = 0$, the model behaves normally. Positive values push toward the positive prompt's meaning, negative values push the other way.
    """)
    return


@app.cell
def _(mo):
    coeff_slider = mo.ui.slider(-550, 550, value=0, step=0.1, label="Steering coefficient α")
    generation_prompt = mo.ui.text(value="I think this movie is", label="Generation prompt")
    max_tokens_slider = mo.ui.slider(10, 100, value=20, step=10, label="Max new tokens")

    mo.hstack([generation_prompt, coeff_slider, max_tokens_slider])
    return coeff_slider, generation_prompt, max_tokens_slider


@app.cell
def _(
    coeff_slider,
    generation_prompt,
    layer_slider,
    max_tokens_slider,
    model,
    steering_vector,
    tokenizer,
):
    from contextlib import contextmanager


    @contextmanager
    def apply_steering(model, layer_idx, vector, coeff):
        """Context manager that steers a layer's output during forward passes."""

        def hook(module, input, output):
            if isinstance(output, tuple):
                return (output[0] + coeff * vector,) + output[1:]
            return output + coeff * vector

        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()


    _layer = layer_slider.value
    _coeff = coeff_slider.value
    _prompt = generation_prompt.value
    _max_tokens = max_tokens_slider.value

    inputs = tokenizer(_prompt, return_tensors="pt")

    # Generate with steering
    with apply_steering(model, _layer, steering_vector, _coeff):
        steered_ids = model.generate(**inputs, max_new_tokens=_max_tokens, temperature=0.5, do_sample=True)
    steered_text = tokenizer.decode(steered_ids[0], skip_special_tokens=True)

    # Generate without steering for comparison
    baseline_ids = model.generate(**inputs, max_new_tokens=_max_tokens, temperature=0.5, do_sample=True)
    baseline_text = tokenizer.decode(baseline_ids[0], skip_special_tokens=True)

    print(f"=== Baseline ===\n{baseline_text}\n")
    print(f"=== Steered (a = {_coeff}) ===\n{steered_text}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualizing the Effect on Token Probabilities

    Steering does not just change the final text. It reshapes the entire probability distribution over the vocabulary at each step. Let us look at the top predicted tokens for the next word, with and without steering, to see exactly how the vector shifts the model's preferences.
    """)
    return


@app.cell
def _(
    coeff_slider,
    generation_prompt,
    layer_slider,
    mo,
    model,
    np,
    plt,
    steering_vector,
    tokenizer,
    torch,
):
    _layer = layer_slider.value
    _coeff = coeff_slider.value
    _prompt = generation_prompt.value

    _inputs = tokenizer(_prompt, return_tensors="pt")

    # Get logits without steering
    baseline_logits = model(**_inputs).logits[0, -1, :]
    baseline_probs = torch.softmax(baseline_logits.float(), dim=-1)


    # Get logits with steering
    def _steer_hook(module, input, output):
        if isinstance(output, tuple):
            return (output[0] + _coeff * steering_vector,) + output[1:]
        return output + _coeff * steering_vector


    handle = model.model.layers[_layer].register_forward_hook(_steer_hook)
    steered_logits = model(**_inputs).logits[0, -1, :]
    handle.remove()
    steered_probs = torch.softmax(steered_logits.float(), dim=-1)

    # Get top-10 tokens from the union of both distributions
    top_baseline = torch.topk(baseline_probs, 10).indices
    top_steered = torch.topk(steered_probs, 10).indices
    top_tokens = list(set(top_baseline.tolist() + top_steered.tolist()))
    top_tokens.sort(key=lambda t: steered_probs[t].item(), reverse=True)
    top_tokens = top_tokens[:15]

    token_labels = [tokenizer.decode([t]).strip() or f"[{t}]" for t in top_tokens]
    base_vals = [baseline_probs[t].item() for t in top_tokens]
    steer_vals = [steered_probs[t].item() for t in top_tokens]

    x = np.arange(len(top_tokens))
    width = 0.35

    fig_prob, ax_prob = plt.subplots(figsize=(10, 4))
    ax_prob.bar(x - width / 2, base_vals, width, label="Baseline", color="#4dabf7", edgecolor="#333", linewidth=0.5)
    ax_prob.bar(
        x + width / 2, steer_vals, width, label=f"Steered (a={_coeff})", color="#ff8787", edgecolor="#333", linewidth=0.5
    )
    ax_prob.set_xticks(x)
    ax_prob.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=9)
    ax_prob.set_ylabel("Probability")
    ax_prob.set_title(f'Next-token probabilities after: "{_prompt}"')
    ax_prob.legend()
    plt.tight_layout()

    mo.as_html(fig_prob)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How Layer Choice Matters

    Not all layers are equally useful for steering. Early layers encode low-level features (token identity, position). Middle layers encode more abstract concepts (sentiment, topic). Late layers are close to the output and can be too disruptive to modify.

    The visualization below computes the steering vector at every layer and measures two things: its magnitude (how different the two prompts are at that layer) and its cosine similarity with the output embedding vectors for a few sentiment-related tokens. Higher similarity means that layer's steering vector is more "aligned" with pushing specific tokens in or out of the output.
    """)
    return


@app.cell
def _(
    get_activation,
    mo,
    model,
    n_layers,
    negative_prompt_input,
    np,
    plt,
    positive_prompt_input,
    tokenizer,
    torch,
):
    _pos = positive_prompt_input.value
    _neg = negative_prompt_input.value

    # The unembedding matrix maps hidden states to vocabulary logits
    unembed_weight = model.lm_head.weight.detach()

    # Compute steering vectors at each layer
    norms_by_layer = []
    probe_tokens = ["good", "bad", "love", "hate", "great", "terrible"]
    cosine_with_unembed = {tok: [] for tok in probe_tokens}

    for layer_i in range(n_layers):
        _act_p = get_activation(_pos, layer_i)
        _act_n = get_activation(_neg, layer_i)
        # Per-activation normalization before differencing
        _sv = _act_p / _act_p.norm() - _act_n / _act_n.norm()
        _sv_normed = _sv / _sv.norm()
        norms_by_layer.append(_sv.norm().item())

        for token_str in probe_tokens:
            token_id = tokenizer.encode(" " + token_str, add_special_tokens=False)[0]
            unembed_vec = unembed_weight[token_id]
            unembed_normed = unembed_vec / unembed_vec.norm()
            cos_sim = torch.dot(_sv_normed.float(), unembed_normed.float()).item()
            cosine_with_unembed[token_str].append(cos_sim)

    fig_layer, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    layers = np.arange(n_layers)
    ax1.bar(layers, norms_by_layer, color="#51cf66", edgecolor="#333", linewidth=0.5)
    ax1.set_ylabel("Steering vector norm")
    ax1.set_title(f'Steering vector magnitude by layer ("{_pos}" vs "{_neg}")')

    for token_str, sims in cosine_with_unembed.items():
        ax2.plot(layers, sims, marker="o", markersize=4, label=f'"{token_str}"')
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Cosine similarity with output embedding")
    ax2.set_title("Alignment of steering vector with token output directions")
    ax2.legend(fontsize=8, ncol=3)
    ax2.axhline(0, color="#888", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    mo.as_html(fig_layer)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experimenting with Different Axes

    Sentiment is just one axis you can steer along. Try changing the positive and negative prompts above to explore other directions. Here are some ideas to try:

    **Formality:** "formal" vs "casual"

    **Confidence:** "certainly" vs "maybe"

    **Topic shift:** "science" vs "sports"

    **Creativity:** "imagine" vs "recall"

    Each pair of prompts defines a different direction in the model's activation space, and each produces a qualitatively different effect on the generated text. The model's internal geometry encodes far more structure than any single prompt can reveal.

    ::: {.callout-tip title="Try it yourself"}
    Experiment with extreme values of $\alpha$ (like 15 or $-$15). What happens to the text? At some point, the steering overwhelms the model's natural dynamics and the output degenerates into nonsense. Finding the sweet spot, where steering is strong enough to matter but gentle enough to preserve coherence, is part of the art.
    :::
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What Is Really Happening?

    Activation steering works because transformer residual streams form a roughly linear representation space. Concepts like sentiment, topic, and style correspond to directions in this space. When we add a steering vector, we are performing a linear intervention: shifting the model's internal state along a meaningful direction without retraining.

    This is a powerful idea with deep connections to mechanistic interpretability. If we can find the directions that encode specific concepts, we can not only steer the model but also understand what it has learned. Steering vectors are one of the simplest tools in the activation engineering toolkit, and they hint at a future where we control AI systems not just through their inputs and outputs, but through their internal representations.

    The limitations are real. Steering with a single vector is blunt. It affects every token position equally. It assumes linearity in a system that is fundamentally nonlinear. And the choice of layer, prompt pair, and coefficient all matter in ways that are not fully understood. But as a demonstration of what is possible when you open the hood, it is compelling.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


if __name__ == "__main__":
    app.run()
