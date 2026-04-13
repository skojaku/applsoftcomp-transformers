# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.21.1",
#     "numpy==2.4.3",
#     "matplotlib==3.10.1",
#     "torch>=2.0",
#     "transformer-lens>=2.8",
#     "einops",
#     "accelerate",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium", css_file="marimo_lecture_note_theme.css")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Steering a Language Model from the Inside

    *An Interactive Guide to Activation Steering*

    [@SadamoriKojaku](https://skojaku.github.io/)

    When you chat with an LLM, you influence its output through your prompt. But what if you could reach inside the model and nudge its internal representations directly? That is the idea behind **activation steering**: instead of asking the model nicely, you add a carefully chosen vector to its hidden states during the forward pass, pushing its behavior in a direction you choose.

    In this module, we will build a steering vector from scratch using Gemma-2-2B and TransformerLens. We will see how a simple vector addition can shift the model's personality, making it more positive, more negative, or something else entirely.
    """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Step 1: Load the Model

    We will use Google's Gemma-2-2B (2.6 billion parameters) via TransformerLens, a library designed for mechanistic interpretability. TransformerLens wraps Hugging Face models and exposes hooks at every layer, making it easy to read and modify internal activations. Gemma-2-2B is small enough to run on a single GPU (or CPU, slowly) but powerful enough to produce coherent text that responds clearly to steering.
    """
    )


@app.cell
def _():
    import transformer_lens
    from transformer_lens import HookedTransformer
    import torch
    import numpy as np

    torch.set_grad_enabled(False)

    model = HookedTransformer.from_pretrained("gemma-2-2b")
    print(f"Model: {model.cfg.model_name}")
    print(f"Layers: {model.cfg.n_layers}, Hidden dim: {model.cfg.d_model}")
    return model, torch, np, transformer_lens, HookedTransformer


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Step 2: Extract Activations from Contrasting Prompts

    To build a steering vector, we need two prompts that represent opposite ends of some behavioral axis. Let us start with a simple sentiment axis: "Love" versus "Hate". We run each prompt through the model and cache the residual stream activations at every layer.

    The function below extracts the activation of the **last token** at a given layer. Why the last token? In autoregressive models like Gemma, the last token's residual stream is the one that determines what token comes next. It has accumulated the most context.
    """
    )


@app.cell
def _(model, torch):
    def get_activation(prompt, layer):
        """Extract the residual stream activation of the last token at a given layer."""
        _, cache = model.run_with_cache(prompt)
        # Shape: (batch, seq_len, d_model) -> take last token
        activation = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]
        return activation

    # Test it
    act_love = get_activation("Love", layer=13)
    act_hate = get_activation("Hate", layer=13)
    print(f"Activation shape: {act_love.shape}")
    print(f"Love activation norm: {act_love.norm().item():.2f}")
    print(f"Hate activation norm: {act_hate.norm().item():.2f}")
    return (get_activation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Step 3: Build the Steering Vector

    The steering vector is simply the difference between the two activations. This vector points from the "negative" concept toward the "positive" concept in the model's representation space.
    """
    )


@app.cell
def _(mo):
    layer_slider = mo.ui.slider(0, 25, value=13, step=1, label="Layer")
    positive_prompt_input = mo.ui.text(value="Love", label="Positive prompt")
    negative_prompt_input = mo.ui.text(value="Hate", label="Negative prompt")

    mo.hstack([positive_prompt_input, negative_prompt_input, layer_slider])


@app.cell
def _(get_activation, layer_slider, positive_prompt_input, negative_prompt_input, np, mo):
    _layer = layer_slider.value
    _pos = positive_prompt_input.value
    _neg = negative_prompt_input.value

    act_pos = get_activation(_pos, _layer)
    act_neg = get_activation(_neg, _layer)
    steering_vector = act_pos - act_neg

    # Normalize for consistent steering strength across different prompt pairs
    steering_vector_normed = steering_vector / steering_vector.norm()

    mo.md(
        f"""
    **Steering vector built** from "{_pos}" vs "{_neg}" at layer {_layer}.

    The raw steering vector has norm **{steering_vector.norm().item():.2f}**. We normalize it to unit length so the coefficient $\\alpha$ directly controls the magnitude of the nudge.
    """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Step 4: Steer the Model

    Now for the fun part. We hook into the model's forward pass and add our steering vector at the chosen layer. TransformerLens makes this straightforward: we define a hook function that modifies the residual stream in-place, then run generation with that hook active.

    Adjust the coefficient $\alpha$ below. At $\alpha = 0$, the model behaves normally. Positive values push toward the positive prompt's meaning, negative values push the other way.
    """
    )


@app.cell
def _(mo):
    coeff_slider = mo.ui.slider(-20, 20, value=0, step=1, label="Steering coefficient α")
    generation_prompt = mo.ui.text(value="I think this movie is", label="Generation prompt")
    max_tokens_slider = mo.ui.slider(10, 100, value=50, step=10, label="Max new tokens")

    mo.hstack([generation_prompt, coeff_slider, max_tokens_slider])


@app.cell
def _(model, steering_vector_normed, layer_slider, coeff_slider, generation_prompt, max_tokens_slider, transformer_lens):
    _layer = layer_slider.value
    _coeff = coeff_slider.value
    _prompt = generation_prompt.value
    _max_tokens = max_tokens_slider.value

    # Define the steering hook
    def steering_hook(activation, hook):
        # activation shape: (batch, seq_len, d_model)
        # Add steering vector to all token positions
        activation[:, :, :] += _coeff * steering_vector_normed
        return activation

    # Generate with the hook
    hook_name = f"blocks.{_layer}.hook_resid_post"
    steered_output = model.generate(
        _prompt,
        max_new_tokens=_max_tokens,
        temperature=0.7,
        fwd_hooks=[(hook_name, steering_hook)],
    )
    steered_text = model.tokenizer.decode(steered_output[0])

    # Also generate without steering for comparison
    baseline_output = model.generate(
        _prompt,
        max_new_tokens=_max_tokens,
        temperature=0.7,
    )
    baseline_text = model.tokenizer.decode(baseline_output[0])

    print(f"=== Baseline (α = 0) ===\n{baseline_text}\n")
    print(f"=== Steered (α = {_coeff}) ===\n{steered_text}")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Visualizing the Effect on Token Probabilities

    Steering does not just change the final text. It reshapes the entire probability distribution over the vocabulary at each step. Let us look at the top predicted tokens for the next word, with and without steering, to see exactly how the vector shifts the model's preferences.
    """
    )


@app.cell
def _(model, steering_vector_normed, layer_slider, coeff_slider, generation_prompt, torch, np, plt, mo):
    _layer = layer_slider.value
    _coeff = coeff_slider.value
    _prompt = generation_prompt.value

    # Get logits without steering
    baseline_logits = model(_prompt)[0, -1, :]  # last token logits
    baseline_probs = torch.softmax(baseline_logits, dim=-1)

    # Get logits with steering
    hook_name_viz = f"blocks.{_layer}.hook_resid_post"

    def _steer_hook(activation, hook):
        activation[:, :, :] += _coeff * steering_vector_normed
        return activation

    steered_logits = model.run_with_hooks(
        _prompt,
        fwd_hooks=[(hook_name_viz, _steer_hook)],
    )[0, -1, :]
    steered_probs = torch.softmax(steered_logits, dim=-1)

    # Get top-10 tokens from the union of both distributions
    top_baseline = torch.topk(baseline_probs, 10).indices
    top_steered = torch.topk(steered_probs, 10).indices
    top_tokens = list(set(top_baseline.tolist() + top_steered.tolist()))
    top_tokens.sort(key=lambda t: steered_probs[t].item(), reverse=True)
    top_tokens = top_tokens[:15]

    token_labels = [model.tokenizer.decode([t]).strip() for t in top_tokens]
    base_vals = [baseline_probs[t].item() for t in top_tokens]
    steer_vals = [steered_probs[t].item() for t in top_tokens]

    x = np.arange(len(top_tokens))
    width = 0.35

    fig_prob, ax_prob = plt.subplots(figsize=(10, 4))
    ax_prob.bar(x - width / 2, base_vals, width, label="Baseline", color="#4dabf7", edgecolor="#333", linewidth=0.5)
    ax_prob.bar(x + width / 2, steer_vals, width, label=f"Steered (α={_coeff})", color="#ff8787", edgecolor="#333", linewidth=0.5)
    ax_prob.set_xticks(x)
    ax_prob.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=9)
    ax_prob.set_ylabel("Probability")
    ax_prob.set_title(f'Next-token probabilities after: "{_prompt}"')
    ax_prob.legend()
    plt.tight_layout()

    mo.as_html(fig_prob)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## How Layer Choice Matters

    Not all layers are equally useful for steering. Early layers encode low-level features (token identity, position). Middle layers encode more abstract concepts (sentiment, topic). Late layers are close to the output and can be too disruptive to modify.

    The visualization below shows the cosine similarity between the steering vector at each layer and the final layer's unembedding directions for a few sentiment-related tokens. Higher similarity means that layer's steering vector is more "aligned" with pushing specific tokens in or out of the output.
    """
    )


@app.cell
def _(model, get_activation, positive_prompt_input, negative_prompt_input, torch, np, plt, mo):
    _pos = positive_prompt_input.value
    _neg = negative_prompt_input.value

    # Compute steering vectors at each layer
    norms_by_layer = []
    cosine_with_unembed = {token: [] for token in ["good", "bad", "love", "hate", "great", "terrible"]}

    for layer_i in range(model.cfg.n_layers):
        _act_p = get_activation(_pos, layer_i)
        _act_n = get_activation(_neg, layer_i)
        _sv = _act_p - _act_n
        _sv_normed = _sv / _sv.norm()
        norms_by_layer.append(_sv.norm().item())

        # Check alignment with specific token embeddings in the unembedding matrix
        for token_str in cosine_with_unembed:
            token_id = model.tokenizer.encode(" " + token_str)[0]
            unembed_vec = model.W_U[:, token_id]
            unembed_normed = unembed_vec / unembed_vec.norm()
            cos_sim = torch.dot(_sv_normed, unembed_normed).item()
            cosine_with_unembed[token_str].append(cos_sim)

    fig_layer, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    layers = np.arange(model.cfg.n_layers)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## What Is Really Happening?

    Activation steering works because transformer residual streams form a roughly linear representation space. Concepts like sentiment, topic, and style correspond to directions in this space. When we add a steering vector, we are performing a linear intervention: shifting the model's internal state along a meaningful direction without retraining.

    This is a powerful idea with deep connections to mechanistic interpretability. If we can find the directions that encode specific concepts, we can not only steer the model but also understand what it has learned. Steering vectors are one of the simplest tools in the activation engineering toolkit, and they hint at a future where we control AI systems not just through their inputs and outputs, but through their internal representations.

    The limitations are real. Steering with a single vector is blunt. It affects every token position equally. It assumes linearity in a system that is fundamentally nonlinear. And the choice of layer, prompt pair, and coefficient all matter in ways that are not fully understood. But as a demonstration of what is possible when you open the hood, it is compelling.
    """
    )


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
