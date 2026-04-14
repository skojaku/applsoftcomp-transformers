# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch==2.6.0",
#     "transformers==4.49.0",
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

    The model gives us a conditional probability distribution over the next token, but finding the *best* full sequence is intractable. Instead, we use **sampling strategies** to generate text. Let's explore them interactively.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Sampling Strategies

    **Greedy sampling** always picks the highest-probability token. It is deterministic but often repetitive.

    **Beam search** tracks the $k$ best sequences at each step, expanding and pruning to find higher-probability outputs.

    **Top-k sampling** restricts the candidate pool to the $k$ most likely tokens, then samples randomly among them.

    **Nucleus (top-p) sampling** dynamically selects the smallest set of tokens whose cumulative probability exceeds threshold $p$. When the model is confident, fewer tokens qualify. When uncertain, more do.

    **Temperature** $\\tau$ controls randomness by scaling logits before softmax:

    $$p_i = \\frac{\\exp(z_i / \\tau)}{\\sum_j \\exp(z_j / \\tau)}$$

    Low $\\tau$ sharpens the distribution (more deterministic). High $\\tau$ flattens it (more diverse).

    These techniques can be combined: e.g., top-k + top-p + temperature.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Load Model

    We use **SmolLM2-135M**, a compact language model that runs comfortably on CPU.
    """)
    return


@app.cell
def _():
    import torch
    from transformers import pipeline

    _device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    generator = pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-135M",
        device=_device,
    )
    return (generator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Interactive Text Generation")
    return


@app.cell(hide_code=True)
def _(mo):
    prompt_input = mo.ui.text(
        value="Once upon a time",
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
    )
    num_beams_slider = mo.ui.slider(
        start=2, stop=10, step=1, value=5,
        label="Number of beams",
    )
    top_k_slider = mo.ui.slider(
        start=1, stop=50, step=1, value=10,
        label="Top-k",
    )
    top_p_slider = mo.ui.slider(
        start=0.1, stop=1.0, step=0.05, value=0.95,
        label="Top-p",
    )
    temperature_slider = mo.ui.slider(
        start=0.1, stop=3.0, step=0.1, value=1.0,
        label="Temperature",
    )
    generate_button = mo.ui.run_button(label="Generate")
    return (
        generate_button,
        max_tokens_slider,
        method_select,
        num_beams_slider,
        prompt_input,
        temperature_slider,
        top_k_slider,
        top_p_slider,
    )


@app.cell(hide_code=True)
def _(
    generate_button,
    max_tokens_slider,
    method_select,
    mo,
    num_beams_slider,
    prompt_input,
    temperature_slider,
    top_k_slider,
    top_p_slider,
):
    # Show controls conditionally based on method
    _method = method_select.value
    _controls = [prompt_input, method_select, max_tokens_slider]

    if _method == "Beam Search":
        _controls.append(num_beams_slider)
    elif _method == "Top-k":
        _controls.extend([top_k_slider, temperature_slider])
    elif _method == "Top-p (Nucleus)":
        _controls.extend([top_p_slider, temperature_slider])
    elif _method == "Top-k + Top-p + Temperature":
        _controls.extend([top_k_slider, top_p_slider, temperature_slider])

    _controls.append(generate_button)

    mo.vstack(_controls)
    return


@app.cell
def _(
    generate_button,
    generator,
    max_tokens_slider,
    method_select,
    mo,
    num_beams_slider,
    prompt_input,
    temperature_slider,
    top_k_slider,
    top_p_slider,
):
    mo.stop(not generate_button.value, mo.md("*Click **Generate** to see output.*"))

    _method = method_select.value
    _prompt = prompt_input.value
    _max_tokens = max_tokens_slider.value
    _pad_id = generator.tokenizer.eos_token_id

    if _method == "Greedy":
        _outputs = generator(
            _prompt, do_sample=False, max_new_tokens=_max_tokens,
            pad_token_id=_pad_id,
        )
    elif _method == "Beam Search":
        _n = num_beams_slider.value
        _outputs = generator(
            _prompt, do_sample=False, max_new_tokens=_max_tokens,
            pad_token_id=_pad_id, num_beams=_n, num_return_sequences=_n,
        )
    elif _method == "Top-k":
        _outputs = generator(
            _prompt, do_sample=True, max_new_tokens=_max_tokens,
            pad_token_id=_pad_id, top_k=top_k_slider.value,
            temperature=temperature_slider.value,
        )
    elif _method == "Top-p (Nucleus)":
        _outputs = generator(
            _prompt, do_sample=True, max_new_tokens=_max_tokens,
            pad_token_id=_pad_id, top_p=top_p_slider.value,
            temperature=temperature_slider.value,
        )
    else:  # Combined
        _outputs = generator(
            _prompt, do_sample=True, max_new_tokens=_max_tokens,
            pad_token_id=_pad_id, top_k=top_k_slider.value,
            top_p=top_p_slider.value, temperature=temperature_slider.value,
        )

    _result_lines = []
    for i, out in enumerate(_outputs):
        _text = out["generated_text"]
        if len(_outputs) > 1:
            _result_lines.append(f"**Beam {i+1}:** {_text}")
        else:
            _result_lines.append(_text)

    mo.callout(mo.md("\n\n".join(_result_lines)), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    **Greedy** is deterministic but repetitive. **Beam search** explores multiple paths but is still deterministic. **Top-k** and **top-p** introduce controlled randomness, and **temperature** tunes how much. In practice, these are combined (e.g., top-k=50, top-p=0.95, temperature=0.7) to balance coherence and diversity.

    /// tip | Try it yourself
    Experiment with the controls above. Try greedy first to see the repetition problem, then switch to top-k or nucleus sampling. What happens as you crank up the temperature?
    ///
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
