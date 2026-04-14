---
title: "GPT Sampling Exercises"
---

# GPT Sampling Exercises

Work through the exercises below. For each one, write your answers in the space provided and submit this sheet as your exit ticket before leaving.

Softmax converts logits $z = [z_1, \dots, z_V]$ into probabilities:

$$P(i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

where temperature $T > 0$ scales the logits before softmax.

A model outputs logits $z = [1, 0, -1]$ for tokens $\{a, b, c\}$.

Implement softmax with temperature and compute the full probability distribution for $T \in \{0.5, 1, 2, 5\}$. Plot $P(\text{token})$ vs. $T$ for each token and describe what happens to the distribution as $T$ increases.

## Exercise 2: Top-$k$ and Top-$p$ Sampling

Top-$k$ keeps the $k$ highest-probability tokens and sets the rest to zero. Top-$p$ (nucleus sampling) keeps the fewest tokens whose cumulative probability reaches $p$.

A model produces logits over a 10-token vocabulary:

$$z = [3.0,\ 2.0,\ 1.5,\ 1.0,\ 0.5,\ 0.0,\ -0.5,\ -1.0,\ -1.5,\ -2.0]$$

Implement both sampling methods and use them to answer the following. For each method, your function should return the renormalized probability distribution over all tokens (zero for excluded tokens).

1. Apply top-$k$ with $k = 3$. What are the sampling probabilities for all 10 tokens?
2. Apply top-$p$ with $p = 0.8$. How many tokens are included, and what are their sampling probabilities?
3. Now set $T = 2$ and recompute top-$p$ with $p = 0.8$. Does the number of included tokens change?

## Exercise 3: Decoding Strategies

Both greedy and beam search are *decoding* strategies — they select tokens over multiple steps without sampling. Greedy always picks the single highest-probability token ($B=1$). Beam search keeps the $B$ best sequences at each step, trading compute for a better global solution.

Vocab $\{a, b, c, d, e\}$, three steps. Transition probabilities follow $p(1-p)^d$ normalized per row, where $d = |i-j|$ is the distance between token positions and $p = 0.5$. Step 3 conditions only on the second previous token $t_1$, not $t_2$.

Step 1 (log-probs from start):

| Token | $\ln P(t_1 \mid \text{start})$ |
|-------|------|
| a | $-2.30$ |
| b | $-1.61$ |
| c | $-0.92$ |
| d | $-1.61$ |
| e | $-2.30$ |

Step 2 (first-order). Step 3 uses the same table, indexed by $t_1$:

| From | $\ln P(a)$ | $\ln P(b)$ | $\ln P(c)$ | $\ln P(d)$ | $\ln P(e)$ |
|------|----------|----------|----------|----------|----------|
| a    | $-0.66$ | $-1.35$ | $-2.05$ | $-2.74$ | $-3.43$ |
| b    | $-1.56$ | $-0.86$ | $-1.56$ | $-2.25$ | $-2.94$ |
| c    | $-2.30$ | $-1.61$ | $-0.92$ | $-1.61$ | $-2.30$ |
| d    | $-2.94$ | $-2.25$ | $-1.56$ | $-0.86$ | $-1.56$ |
| e    | $-3.43$ | $-2.74$ | $-2.05$ | $-1.35$ | $-0.66$ |

1. Greedy ($B=1$): at each step pick the highest-probability token. Report the sequence and cumulative log-probability.
2. Beam search ($B=3$): at each step expand all beams, keep the top 3 sequences. List the candidates at each step and the final top sequence.

**Q1 answer** (sequence and log-probability):

&nbsp;

**Q2 answer** (beams kept at each step, final top sequence):

&nbsp;

&nbsp;

&nbsp;

**Hint:** The table is symmetric — tokens closer to the source are always more likely. For step 3, look up the row for $t_1$ (the first token generated), not $t_2$.
