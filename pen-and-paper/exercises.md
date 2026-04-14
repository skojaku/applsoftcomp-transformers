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

**Colab link:** _______________________________________________

**In one sentence, what happens to the distribution as $T$ increases?**

&nbsp;

&nbsp;

## Exercise 2: Top-$k$ and Top-$p$ Sampling

Top-$k$ keeps the $k$ highest-probability tokens and sets the rest to zero. Top-$p$ (nucleus sampling) keeps the fewest tokens whose cumulative probability reaches $p$.

A model produces logits over a 10-token vocabulary:

$$z = [3.0,\ 2.0,\ 1.5,\ 1.0,\ 0.5,\ 0.0,\ -0.5,\ -1.0,\ -1.5,\ -2.0]$$

Implement both sampling methods and use them to answer the following. For each method, your function should return the renormalized probability distribution over all tokens (zero for excluded tokens).

1. Apply top-$k$ with $k = 3$. What are the sampling probabilities for all 10 tokens?
2. Apply top-$p$ with $p = 0.8$. How many tokens are included, and what are their sampling probabilities?
3. Now set $T = 2$ and recompute top-$p$ with $p = 0.8$. Does the number of included tokens change?

**Colab link:** _______________________________________________

**Q1 answer** (sampling probabilities, top-3 only, others zero):

&nbsp;

**Q2 answer** (number of tokens included, their probabilities):

&nbsp;

**Q3 answer** (does the count change, and why?):

&nbsp;

&nbsp;

## Exercise 3: Beam Search (Second-Order Markov)

Vocab $\{a, b\}$, three steps. Step 3 depends on both previous tokens.

Step 1 (log-probs, base $e$):

| Token | $\ln P(t_1 \mid \text{start})$ |
|-------|------|
| a     | $-0.51$ |
| b     | $-0.92$ |

Step 2 (first-order):

| From | $\ln P(a)$ | $\ln P(b)$ |
|------|-----------|-----------|
| a    | $-1.20$   | $-0.36$   |
| b    | $-0.22$   | $-1.61$   |

Step 3 (second-order):

| From    | $\ln P(a)$ | $\ln P(b)$ |
|---------|-----------|-----------|
| (a, a)  | $-1.61$   | $-0.22$   |
| (a, b)  | $-0.92$   | $-0.51$   |
| (b, a)  | $-0.11$   | $-2.30$   |
| (b, b)  | $-0.92$   | $-0.51$   |

1. Trace greedy decoding ($B=1$). Report the sequence and its cumulative log-probability (round to two decimals).
2. Trace beam search ($B=3$) for all three steps. At each step, list all candidate sequences with their cumulative log-probabilities and identify which three are kept. Report the top sequence and its log-probability.

**Q1 answer** (sequence and log-probability):

&nbsp;

**Q2 answer** (beams kept at each step, final top sequence):

&nbsp;

&nbsp;

&nbsp;

**Hint for 2:** At step 3, note that $\ln P(a \mid b, a) = -0.11$ (very confident, $P \approx 0.9$), while the greedy path's step-3 log-prob is much worse. Even though $b$ has a lower step-1 log-prob than $a$, the strong continuation after $(b, a)$ makes $b\!\to\!a\!\to\!a$ the best overall path.
