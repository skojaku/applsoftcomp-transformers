---
title: "GPT Sampling Exercises"
---

# GPT Sampling Exercises

Softmax converts logits $z = [z_1, \dots, z_V]$ into probabilities:

$$P(i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

where temperature $T > 0$ scales the logits before softmax.

A model outputs logits $z = [1, 0, -1]$ for tokens $\{a, b, c\}$.

In a Colab notebook, implement softmax with temperature and compute the full probability distribution for $T \in \{0.5, 1, 2, 5\}$. Plot $P(\text{token})$ vs. $T$ for each token and describe what happens to the distribution as $T$ increases. Submit your notebook link.

## Exercise 3: Nucleus Sampling

Nucleus sampling ($p$) includes the fewest tokens whose cumulative probability reaches $p$, then renormalizes. The number of included tokens depends on how peaked the distribution is.

**Peaked distribution:** $P(a)=0.6,\ P(b)=0.2,\ P(c)=0.1,\ P(d)=0.1$.

**Flat distribution:** $P(a)=0.35,\ P(b)=0.25,\ P(c)=0.22,\ P(d)=0.18$.

1. For $p = 0.7$, how many tokens are included in each distribution?

## Exercise 4: Beam Search (Second-Order Markov)

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

**Hint for 2:** At step 3, note that $\ln P(a \mid b, a) = -0.11$ (very confident, $P \approx 0.9$), while the greedy path's step-3 log-prob is much worse. Even though $b$ has a lower step-1 log-prob than $a$, the strong continuation after $(b, a)$ makes $b\!\to\!a\!\to\!a$ the best overall path.
