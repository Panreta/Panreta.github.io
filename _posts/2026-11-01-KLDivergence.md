---
layout: post
title: 'KL-Divergence and Entropy'
date: 2026-11-01 10:47
description: "The origin of care divergence and why it has such property."
tags: [Machine Learning, Reinforcement Learning,Artificial Intelligence]
categories: Machine Learning
tabs: true
---
$$
B_\varphi(p, q) = \varphi(p) - \varphi(q) - \langle \nabla\varphi(q),\, p - q \rangle
$$

Choose $\varphi(p) = \sum_i p_i \log p_i$, $\; p \in \mathbb{R}^n$:(The negative Shannon entropy is actually seems a positive one, but if you consider the value is negative..)

$$
\nabla\varphi(q) = \big(\log q_i + 1\big)_{i \in [n]}
$$

$$
\begin{aligned}
B_\varphi(p, q) &= \varphi(p) - \underbrace{\varphi(q)} - \sum_i p_i \log q_i + \underbrace{\sum_i q_i \log q_i} \\
&= \sum_i p_i \log \frac{p_i}{q_i}. 
\end{aligned}
$$
This is the KL divergence.

Since when p is fixed, you don't have to consider constant in the optimization.

$$
Cross-Entropy(p, q) = - \sum_i p_i \log q_i
.\;
$$

Besides, since $\varphi(.)$ is a convex function, obviously
$$B_\varphi(p, q) = \varphi(p) - \varphi(q) - \langle \nabla\varphi(q),\, p - q \rangle\ge0.$$
It's easy to figure out by Talor expansion and geometrically understanding the  tangent is always under the convex epi-graph.

On KL-divergence:
$$\sum_i p_i \log p_i\ge\sum_i p_i \log q_i.$$
It's also the inequality shows Shannon entropy is the optimal encoding.
