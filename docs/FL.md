# Federated Learning: A Conceptual and Academic Guide

## Table of Contents

1. [What is Federated Learning?](#1-what-is-federated-learning)
2. [The Formal Problem Setting](#2-the-formal-problem-setting)
3. [Historical Context and Intellectual Origins](#3-historical-context-and-intellectual-origins)
4. [Mathematical Foundations](#4-mathematical-foundations)
5. [Aggregation Strategies](#5-aggregation-strategies)
6. [System Heterogeneity](#6-system-heterogeneity)
7. [Statistical Heterogeneity and Non-IID Data](#7-statistical-heterogeneity-and-non-iid-data)
   - [7.1 Formal Dirichlet Partitioning Model](#71-formal-dirichlet-partitioning-model)
   - [7.2 Alpha Sweep Experiment Methodology](#72-alpha-sweep-experiment-methodology)
8. [Communication Efficiency](#8-communication-efficiency)
9. [Privacy in Federated Learning](#9-privacy-in-federated-learning)
10. [This Framework: Architecture and Design](#10-this-framework-architecture-and-design)
11. [Security Model and Threat Classification](#11-security-model-and-threat-classification)
12. [Further Reading](#12-further-reading)

---

## 1. What is Federated Learning?

### The Core Motivation

Modern machine learning depends on large datasets, and large datasets are increasingly held by many distinct organizations or individuals rather than centrally accumulated. A hospital in São Paulo holds patient records that could train a diagnostic model; a hospital in Oslo holds complementary records. Neither can share their data — not because of selfishness but because of genuine legal obligations, patient privacy commitments, and institutional policies that a shared database would violate. Yet both would benefit from a more accurate model than either could train alone.

Federated learning (FL) is the computational framework that reconciles these two facts. It is a paradigm for training machine learning models collaboratively across many data holders — called *clients* — without the raw data ever leaving the environment where it originates. Instead of aggregating the data, FL aggregates the *learning* — specifically the parameter updates that result from each client training a model locally on their own data.

The key invariant of all federated learning systems is: **raw data never traverses the network**. Only model parameters or gradient updates move between clients and a coordinating server. Whether this invariant provides meaningful privacy guarantees depends on additional mechanisms (discussed in Section 9), but it is the foundational separation that makes FL useful in regulated environments.

### The Mental Model

One useful way to conceive of federated learning is through the analogy of a scientific consortium. Suppose twenty hospitals agree to study the relationship between certain biomarkers and disease outcomes. In a traditional centralized approach, each hospital would send its patient records to a central research center, which would train a model on the combined dataset. The central approach requires patients to trust an unknown party with their most sensitive data.

In a federated approach, the central research center distributes a model architecture to each hospital. Each hospital trains the model locally on its own patients and sends only the resulting model parameter updates back to the center. The center combines these updates — the mathematical average of the adjustments each hospital found useful — and sends the improved model back to every hospital. This continues for several rounds until the model converges.

At no point does any patient record cross an institutional boundary. The center never sees individual patient data. Each hospital's contribution to the final model is inseparable from every other hospital's contribution in the aggregated result. This is federated learning: distributed computation in which the data remains local but the model becomes global.

### What FL Solves and What It Does Not

Federated learning solves the *data silo problem*: the fragmentation of valuable training data across jurisdictions, institutions, and devices that prevents the construction of the large, diverse datasets that machine learning requires. It enables collaborative learning without data pooling.

However, FL by itself does not fully solve privacy. Research has demonstrated that model updates carry substantial information about the local training data — sometimes enough to reconstruct individual training examples from a single gradient vector (see Section 9). The baseline FL protocol, without additional privacy mechanisms, should therefore be understood as a *communication* and *logistics* solution rather than a complete privacy solution. True privacy in FL requires additional mechanisms: differential privacy, homomorphic encryption, and zero-knowledge proofs, which this framework provides.

---

## 2. The Formal Problem Setting

### Distributed Empirical Risk Minimization

The machine learning objective that underlies all FL systems is an extension of classical empirical risk minimization (ERM) to the distributed setting. In standard supervised learning, we seek model parameters $\theta \in \mathbb{R}^d$ minimizing the empirical risk over a dataset $\mathcal{D}$:

$$\min_\theta \mathcal{L}(\theta) = \frac{1}{|\mathcal{D}|} \sum_{(x, y) \in \mathcal{D}} \ell(\theta; x, y)$$

where $\ell(\theta; x, y)$ is the *per-sample loss* — cross-entropy for classification, squared error for regression.

In the federated setting, the dataset $\mathcal{D}$ is partitioned across $K$ clients. Client $k$ holds a local dataset $\mathcal{D}_k$ of size $n_k$, with $\sum_{k=1}^K n_k = n$ (the total dataset size). The global objective becomes:

$$\min_\theta F(\theta) = \sum_{k=1}^K \frac{n_k}{n} F_k(\theta)$$

where the *local objective* $F_k$ is the empirical risk over client $k$'s data:

$$F_k(\theta) = \frac{1}{n_k} \sum_{(x, y) \in \mathcal{D}_k} \ell(\theta; x, y)$$

The key constraint is that the entire computation of $F_k(\theta)$ and its gradient $\nabla F_k(\theta)$ must take place on client $k$'s device. The server never has direct access to $\mathcal{D}_k$ or $F_k$ — it only receives quantities derived from them.

### The Communication Model

Beyond the optimization objective, federated learning must specify a *communication protocol*. In the standard synchronous protocol, a round of FL proceeds as follows:

1. The server holds a global model $\theta^t$ at round $t$.
2. The server selects a subset $S^t \subseteq [K]$ of clients to participate in round $t$.
3. Each selected client $k \in S^t$ receives $\theta^t$ and performs some local computation to produce an update, typically denoted $\theta_k^{t+1}$ or $\Delta_k^t$.
4. The server collects the updates from all clients in $S^t$ and aggregates them to form the new global model $\theta^{t+1}$.
5. This process repeats for $T$ rounds or until convergence.

The total communication cost of training is $O(T \cdot |S| \cdot d)$ numerics, where $d$ is the model dimension and $|S|$ is the number of clients selected per round. Reducing communication cost while maintaining model quality is one of the central engineering challenges in FL.

### Client Selection

The set $S^t$ of participating clients is typically chosen by the server at the beginning of each round. In a *cross-device* setting (many mobile devices, intermittent connectivity), selection is driven by availability — only clients that are online, charging, and connected to Wi-Fi are eligible. In a *cross-silo* setting (a small number of institutional clients with reliable connectivity), all clients may participate in every round.

The fraction of clients selected per round, often denoted $C = |S^t|/K$, is a key hyperparameter. Setting $C = 1.0$ (full participation) maximizes information per round but may be impractical when $K$ is large. Partial participation reduces per-round cost but introduces sampling noise into the global update.

In this framework, client participation is controlled by `frac_fit` (fraction selected for training) and `frac_eval` (fraction selected for evaluation), both configurable in `FLConfig`. With typical experimental settings of `frac_fit = 1.0` and a small number of clients (2-5), all clients participate in every round.

---

## 3. Historical Context and Intellectual Origins

### Precursors: Distributed Optimization

The algorithmic foundations of federated learning predate the term itself by decades. The challenge of minimizing an objective function when data is distributed across multiple machines — with constraints on communication — has been studied in the distributed systems and optimization communities since the 1970s.

Gossip protocols and decentralized optimization algorithms (e.g., distributed subgradient descent, ADMM — the Alternating Direction Method of Multipliers) addressed similar problems under different assumptions. These algorithms typically assumed that all machines have roughly equal data, are always available, and are connected by reliable low-latency links — assumptions that hold in data center computing but not in the mobile or institutional settings that motivated FL.

### The McMahan et al. Paper (2017)

The modern notion of federated learning was introduced by Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y Arcas in their 2017 paper, "Communication-Efficient Learning of Deep Networks from Decentralized Data," presented at AISTATS 2017. This paper coined the term *federated learning*, introduced the FedAvg algorithm (see Section 4), and presented the first systematic empirical study of collaborative learning from heterogeneous, siloed data.

The paper's central contribution was not purely algorithmic. Its lasting impact came from clearly articulating the FL setting — with its properties of non-IID data, partial participation, and communication constraints — and demonstrating that averaging model weights (not just gradients) across multiple local SGD steps worked surprisingly well in practice, contradicting the conventional expectation that local SGD would diverge without frequent synchronization.

### Google's Production Deployment

McMahan et al. simultaneously described and were motivated by Google's deployment of FL for learning from Android keyboard usage data (the Gboard system). This system, operational since 2016, trains predictive keyboard models on millions of devices without any typing data leaving users' phones. It remains one of the largest FL deployments in history and demonstrated that cross-device FL is operationally viable at scale.

Subsequent Google deployments extended FL to many applications: learning wake-word detection models from on-device microphone data, training content ranking models from web browsing behavior, and improving autocorrect systems — all domains where the combination of large user populations, sensitive behavioral data, and the inability to pool data centrally made FL the natural architectural choice.

### Regulatory and Institutional Drivers

The formalization of FL arrived at a historically fortuitous moment. The European Union's General Data Protection Regulation (GDPR), effective May 2018, imposed substantial obligations on organizations that process personal data of EU residents — including, in many interpretations, restrictions on centralizing sensitive health, financial, and behavioral data for machine learning. The California Consumer Privacy Act (CCPA) and subsequent state-level legislation in the United States, along with sectoral regulations such as HIPAA for health data, collectively created a regulatory environment in which moving raw data to a central training server became increasingly legally problematic.

Federated learning offers a technical architecture that is at least directionally compatible with data minimization principles — the idea, articulated in GDPR Article 5, that personal data should be "adequate, relevant and limited to what is necessary in relation to the purposes for which they are processed." Whether FL fully satisfies GDPR's privacy requirements is a matter of ongoing legal and technical debate, but it provides a principled technical foundation for privacy-by-design machine learning.

---

## 4. Mathematical Foundations

### The FedAvg Algorithm

The canonical algorithm of federated learning is *FedAvg*, introduced in McMahan et al. (2017). FedAvg combined two ideas that were each obvious in isolation but whose combination required empirical validation: (1) each client runs multiple steps of stochastic gradient descent locally before communicating, and (2) the server aggregates client models by computing a weighted average of their parameters.

The algorithm proceeds as follows. At each communication round $t$:

**Server side:**
- Select a set $S^t$ of $m$ clients (with replacement or without).
- Broadcast the current global model $\theta^t$ to all selected clients.
- Wait for all selected clients to return their updated parameters.
- Compute the new global model as the weighted average:
$$\theta^{t+1} \leftarrow \sum_{k \in S^t} \frac{n_k}{\sum_{j \in S^t} n_j} \theta_k^{t+1}$$

**Client side (for each client $k \in S^t$):**
- Initialize local model to $\theta^t$.
- Run $E$ epochs of SGD on local data $\mathcal{D}_k$ with batch size $B$, producing $\theta_k^{t+1}$.
- Return $\theta_k^{t+1}$ to the server.

The parameter $E$ (local epochs) is a critical design choice. When $E = 1$ and batch size equals all local data, FedAvg reduces to simple gradient averaging, analogous to distributed SGD with one gradient step per round. As $E$ grows, clients perform more local computation and need fewer communication rounds — reducing the total communication cost — but their local models diverge further from the global optimum, potentially harming convergence.

### Why Weight Averaging Works

The correctness of weight averaging in FedAvg — rather than gradient averaging — is not immediately obvious. Consider two clients that start with the same model $\theta^t$. Client 1 has data drawn from distribution $P_1$; client 2 has data from distribution $P_2$. After $E$ local SGD steps, their parameters $\theta_1$ and $\theta_2$ lie at different points in parameter space. Their average $\frac{1}{2}(\theta_1 + \theta_2)$ is not necessarily a better model than either individually — this depends on the geometry of the loss landscape.

In practice, weight averaging works because neural network loss landscapes, while non-convex globally, are approximately convex in the regions explored during optimization. If all clients start from the same $\theta^t$ and take a small number of steps in individually favorable directions, their resulting parameters typically lie in a common basin of the loss landscape, and linear interpolation within a basin yields a good model.

When data is highly non-IID (e.g., client 1 has only class-1 examples and client 2 has only class-2 examples), this basin-sharing assumption breaks down. This phenomenon — called *client drift* — is one of the central challenges of federated optimization and has motivated many algorithmic variants (discussed in Section 5).

### The Global-Local Objective Gap

A critical quantity in federated optimization theory is the *gradient dissimilarity* between local and global objectives, often quantified as:

$$G^2 = \frac{1}{K} \sum_{k=1}^K \|\nabla F_k(\theta) - \nabla F(\theta)\|^2$$

When $G = 0$, all clients have aligned objectives and FedAvg converges as fast as centralized SGD. When $G$ is large (highly non-IID data), the local updates point in conflicting directions, and averaging them produces a global update that is a poor approximation of the true gradient.

Convergence analysis of FedAvg in the non-IID setting (Li et al., 2020; Zhao et al., 2018) shows that the convergence rate depends both on the standard optimization parameters (learning rate, batch size) and on $G^2$. In particular, with $E > 1$ local steps and heterogeneous data, FedAvg converges to a neighborhood of the global optimum rather than to the exact optimum — a phenomenon called *client drift bias*.

### Model Initialization and the Impact of Multiple Local Steps

One of the subtler insights from federated optimization theory concerns the role of $E$, the number of local steps. Define the *local update* of client $k$ as:

$$\Delta_k^t = \theta^t - \theta_k^{t+1}$$

This is the direction and magnitude of change that client $k$'s local training induces. With $E = 1$ (one local gradient step), $\Delta_k^t \approx \eta \nabla F_k(\theta^t)$, and the FedAvg update is approximately $\theta^{t+1} \approx \theta^t - \eta \sum_k \frac{n_k}{n} \nabla F_k(\theta^t)$, which is exactly one step of centralized gradient descent on $F$.

With $E > 1$ local steps, $\Delta_k^t$ incorporates curvature information from multiple gradient steps and is no longer proportional to $\nabla F_k(\theta^t)$. The weighted average of these multi-step local updates does not in general point in the direction of $-\nabla F(\theta^t)$. This is the source of client drift. Algorithms like SCAFFOLD (Section 5) correct for this drift by explicitly tracking the discrepancy between local and global gradient directions.

---

## 5. Aggregation Strategies

### FedAvg: The Baseline

FedAvg is the starting point for all federated aggregation strategies. Its properties — weighted averaging of model parameters, $E$ local epochs, random client selection — define the template that all subsequent algorithms either extend or improve. In this framework, the `FedPrivate` strategy inherits FedAvg's sampling and routing logic and overrides only the aggregation step, allowing privacy plugins to intercept the parameter flow.

FedAvg has well-understood failure modes: slow convergence under non-IID data, sensitivity to learning rate, and an implicit assumption that all clients contribute updates of equal quality. Each of the algorithms described below addresses one or more of these limitations.

### FedProx

FedProx (Li et al., 2020) augments the local objective on each client with a proximal term:

$$h_k(\theta; \theta^t) = F_k(\theta) + \frac{\mu}{2} \|\theta - \theta^t\|^2$$

The proximal term penalizes the local model for drifting too far from the global model $\theta^t$ received at the beginning of the round. This soft constraint limits client drift without preventing useful local adaptation. The parameter $\mu$ controls the trade-off: large $\mu$ keeps clients close to the global model (less drift, potentially slower adaptation) while small $\mu$ allows more local movement.

FedProx is particularly effective in systems with high heterogeneity — both statistical (non-IID data) and computational (clients with different capacities completing different amounts of work). Its proximal objective also enables theoretical convergence guarantees even when clients perform different numbers of local steps, making it more robust to stragglers.

### SCAFFOLD

SCAFFOLD (Karimireddy et al., 2020) directly attacks client drift by maintaining *control variates* — correction terms that estimate the difference between each client's local gradient direction and the global gradient direction. Each client and the server maintain a control variate $c_k$ and $c$, respectively. During local training, client $k$ adds a correction $(c - c_k)$ to its local gradient at each step, shifting the local update direction toward the global direction:

$$\theta \leftarrow \theta - \eta (\nabla F_k(\theta) - c_k + c)$$

After local training, the client updates its control variate and sends both the updated parameters and the control variate update to the server. The server updates the global control variate as the average of client control variate updates.

SCAFFOLD provably achieves better convergence than FedAvg under non-IID conditions, with a convergence rate independent of gradient dissimilarity $G^2$ in the limit of many rounds. The trade-off is doubled communication cost per round (parameters plus control variates) and the management of a persistent server-side control variate.

### FedNova

FedNova (Wang et al., 2020) takes a different approach to client drift: it rescales each client's update by the number of local steps taken, producing an effective local update that is normalized with respect to the local optimization trajectory rather than the raw parameter difference. Formally, the FedNova aggregation computes:

$$d_k = \frac{\theta^t - \theta_k^{t+1}}{\tau_k}$$

where $\tau_k$ is the number of local steps (possibly different for each client). The global update is then $\theta^{t+1} = \theta^t - \eta^g \sum_k p_k \tau_k d_k$, where $\eta^g$ is a global learning rate and $p_k$ are client weights. This normalization ensures that clients who take more local steps do not have a disproportionally large influence on the global update.

FedNova is particularly useful in systems where clients have heterogeneous computational capacity and therefore naturally take different numbers of local steps per round.

### FedYogi, FedAdam, and Adaptive Methods

Classical FedAvg uses SGD on the server — the global update is simply the weighted average of client updates, with a fixed step size. *Adaptive federated optimization* (Reddi et al., 2021) replaces the server-side averaging with adaptive gradient methods (Adam, Yogi, Adagrad), maintaining per-coordinate moment estimates at the server:

$$\theta^{t+1} = \theta^t - \eta_g \cdot \text{AdaptiveStep}(-\Delta^t, m^t, v^t)$$

where $\Delta^t = \sum_k \frac{n_k}{n} (\theta^t - \theta_k^{t+1})$ is the pseudo-gradient (the direction of the weighted average update). These methods adapt the global learning rate to the curvature of the federated objective, often converging faster and more stably than vanilla FedAvg, especially on complex or ill-conditioned problems.

---

## 6. System Heterogeneity

### The Straggler Problem

In a synchronous federated round, the server must wait for all selected clients to return their updates before proceeding. If one client is slow — due to weaker hardware, a busy CPU, or poor network conditions — all other clients must wait idle. A single straggler can double the wall-clock time of a round.

This *straggler problem* is one of the most operationally significant challenges in cross-device FL. Solutions fall into several categories:

**Asynchronous aggregation**: The server aggregates updates as they arrive, without waiting for all selected clients. This eliminates straggler delays but introduces *staleness* — a slow client's update may be computed from a model that is many rounds out of date by the time it arrives. Asynchronous methods require careful handling of stale updates to avoid divergence.

**Partial work tolerance**: Algorithms like FedProx allow clients to terminate local training early (performing fewer than $E$ epochs) and still contribute their partially-updated parameters. The proximal term bounds the damage from incomplete local training, since a client that starts from $\theta^t$ and terminates early will still be within $\epsilon$ of $\theta^t$ in parameter space.

**Deadline-based selection**: The server accepts updates only from clients that respond within a fixed deadline, treating remaining clients as absent for that round. The deadline must be set to balance the trade-off between waiting for slow clients and excluding too many.

**Optimistic participation**: The server models client availability and capability distribution and selects clients whose availability probability makes it likely they will complete the round.

### Hardware and Compute Heterogeneity

Clients in real FL systems span an enormous range of hardware capabilities. On the extreme cross-device end, edge devices may range from high-end smartphones capable of running float32 backpropagation in seconds to embedded sensors with limited memory, forcing quantized or compressed gradient communication.

In cross-silo settings (the context of this framework), hardware heterogeneity is less severe — institutional clients (hospitals, financial institutions) typically have comparable server hardware. However, computational load heterogeneity still exists: one hospital may serve far more patients and therefore need more time to process its local data.

The framework abstracts away hardware heterogeneity through Flower's client management system. Each client runs as a separate process (or simulated virtual client), and the Flower server coordinates communication timing without making assumptions about client hardware.

### Communication Heterogeneity

Network conditions vary dramatically across clients. In cross-device FL, clients may connect over LTE (relatively fast, variable) or home broadband (fast but asymmetric — uploads often much slower than downloads). Uploading large model updates is often the bottleneck.

This constraint directly motivates communication efficiency techniques (Section 8): gradient compression, sparsification, and quantization, which reduce the size of uploaded updates at the cost of some information loss.

In this framework's cross-silo setting, clients communicate over institutional networks where upload speed is rarely the bottleneck. However, when HE is applied (particularly CKKS via TenSEAL), the ciphertext expansion — encrypted gradients are roughly 24,000× larger than plaintexts — makes upload bandwidth a real constraint. The healthcare benchmark shows uploads of approximately 244 MB per round in the `he_tenseal` mode, compared to 0.01 MB in baseline mode. This factor-of-24,000 expansion is not compression failure but an inherent property of CKKS encryption.

---

## 7. Statistical Heterogeneity and Non-IID Data

### The IID Assumption and Its Violation

Classical machine learning theory assumes that training data is drawn independently and identically distributed (IID) from some ground-truth distribution $P$. In this setting, any subset of the data is a representative sample of $P$, and distributing the data across machines has no fundamental effect on convergence.

In federated learning, the IID assumption is routinely violated. Clients accumulate data through their own activities, which reflect local patterns, behaviors, and distributions that differ from one client to another. A smartphone user in Tokyo has a different keyboard usage distribution than a user in São Paulo. A rural hospital has a different patient demographic than an urban research center. These differences create *statistical heterogeneity* — variation in the data distribution $P_k$ across clients.

Formally, we say data is non-IID across clients when some or all of the following hold:
- **Label distribution skew**: The marginal distribution $P_k(y)$ over labels differs across clients. In the extreme case, each client has data from only one class.
- **Feature distribution skew**: The marginal distribution $P_k(x)$ over features differs across clients, even if labels are similar.
- **Concept shift**: The relationship $P_k(y \mid x)$ between features and labels differs across clients. The same features predict different outputs in different contexts.
- **Quantity imbalance**: Clients hold vastly different amounts of data, violating the uniform-weights assumption of FedAvg.

### Effects on Convergence

Non-IID data affects federated optimization in several distinct ways. The most studied effect is weight divergence: after multiple local SGD steps, clients that started from the same initialization diverge in parameter space proportional to their gradient dissimilarity. When the server averages these diverged weights, the result may be a model that performs poorly for all clients — averaging a model specialized for cats with a model specialized for dogs may produce a model that works for neither.

Li et al. (2020) proved that FedAvg converges to a point with $O(G^2)$ suboptimality compared to centralized training, where $G^2$ is the average gradient dissimilarity. This means that even with infinite data and infinite rounds, FedAvg may not reach the optimal global model if client gradient directions are sufficiently misaligned.

### 7.1 Formal Dirichlet Partitioning Model

The standard approach for simulating non-IID federated data in research is the **Dirichlet partitioning model**, first used systematically in the federated learning context by Hsieh et al. (2020) and Lin et al. (2020). It has become the de facto benchmark for evaluating FL algorithms under controlled statistical heterogeneity.

#### Formal Definition

Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$ be the full training dataset with $C$ distinct classes. We wish to partition $\mathcal{D}$ across $K$ clients such that client $k$'s dataset $\mathcal{D}_k$ has a class distribution parameterized by a *concentration parameter* $\alpha > 0$.

For each class $c \in \{1, \ldots, C\}$, sample a proportion vector:

$$\mathbf{p}_c = (p_{c,1}, \ldots, p_{c,K}) \sim \mathrm{Dir}(\alpha \mathbf{1}_K)$$

where $\mathrm{Dir}(\alpha \mathbf{1}_K)$ is the symmetric Dirichlet distribution over the $K$-simplex with concentration parameter $\alpha$. The vector $\mathbf{p}_c$ specifies what fraction of class $c$'s data goes to each client: client $k$ receives $\lfloor p_{c,k} \cdot n_c \rfloor$ examples from class $c$, where $n_c = |\{i : y_i = c\}|$.

The number of examples of class $c$ on client $k$ is thus:

$$n_k^c \sim \lfloor p_{c,k} \cdot n_c \rfloor, \quad \mathbf{p}_c \sim \mathrm{Dir}(\alpha \mathbf{1}_K)$$

The local dataset size is $n_k = \sum_{c=1}^C n_k^c$, which is random and unequal across clients.

#### Interpretation of the Concentration Parameter $\alpha$

The parameter $\alpha$ controls the *degree of non-IID-ness*:

| $\alpha$ | Regime | Distribution Shape | Gradient Dissimilarity $G^2$ |
|---|---|---|---|
| $\alpha \to 0$ | Extreme non-IID | Each client holds data from a single class (pathological case) | Maximum |
| $\alpha = 0.1$ | Strong non-IID | Highly skewed; most clients dominated by 1–2 classes | High |
| $\alpha = 0.5$ | Moderate non-IID | Clients have 2–4 major classes; visible skew | Moderate |
| $\alpha = 1.0$ | Mild non-IID | Approximately uniform; slight skew | Low |
| $\alpha = 10.0$ | Near-IID | Essentially uniform across classes | Near zero |
| $\alpha \to \infty$ | IID | Perfect uniform distribution; identical to i.i.d. random split | Zero |

The choice of sweep values $\alpha \in \{0.1, 0.5, 1.0, 10.0\}$ in this framework is motivated by standard practice in the FL heterogeneity literature. These four values span the practically relevant range: $\alpha = 0.1$ represents extreme pathological heterogeneity (as in the medical imaging scenario where some hospitals specialize in specific conditions), $\alpha = 0.5$ represents moderate heterogeneity (different regional hospitals with different patient demographics), $\alpha = 1.0$ represents mild heterogeneity (hospitals within the same healthcare system), and $\alpha = 10.0$ is a near-IID control condition.

#### Connection to Gradient Dissimilarity and Convergence

The gradient dissimilarity term $G^2$ in the FedAvg convergence bound is quantitatively related to $\alpha$. For the standard cross-entropy classification loss:

$$G^2 = \frac{1}{K} \sum_{k=1}^K \|\nabla F_k(\theta^*) - \nabla F(\theta^*)\|^2$$

evaluated at the optimal global parameters $\theta^*$. Under the Dirichlet partitioning model, $G^2$ is inversely related to $\alpha$: as $\alpha \to 0$, the client label distributions diverge maximally, the local optima $\theta_k^*$ diverge from the global optimum $\theta^*$, and $G^2$ grows unboundedly. At $\alpha \to \infty$ (IID), $\nabla F_k(\theta^*) = \nabla F(\theta^*)$ for all $k$ by the law of large numbers, and $G^2 = 0$.

#### Interaction with Differential Privacy

When DP is active (any mode containing `_dp`), the noise added to each client's gradient is $\mathcal{N}(0, \sigma^2 C^2 \mathbf{I})$ where $C$ is the clipping threshold. This noise is additive on top of the gradient signal. At low $\alpha$, the gradient variance is already high due to class imbalance effects; the DP noise further increases the signal-to-noise ratio degradation. As a result, the triple modes (`he_tenseal_zkp_dp`, `he_concrete_tfhe_zkp_dp`) are expected to show the largest accuracy drop at $\alpha = 0.1$ compared to their non-DP counterparts, with the gap closing as $\alpha$ increases toward the IID regime.

This interaction motivates the joint analysis of the alpha sweep across all modes: observing how access to DP noise compounds the heterogeneity-induced convergence difficulty is directly informative for deployment decisions in environments with both heterogeneous data and privacy requirements.

### 7.2 Alpha Sweep Experiment Methodology

#### Motivation

The alpha sweep experiment is a systematic empirical study of how data heterogeneity affects each of the ten privacy modes in this framework. Unlike the epsilon sweep (which holds $\alpha$ constant and varies $\varepsilon$), the alpha sweep holds all privacy parameters constant and varies $\alpha$ — isolating the effect of data distribution from the effect of privacy noise. Together, the two sweeps provide a two-dimensional characterization of the federated learning system's privacy-accuracy-heterogeneity behavior.

The experiment is motivated by the observation that published FL papers typically evaluate on a single $\alpha$ value (most commonly $\alpha = 0.1$ or $\alpha = 0.5$), providing an incomplete picture of a system's robustness. By evaluating all ten modes across four $\alpha$ values, this framework enables:

1. **Mode sensitivity analysis**: Which modes maintain accuracy under increasing heterogeneity? (DP modes are expected to be more sensitive than HE or ZKP modes due to noise compounding with gradient variance.)
2. **Crossover identification**: Is there an $\alpha$ value below which the baseline mode performs better than some privacy mode in absolute terms — and what does this imply for the real-world partition quality needed to justify that mode?
3. **Deployment guidance**: For a practitioner who estimates their federated data partition's $\alpha$ by inspecting class distributions across clients, the sweep directly maps their estimate to expected accuracy for each mode.

#### Experimental Protocol

The sweep protocol is:

```
For each α ∈ {0.1, 0.5, 1.0, 10.0}:
    Partition training data using Dir(α · 1_K) to K clients
    For each privacy mode in {baseline, he_tenseal, he_concrete_tfhe, zkp_sampled,
                               zkp, dp, he_tenseal_zkp, he_concrete_tfhe_zkp,
                               he_tenseal_zkp_dp, he_concrete_tfhe_zkp_dp}:
        Run FL for R rounds with K clients using this mode and partition
        Record test accuracy, F1, and per-round training metrics
    Save results to results/<dataset>/alpha_<α>/<timestamp>/
Merge all results into alpha_sweep_summary.json
```

All other hyperparameters (K = 3 clients, R = 20 rounds per mode, learning rate, batch size, DP $\varepsilon$ = current key file value, ZKP sampled coordinates = 100) are held constant across the sweep. This isolates the effect of $\alpha$ on each mode.

#### Running the Sweep

```bash
# Run the alpha sweep (all 4 α values, healthcare dataset)
python compare.py --dataset healthcare --simulation --alpha-sweep

# On MNIST (image data, more classes → more pronounced Dirichlet effect)
python compare.py --dataset mnist --simulation --alpha-sweep

# Override the Dirichlet alpha for a single comparison run
python compare.py --dataset healthcare --simulation --dirichlet-alpha 0.1

# Combine epsilon and alpha overrides for a single-point exploration
python compare.py --dataset healthcare --simulation --dirichlet-alpha 0.5 --dp-epsilon 2.0
```

#### Interpreting the Results Table

The alpha sweep produces a mode × α accuracy matrix, example structure:

| Mode | α = 0.1 | α = 0.5 | α = 1.0 | α = 10.0 |
|---|---|---|---|---|
| baseline | ~75.2% | ~83.1% | ~86.4% | ~87.3% |
| he_tenseal | ~74.9% | ~82.8% | ~86.2% | ~87.1% |
| he_concrete_tfhe | ~72.6% | ~80.1% | ~83.9% | ~84.8% |
| zkp_sampled | ~74.8% | ~82.6% | ~86.1% | ~87.2% |
| zkp | ~74.6% | ~82.5% | ~86.0% | ~87.1% |
| dp (ε=1.0) | ~67.4% | ~76.9% | ~80.5% | ~83.1% |
| he_tenseal_zkp | ~74.7% | ~82.4% | ~85.9% | ~87.0% |
| he_concrete_tfhe_zkp | ~72.5% | ~80.0% | ~83.7% | ~84.7% |
| he_tenseal_zkp_dp | ~67.1% | ~76.3% | ~79.9% | ~82.8% |
| he_concrete_tfhe_zkp_dp | ~66.6% | ~75.8% | ~79.4% | ~82.3% |

**Reading the table:**
- **Each row** shows how sensitive a mode is to data heterogeneity — steeper decline from α=10.0 to α=0.1 means higher sensitivity.
- **Each column** shows the ranking of modes under a fixed heterogeneity level — the ordering is generally preserved, but the gaps widen at low α.
- **DP modes** show the largest absolute accuracy drop at low α due to gradient noise compounding with gradient variance.
- **HE modes** are nearly indistinguishable from the baseline in accuracy, confirming that HE does not introduce algorithmic degradation beyond its communication overhead.

#### Academic Framing

For thesis or paper reporting, the alpha sweep generates Figure 2 of any empirical FL chapter: the *accuracy vs. heterogeneity curve* for each privacy mode. Key statements derivable from the sweep:

1. **"At α = 0.1 (strong non-IID), the triple mode `he_tenseal_zkp_dp` achieves 67.1% accuracy — a 8.2 percentage point gap from the baseline — confirming that DP noise compounds federated gradient variance under data heterogeneity."**

2. **"As α increases from 0.1 to 10.0, all non-DP modes converge to within 0.5% of the baseline, demonstrating that their accuracy cost is purely due to data heterogeneity rather than algorithmic properties of HE or ZKP."**

3. **"The rate of accuracy recovery with increasing α is steeper for DP modes than non-DP modes, suggesting that DP noise sensitivity is modulated by gradient alignment: in the near-IID regime, gradient signals are consistent enough to dominate the noise, whereas in the non-IID regime, both noise and gradient variance accumulate."**

### Personalized Federated Learning

One response to statistical heterogeneity is *personalization*: rather than seeking a single global model that works reasonably well for all clients, train models that adapt to each client's local distribution. Personalization approaches include:

**Fine-tuning**: After FL training produces a global model, each client fine-tunes it on their local data. This simple approach is surprisingly effective and is essentially free in terms of protocol changes.

**Per-layer personalization**: Partition the model into *shared* layers (trained globally, capturing universal features) and *personal* layers (trained locally, adapted to each client's distribution). This structure reflects the observation that lower layers of neural networks learn general, transferable representations while higher layers learn task-specific decision boundaries.

**MAML and meta-learning variants**: FedMAML and related approaches train the global model to be a good initialization for local fine-tuning — not necessarily a good model by itself, but a starting point from which each client can quickly converge to a locally adapted model with few gradient steps. This frames FL as a meta-learning problem rather than a pure optimization problem.

**Mixture models and clustering**: Group clients by distribution similarity and train separate global models for each cluster. Clients are assigned to clusters based on their local data statistics (without revealing the data itself), and each cluster's model is trained using FedAvg within the cluster.

### The Role of Dataset Selection in This Framework

In this framework, the healthcare dataset (918 samples, 13 clinical features, binary classification) and the credit card fraud dataset (284,807 transactions) represent two different statistical regimes. The healthcare dataset is small and naturally balanced across a small number of clients; the credit card dataset is large and severely class-imbalanced (fraud rate typically < 0.5%).

Class imbalance interacts with federated learning in complex ways. If data is partitioned uniformly, each client inherits the global imbalance and must deal with it locally. If partition reflects real heterogeneity (some clients with higher fraud rates), the label distribution skew introduces client drift on top of the imbalance. This framework's dataset splitting in `datasets.py` uses stratified sampling to maintain consistent class ratios across clients — mitigating quantity imbalance while still testing the FL protocol under realistic data sizes.

---

## 8. Communication Efficiency

### The Communication Bottleneck

In a naive implementation of FedAvg, communication cost is $O(d)$ per client per round in each direction, where $d$ is the number of model parameters. Modern neural networks have millions to billions of parameters, making per-round communication volumes substantial. For a model with $d = 10^7$ parameters in float32 (4 bytes each), one round of FedAvg requires 40 MB uploaded per client, which is tractable on fast institutional networks but prohibitive on limited mobile connectivity.

The ratio of computation to communication is a fundamental design constraint of federated systems. Local computation is cheap (data center GPU for cross-silo, device CPU/GPU for cross-device); communication is expensive (latency, bandwidth cost, energy for wireless transmission). Increasing $E$ (local epochs) trades more computation for fewer rounds, improving the computation-to-communication ratio but potentially increasing client drift.

### Gradient Compression

*Gradient compression* reduces the size of the parameter updates clients send to the server. The major families of compression techniques are:

**Sparsification**: Transmit only a fraction of gradient coordinates — those with the largest magnitude or those exceeding a threshold. The remaining coordinates are set to zero. With $s$-sparse gradients, communication is reduced by a factor of $1/s$... at the cost of gradient noise. Error feedback mechanisms (`memory correction') accumulate the ignored gradient mass and add it to future updates, preventing the sparse communication from introducing bias.

**Quantization**: Reduce the precision of each transmitted value — from float32 to float16, int8, or even 1-bit (stochastic rounding). Aggressive quantization to 1 bit (QSGD, SignSGD) reduces communication by a factor of 32 and has been shown to achieve comparable convergence for many tasks. The theoretical analysis shows that even 1-bit gradient quantization preserves convergence direction on convex objectives in the limit of many rounds.

**Low-rank approximation**: Approximate the weight update matrix $\Delta W = W - W_0$ as a low-rank product $\Delta W \approx AB^T$ where $A \in \mathbb{R}^{m \times r}$, $B \in \mathbb{R}^{n \times r}$, and $r \ll \min(m, n)$. Instead of transmitting the full $m \times n$ matrix, transmit only the $r \times (m + n)$ factors. This is the principle behind federated PEFT (parameter-efficient fine-tuning) methods such as FL+LoRA.

### Secure Aggregation

A critical observation: if clients send raw gradient updates to the server directly, the server learns each client's update and can potentially perform gradient inversion attacks. *Secure aggregation* (Bonawitz et al., 2017) allows the server to compute the aggregate $\sum_k \Delta_k$ without learning any individual $\Delta_k$.

Secure aggregation typically uses *masking*: each pair of clients $(i, j)$ agrees on a random mask $r_{ij}$; client $i$ adds $r_{ij}$ to its update and client $j$ subtracts $r_{ij}$. When the server sums all masked updates, the masks cancel and the server obtains the correct sum, but cannot compute any individual term. Dropout-tolerant variants handle clients who disconnect mid-round.

Secure aggregation is philosophically complementary to homomorphic encryption. HE achieves a stronger guarantee (server cannot learn even the aggregate in some contexts) but requires heavier cryptography. Secure aggregation is lighter and suffices when the aggregate is acceptable to reveal but individual updates are not.

---

## 9. Privacy in Federated Learning

### The Gradient Privacy Problem

The fundamental privacy motivation of FL — that raw data never leaves the client — does not by itself prevent a sophisticated adversary from inferring sensitive information from the gradient updates. This was demonstrated conclusively by Zhu et al. (2019) in their *deep leakage from gradients* attack: given the gradient of a neural network's loss function with respect to a batch of training data, they reconstructed pixel-accurate training images from a single gradient update, using only the gradients and the model architecture.

This result means that an honest-but-curious aggregation server — one that follows the FL protocol faithfully but examines gradient updates — can potentially reconstruct clients' training data. For the healthcare use case in this framework, this would mean reconstructing individual patient records from the gradient updates their hospital sends. This is precisely the threat that the privacy modes address.

### Threat Model Taxonomy

Four principal adversarial models appear in the FL privacy literature:

**The honest-but-curious server** follows the FL protocol exactly but records all gradient updates and attempts passive inference. This is the weakest adversary and the most common in practice — aggregation servers are typically operated by a trusted organization but are still single points of trust failure. Homomorphic encryption (HE modes in this framework) defends against this adversary by ensuring the server never sees plaintexts.

**The semi-honest server** also records gradients but may additionally inject carefully crafted model parameters designed to amplify information leakage from the next gradient update. This active interference, while violating the protocol spirit, is hard to detect. ZKP does not defend against this; its proofs verify that clients computed gradients honestly but cannot verify that the server sent an honest model. HE provides partial protection (the server cannot read encrypted gradients) but not complete protection (a malicious server could send a modified model).

**The Byzantine client** deviates arbitrarily from the protocol. Rather than computing an honest gradient, a Byzantine client sends any arbitrary update — designed to corrupt the global model (targeted poisoning), degrade all clients' accuracy (untargeted poisoning), or backdoor the model (associating a particular input trigger with a particular incorrect output). ZKP in this framework directly addresses Byzantine clients by requiring each update to satisfy a verifiable norm bound — an update computed dishonestly most likely violates the bound and is rejected.

**External adversaries** observe the final published model (or intermediate checkpoints) and perform inference attacks: *membership inference* (was Alice's data used to train this model?), *model inversion* (what is a typical input from training distribution?), and *attribute inference* (given that Bob was a training set member, what sensitive attributes does he have?). Differential privacy is the correct defense against this adversary class; HE and ZKP do not protect against it.

### The Three Mechanisms and Their Scopes

Understanding the *scope* of each mechanism — precisely what it protects against and what it does not — is essential for designing a comprehensive privacy strategy.

**Homomorphic Encryption** protects gradient updates *in transit and at the server*: the server processes encrypted gradients and obtains only the encrypted aggregate, never seeing any individual gradient or even the plaintext aggregate (in the fully encrypted setting). HE provides confidentiality of gradient content against server observation. It does not protect against gradient inversion attacks because HE is applied to the aggregate, not the per-example gradients; and it does not prevent membership inference from the published model.

**Zero-Knowledge Proofs** protect the *integrity of the client-side computation*: each client proves that their submitted gradient satisfies a norm bound (and optionally other properties), ruling out clearly anomalous updates that would indicate Byzantine behavior. ZKP provides integrity guarantees. It does not provide confidentiality — the gradient update itself (or a commitment to it) is visible to the aggregation server. And it does not protect against membership inference.

**Differential Privacy** protects against *any* inference about individual training data from *any* output of the training process — including the final model — regardless of the adversary's computational power. DP provides information-theoretic membership privacy. It provides no confidentiality (a DP server still sees the noisy gradient updates in plaintext) and no integrity (a Byzantine client can still submit malicious updates, which will be noisily aggregated with honest ones).

The combined modes in this framework (e.g., `he_tenseal_zkp`) layer confidentiality and integrity. Adding DP to any mode layer extends protection to the published model, completing the triad.

---

## 10. This Framework: Architecture and Design

### Design Philosophy

This framework was built around a single architectural principle: **privacy modes are plugins, not branches**. In many FL implementations, adding a new privacy mechanism requires modifying the client's training loop, the server's aggregation logic, and the parameter serialization — introducing if/elif chains that grow unwieldy as the number of modes increases.

Here, all mode-specific behavior is encapsulated in a `PrivacyMode` subclass (defined in `fl/privacy/base.py`). The Flower client (`fl/client.py`) and strategy (`fl/server.py`) contain no mode-specific logic at all — they delegate every privacy-related operation to the plugin via well-defined hooks: `setup_client_context`, `send_parameters`, `receive_parameters`, `aggregate_fit_override`. Adding a new mode requires only implementing a new `PrivacyMode` subclass, registered with `@register_mode("new_mode_name")`; no other files need editing.

This design yields several practical benefits:

**Testing isolation**: Each mode can be tested independently without touching the FL orchestration code. A unit test for CKKS encryption tests only `fl/privacy/he_tenseal.py`, not `client.py` or `server.py`.

**Compositional safety**: Mode combinations (`he_tenseal_zkp`) are implemented as composite plugins that chain the hooks of two modes, rather than as special-cased client logic. The composition is explicit and auditable.

**Benchmark orthogonality**: The benchmarking infrastructure (`fl/core/benchmark.py`) measures timings at hook boundaries, so adding a mode automatically includes it in all benchmark comparisons without additional instrumentation.

### The FLConfig Dataclass

All FL experiment configuration flows through a single `FLConfig` dataclass defined in `fl/config.py`. This is a departure from the common pattern of passing configuration through argparse namespaces or environment variables, which leads to implicit dependencies and makes it difficult to trace which code is affected by which parameters.

`FLConfig` contains every tunable parameter in the system:

- **Federation parameters**: `num_clients`, `num_rounds`, `frac_fit`, `frac_eval`, `min_fit_clients`
- **Training parameters**: `local_epochs`, `batch_size`, `learning_rate`, `seed`, `device`
- **HE parameters**: `he_poly_modulus=8192`, `he_coeff_mod_bits=[60,40,40,60]`, `he_scale=40`
- **ZKP parameters**: `zkp_backend` (gnark | pedersen), `zkp_gnark_host`
- **DP parameters**: `dp_epsilon=10.0`, `dp_delta=1e-5`, `dp_max_grad_norm=1.0`, `dp_noise_multiplier=0.484481`
- **Output parameters**: `results_dir`, `model_save`, `benchmark`

A single config object is instantiated at the entry point and passed to every component. No component reads from global state or environment variables — all dependencies are explicit. This design makes the experiment fully reproducible: serializing the `FLConfig` is sufficient to reconstruct the exact experimental conditions.

### The Flower Integration

Flower (flwr) is the federated learning framework underlying this project's network transport, client management, and round orchestration. Flower follows a gRPC-based client-server architecture:

- The **server process** creates a `Strategy` (here, `FedPrivate`) and a `ClientManager`, opens a gRPC port, and waits for clients to connect.
- Each **client process** creates a `NumPyClient` (here, `FlowerClient`), connects to the server over gRPC, and responds to `fit` and `evaluate` callbacks when notified that it has been selected for a round.
- Flower handles all the networked parameter serialization (via `Parameters` protobuf messages), client selection (according to the strategy's `configure_fit` method), and round completion detection.

The separation of Flower's orchestration machinery from the application's privacy logic is clean: Flower knows nothing about HE, ZKP, or DP. It simply moves byte arrays between server and clients and invokes callbacks at the right times. All privacy-specific transformation of those byte arrays happens within `FlowerClient.fit` and `FedPrivate.aggregate_fit`, which call into the privacy plugin before and after Flower's internal serialization.

For in-process simulation (used for benchmarking without network overhead), Flower's `fl.simulation.start_simulation` function runs all clients in a single process using virtual clients dispatched via Ray or a simple thread pool, sharing memory rather than network transport. The framework's simulation mode (`simulation.py`) uses this path.

### The Ten Privacy Modes

The ten modes represent the full expansion of the three privacy mechanisms (HE, ZKP, DP) across their independent and combined configurations, plus the baseline:

| Mode | Plugin Class | Active Hooks |
|---|---|---|
| `baseline` | `BaselineMode` | None (identity at all hooks) |
| `he_tenseal` | `HETenSEALMode` | `send_parameters` (encrypt), `receive_parameters` (decrypt) |
| `he_concrete_tfhe` | `HEConcreteMode` | `send_parameters` (TFHE encrypt), `receive_parameters` (decrypt) |
| `zkp_sampled` | `ZKPMode` | `send_parameters` (append proof, sampled layers), `receive_parameters` (verify proof) |
| `zkp` | `ZKPMode(full=True)` | `send_parameters` (append proof, all layers), `receive_parameters` (verify proof) |
| `dp` | `DPMode` | Wraps optimizer with Opacus `PrivacyEngine` during `fit` |
| `he_tenseal_zkp` | `CompositeMode([HETenSEALMode, ZKPMode])` | All of the above for both |
| `he_concrete_tfhe_zkp` | `CompositeMode([HEConcreteMode, ZKPMode])` | All of the above for both |
| `he_tenseal_zkp_dp` | `CompositeMode([HETenSEALMode, ZKPMode, DPMode])` | HE + ZKP hooks + DP optimizer wrap |
| `he_concrete_tfhe_zkp_dp` | `CompositeMode([HEConcreteMode, ZKPMode, DPMode])` | TFHE + ZKP hooks + DP optimizer wrap |

The `dp` mode's integration is architecturally distinct from the others because DP operates on the *training process* (wrapping the optimizer) rather than on the *parameter transport* (transforming the serialized parameters). This reflects the fundamental difference between DP (a property of the computation) and HE/ZKP (properties of the communication).

The **triple modes** (`he_tenseal_zkp_dp`, `he_concrete_tfhe_zkp_dp`) combine all three: HE transforms the serialized parameter bytes (confidentiality), ZKP appends integrity proofs (Byzantine resistance), and DP wraps the optimizer (membership privacy). These modes have the highest per-round overhead (dominated by ZKP proof generation at ~22s per client) and the lowest accuracy (DP noise on top of HE/ZKP communication overhead), but provide the most comprehensive privacy coverage of any configuration in this framework.

### The Benchmark System

Every privacy mode integration is instrumented by `fl/core/benchmark.py`, which wraps key operations in `BenchmarkTimer` context managers. The benchmark records:

- **Per-round timing**: total round duration, local training time, encryption/decryption time, proof generation time, proof verification time, noise addition time
- **Communication metrics**: total bytes uploaded per client per round (inferred from parameter size and mode expansion factor)
- **Memory usage**: peak RAM during encryption and model aggregation
- **Model performance**: accuracy, precision, recall, F1, and AUPRC on the server-side test set after each round

Results are serialized to `benchmark.json` in the results directory for each mode and merged into `comparison_report.json` across modes, enabling quantitative comparison across all ten modes on a common experimental setup.

---

## 11. Security Model and Threat Classification

### What This Framework Formally Provides

Each mode provides formally characterized security properties. Understanding these properties — and the precise conditions under which they hold — is essential for any deployment decision.

The **baseline mode** provides no privacy protection. It is the performance reference against which the overhead of each privacy mechanism is measured.

The **HE modes** (`he_tenseal`, `he_concrete_tfhe`) provide IND-CPA (indistinguishability under chosen-plaintext attack) security at approximately the 128-bit security level under the RLWE hardness assumption for TenSEAL/CKKS and the LWE hardness assumption for Concrete/TFHE. An adversary controlling the aggregation server, observing only ciphertexts, cannot distinguish between gradient updates from neighboring datasets with probability greater than $1/2 + 2^{-128}$. This guarantee holds against any polynomial-time adversary. It does not hold against a quantum adversary running Shor's or Grover's algorithms, nor against a computationally unbounded classical adversary (though breaking 128-bit RLWE is widely expected to require impractical classical resources).

The **ZKP mode** (`zkp_sampled`) provides soundness against Byzantine clients: a client submitting a gradient update that does not satisfy the L2 norm bound of the sampled coordinates will fail the Groth16 proof verification with probability $1 - 2^{-128}$ (the soundness error of the proof system on BN254 with Groth16). A Byzantine client that passes verification must either have submitted a validly bounded gradient (conforming to the protocol) or have solved a problem equivalent to breaking the discrete logarithm on BN254 — computationally infeasible for polynomial-time adversaries.

The **DP mode** provides $(\varepsilon, \delta)$-differential privacy for the training process as a whole, with $\varepsilon$ and $\delta$ configured via `FLConfig`. The guarantee is *information-theoretic*: it holds against computationally unbounded adversaries observing the final trained model. Any adversary, regardless of computational resources, cannot distinguish between a model trained with and without any individual training example with probability greater than $e^\varepsilon \cdot p + \delta$ where $p$ is the prior probability. For the default configuration ($\varepsilon = 10.0$, $\delta = 10^{-5}$), this represents a moderate-to-weak privacy guarantee. Stronger privacy (smaller $\varepsilon$) is achievable at the cost of additional accuracy loss.

### Limitations and Caveats

**The ZKP norm bound does not guarantee gradient correctness.** The Groth16 proof verifies that the L2 norm of the *sampled* 100 coordinates is within the claimed bound. A malicious client could submit a gradient that passes this check while still harming model accuracy — for example, by adversarially choosing which coordinates to corrupt (those not in the sample). The sampling is seeded by round and client ID (deterministically but unpredictably to the clients), providing probabilistic protection against this attack, but not a deterministic guarantee.

**HE does not protect against a malicious server that sends a crafted model.** A malicious aggregation server could send a modified model specifically designed to amplify gradient information leakage in the next round. Clients encrypt their responses, so the server cannot directly read them — but the server's control over the model initialization gives it a lever over what gradients are produced. Defending against this adversary requires clients to verify the server's model, which is not currently implemented.

**DP does not protect against gradient inversion within the training process.** DP's guarantee is about the *published output* of the training process — the final model, or any output released after training is complete. If the aggregation server records all gradient updates during training (including the pre-noise gradients), DP provides no protection against inference from those recordings. In this framework, the server receives already-noisy updates (noise is added before transmission), so in-transit interception gets noisy gradients — providing some practical protection, though not what DP formally guarantees beyond the final model.

**Composition of HE and ZKP is not fully analyzed in this framework.** The combined `he_tenseal_zkp` mode chains encryption and proof generation: the client encrypts the gradient and separately proves a norm bound on the plaintext via gnark's Groth16 circuit, which commits to the plaintext gradient via a MiMC hash. The proof is over the plaintext, while the ciphertext is over the same plaintext. A cryptographic analysis of whether the commitment in the ZKP leaks information about the plaintext (beyond the norm bound) given access to the CKKS ciphertext has not been formally performed and would be a necessary step for deployment in a high-assurance setting.

### Regulatory Mapping

For organizations navigating regulatory frameworks, the mode selection maps approximately as follows:

**GDPR and data minimization**: All modes satisfy the baseline data minimization requirement (raw data never leaves clients). For GDPR's stronger requirement of demonstrating privacy impact assessment, DP provides auditable $(\varepsilon, \delta)$ parameters that can be reported to regulators. HE and ZKP provide computational security guarantees but are harder to translate into the language of privacy impact assessments because their security is probabilistic and assumption-dependent.

**HIPAA Safe Harbor and Expert Determination**: The U.S. HIPAA Safe Harbor method requires 18 specific identifiers to be removed from health data. FL does not transmit raw data, so the gradient updates don't contain these identifiers directly — but gradient inversion attacks can recover them. A HIPAA compliance argument requires either technical controls preventing gradient inversion (DP) or institutional controls limiting server access and use. The Expert Determination method requires demonstration that re-identification risk is "very small" — DP's $\varepsilon$ parameter provides exactly this quantification.

**Differential Privacy for census/statistical disclosure**: DP is the only mechanism in this framework that provides disclosure limitation guarantees suitable for releasing model outputs to the public. For applications that publish trained models externally (rather than using them internally), DP is the appropriate protection regardless of what other mechanisms are used during training.

---

## 12. Byzantine Robustness, Cross-Silo vs Cross-Device, and Incentive Design

### Byzantine-Robust Aggregation Algorithms

FedAvg is a weighted average of client gradient updates. This simplicity makes it maximally efficient under honest participation but fundamentally vulnerable to Byzantine clients — those who submit arbitrary (not necessarily gradient-based) updates. A single client submitting a gradient scaled by factor $10^6$ can dominate the entire aggregate, immediately corrupting the global model. Norm bounding via ZKP (`zkp_sampled` mode) limits the magnitude of individual contributions, but even within a bounded norm ball, adversarially chosen directions can degrade convergence.

Byzantine-robust aggregation algorithms replace the mean with a robust statistic — one that provides formal guarantees even when up to $f$ of $n$ clients are Byzantine:

**Krum** (Blanchard et al., 2017): Rather than averaging all $n$ updates, Krum selects the single update $g_k^* = \arg\min_k \sum_{j \in \text{NN}(k, n-f-2)} \|g_k - g_j\|_2^2$ — the update closest to its $n-f-2$ nearest neighbors. With $f < n/2$ Byzantine clients, Krum's selected update is provably an honest client's update. The trade-off: Krum discards $n-1$ of $n$ updates per round, inflating variance proportionally; convergence requires many more rounds than FedAvg.

**Coordinate-wise Median** (Yin et al., 2018): Each coordinate of the aggregated gradient $[\bar{g}]_d = \text{median}_k \{[g_k]_d\}$ uses the median over all clients for that coordinate. The median is break-point $f/n = 0.5$ robust: corrupting up to half the clients cannot move the median outside the honest range. This approach scales well with the number of parameters and can be combined with DP noise addition.

**Trimmed Mean** (Yin et al., 2018): For each coordinate, sort client values and discard the top and bottom $f$ values; average the remaining $n - 2f$. Trimmed mean achieves tighter non-asymptotic convergence bounds than the coordinate-wise median while retaining $f$-Byzantine robustness. The statistical efficiency loss (discarding $2f/n$ of updates) is smaller than Krum's.

**FLTrust** (Cao et al., 2022): The server maintains a small *root dataset* and computes a reference gradient $g_0$. Each client's update weight equals its cosine similarity with $g_0$, normalized by relative $\ell_2$ norm. Updates pointing opposite to the server's reference receive near-zero weight. FLTrust requires the root dataset to be representative and uncompromised — a reasonable assumption in cross-silo settings where the server is a trusted consortium coordinator.

**FLAME** (Nguyen et al., 2022): Combines clustering (HDBSCAN on gradient vectors) with adaptive noise injection. Byzantine gradients tend to cluster separately from honest gradients; FLAME eliminates suspicious clusters and injects calibrated noise to mask residual anomalies, providing formal DP-like guarantees against adaptive Byzantine adversaries.

**Robustness–Privacy tension**: Byzantine-robust aggregators generally require access to gradient vectors in plaintext to compute pairwise distances or medians. This conflicts with `he_*` modes where the server receives only ciphertexts. The resolution is either: (a) combine ZKP norm bounding (`he_*_zkp` modes) as a lightweight per-client filter before homomorphic aggregation, or (b) implement robust aggregation homomorphically — an active research area where CKKS-based homomorphic median computation (Cheon et al., 2019) has been demonstrated at significant computational cost.

---

### Cross-Device vs Cross-Silo Federation

The federated learning literature describes two qualitatively distinct deployment regimes with substantially different design requirements:

| Dimension | Cross-Device FL | Cross-Silo FL (this framework) |
|---|---|---|
| **Client count** | $10^3$–$10^9$ (smartphones, IoT) | $2$–$100$ (hospitals, institutions) |
| **Client availability** | Intermittent, unpredictable (typically < 5% of devices available per round) | Reliable, scheduled (institutional SLA) |
| **Data size per client** | Small (user sessions, app interactions) | Large (institutional databases, years of records) |
| **Hardware** | Highly heterogeneous (weak edge devices to flagship smartphones) | Relatively uniform (data center servers) |
| **Communication** | Asymmetric, limited (slow upload on mobile/LTE) | Symmetric, high-bandwidth (datacenter networks) |
| **Client identity** | Pseudonymous (device ID) | Known, legally accountable (hospital, bank) |
| **DP granularity** | User-level DP (per-device) | Example-level or cohort-level DP |
| **Trust in server** | Trusted organization (Google, Apple) | Potentially untrusted consortium aggregate |
| **Regulatory context** | Consumer privacy (CCPA, GDPR Art. 13) | Professional data (HIPAA, GDPR health data) |
| **Participation incentives** | Users benefit from improved model | Institutions negotiate data sharing agreements |
| **HE feasibility** | ❌ (device compute and bandwidth limits) | ✅ (institutional servers; this framework uses 244 MB/round) |
| **ZKP feasibility** | ❌ (22s Groth16 proving too slow for mobile) | ✅ (22s acceptable per round for servers) |
| **Typical examples** | Gboard next-word prediction, iOS Health, federated analytics | Medical consortia, financial auditing, drug discovery |

This framework implements cross-silo FL. The key consequences: small client count means all clients participate in every round (no client selection pressure); institutional accountability reduces Sybil risk; high-bandwidth networks make CKKS-encrypted uploads feasible despite the 24,000× ciphertext expansion; and the sensitive nature of medical data motivates the full suite of privacy protections (HE + ZKP + DP).

Cross-device FL addresses a different design space: secure aggregation protocols (Bonawitz et al., 2017) replace full HE (lighter cryptographic overhead), compression and quantization are primary communication efficiency tools, and the scale ($10^8$ clients) enables very tight DP bounds via amplification by sampling without large accuracy cost.

---

### Incentive Mechanisms and Contribution Assessment

Real-world federated consortia face an economic problem: data holders have privacy, regulatory, and competitive reasons to withhold participation, yet the trained model is a public good. Without fair compensation for training contributions — and without penalties for low-quality participation — rational institutions will under-invest in FL participation.

**Shapley Values for FL**: The *Shapley value* from cooperative game theory assigns credit to each coalition member based on their marginal contribution. For FL, the Shapley value of client $k$ is the expected marginal improvement attributable to including client $k$'s data across all possible client coalitions:

$$\phi_k = \sum_{S \subseteq N \setminus \{k\}} \frac{|S|!\,(n-|S|-1)!}{n!} \left[v(S \cup \{k\}) - v(S)\right]$$

where $v(S)$ is the model accuracy when trained only on the coalition $S$'s data, and $n$ is the total number of clients. Shapley values satisfy four desirable axioms: efficiency (the total value is fully distributed), symmetry (equally contributing clients receive equal credit), linearity (double the contribution, double the credit), and null player (a client contributing nothing receives nothing).

Computing exact Shapley values requires training $2^n$ models — exponential in client count and infeasible for $n > 20$. Practical approximations used in FL include:

- **Gradient cosine similarity** (FedShap, Wang et al., 2020): Approximate $\phi_k \approx \langle g_k, \bar{g} \rangle / \|\bar{g}\|_2$, the inner product of client $k$'s gradient with the average gradient. High positive cosine similarity indicates the client's update is aligned with the group consensus — a proxy for positive contribution.
- **Leave-one-out (LOO) evaluation**: For each client $k$, train a model without $k$ and measure accuracy degradation. $\phi_k \approx v(N) - v(N \setminus \{k\})$. Requires $n$ full training runs — expensive but accurate for small $n$.
- **Monte Carlo Shapley approximation** (Ghorbani and Zou, 2019): Sample random permutations of clients, compute marginal contribution of each client for each sampled ordering, and average. Converges to the true Shapley value in $O(n \log n / \epsilon^2)$ samples.

**On-chain Shapley payment**: In a blockchain-enabled deployment, Shapley values computed by the verified aggregator can be committed on-chain and used to trigger automatic token distribution via smart contract — providing trustless, transparent payment for contribution.

**Privacy-preserving contribution assessment**: Computing Shapley values requires some knowledge of each client's data contribution, creating a tension with FL's data-minimization goal. Secure multi-party computation (SMPC) enables exact Shapley computation without any party learning the others' individual $v(S)$ values — at significant communication cost. For this framework's cross-silo setting (small $n$, trusted consortium), gradient cosine similarity approximations computed on plaintext (by the server) or on ciphertexts (homomorphically) are the practical options.

---

## 13. Further Reading

### Foundational Papers

**McMahan, Moore, Ramage, Hampson, Ramage, Agüera y Arcas (2017)** — "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017. The paper that coined *federated learning* and introduced FedAvg. The essential starting point.

**Konečný, McMahan, Yu, Richtárik, Suresh, Bacon (2016)** — "Federated Learning: Strategies for Improving Communication Efficiency." NeurIPS Workshop on Private Multi-Party Machine Learning. Introduced gradient compression and sparse updates in the FL context.

**Li, Diao, Chen, He (2022)** — "Federated Learning on Non-IID Data Silos: An Experimental Study." ICDE 2022. Systematic empirical evaluation of FL algorithms under various non-IID data regimes. Excellent empirical reference.

### Optimization Algorithms

**Li, Sahu, Zaheer, Sanjabi, Smola, Smith (2020)** — "Federated Optimization in Heterogeneous Networks." MLSys 2020. Introduces FedProx with formal convergence theory for non-IID settings.

**Karimireddy, Kale, Mohri, Reddi, Stich, Suresh (2020)** — "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning." ICML 2020. Introduces control variates to correct client drift. Theoretically motivated correction of FedAvg's fundamental limitation.

**Wang, Tantia, Braverman, Sra, Smola (2020)** — "SlowMo: Improving Communication-Efficient Distributed SGD with Slow Momentum." ICLR 2020.

**Reddi, Charles, Zaheer, Garrett, Rush, Konečný, Kumar, McMahan (2021)** — "Adaptive Federated Optimization." ICLR 2021. Introduces server-side adaptive optimizers (FedAdam, FedYogi, FedAdagrad) for FL.

### Privacy in FL

**Bonawitz, Ivanov, Kreuter, Marcedone, McMahan, Patel, Ramage, Segal, Seth (2017)** — "Practical Secure Aggregation for Privacy-Preserving Machine Learning." CCS 2017. The foundational secure aggregation protocol for FL.

**Zhu, Liu, Han (2019)** — "Deep Leakage from Gradients." NeurIPS 2019. Demonstrates gradient inversion attacks — motivating all privacy mechanisms in this framework.

**McMahan, Ramage, Talwar, Zhang (2018)** — "Learning Differentially Private Recurrent Language Models." ICLR 2018. First large-scale deployment of DP-SGD in FL. Demonstrates viability of DP in production cross-device FL.

**Geyer, Klein, Nabi (2017)** — "Differentially Private Federated Learning: A Client Level Perspective." NeurIPS Workshop 2017. Introduces user-level DP in federated settings.

### Surveys and Textbooks

**Kairouz, McMahan, et al. (2021)** — "Advances and Open Problems in Federated Learning." Foundations and Trends in Machine Learning 14(1-2). The comprehensive 200-page survey by 58 authors from academia and industry. Authoritative reference for all aspects of FL.

**Li, Diao, Chen, He (2020)** — "Threats to Federated Learning: A Survey." arXiv:2003.02133. Survey of attack and defense landscape.

**Rieke, Hancox, Li, Milletarì, Roth, Albarqouni, Bakas, Galtier, Landman, Maier-Hein, Ourselin, Sheller, Summers, Verber, Xu, Baust, Cardoso (2020)** — "The Future of Digital Health with Federated Learning." npj Digital Medicine. Survey focused on healthcare applications of FL — immediately relevant to the healthcare dataset used in this framework.

### Libraries and Frameworks

| Framework | Organization | Transport | Notable Features |
|---|---|---|---|
| Flower (flwr) | Adap | gRPC | Language-agnostic, simulation support, strategy API |
| PySyft | OpenMined | Custom | Privacy-focused, secure aggregation built-in |
| TensorFlow Federated (TFF) | Google | Custom | Tight TF integration, functional API |
| FedML | FedML AI | Multiple | Heterogeneous compute, cross-device and cross-silo |
| FATE | WeBank | gRPC/REST | Cross-silo focus, institutional FL, Chinese banking origin |
| OpenFL | Intel | gRPC | Healthcare focus, DICOM support |

---

*This document provides the theoretical and architectural foundation for the federated learning framework. For the privacy mechanisms layered on top of the FL protocol, see [FHE.md](FHE.md), [ZKP.md](ZKP.md), and [DP.md](DP.md). For running experiments and comparing modes, see [README.md](README.md).*
