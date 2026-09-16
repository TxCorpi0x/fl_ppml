# Fully Homomorphic Encryption (FHE) — Comprehensive Guide

> **Navigation**: [README.md](README.md) | [FHE.md](FHE.md) | [FHE.md](FHE.md)

## Table of Contents

1. [What is FHE?](#1-what-is-fhe)
2. [History and Milestones](#2-history-and-milestones)
3. [Mathematical Foundations](#3-mathematical-foundations)
4. [FHE Scheme Taxonomy](#4-fhe-scheme-taxonomy)
5. [Core Operations](#5-core-operations)
6. [Noise and Bootstrapping](#6-noise-and-bootstrapping)
7. [Security Parameters](#7-security-parameters)
8. [Scheme Comparison Table](#8-scheme-comparison-table)
9. [FHE in Federated Learning](#9-fhe-in-federated-learning)
10. [Implementation in This Framework](#10-implementation-in-this-framework)
11. [Performance Considerations](#11-performance-considerations)
12. [Limitations and Trade-offs](#12-limitations-and-trade-offs)
13. [Further Reading](#13-further-reading)

---

## 1. What is FHE?

**Fully Homomorphic Encryption** is an encryption scheme that allows arbitrary computations to be performed directly on ciphertexts, producing an encrypted result that, when decrypted, equals the result of performing those same computations on the plaintexts.

Formally: given an encryption function $\text{Enc}$ and a decryption function $\text{Dec}$, for any function $f$ and plaintexts $m_1, m_2, \ldots, m_n$:

$$\text{Dec}\!\left(\, f\!\left(\text{Enc}(m_1),\, \text{Enc}(m_2),\, \ldots,\, \text{Enc}(m_n)\right)\!\right) = f(m_1,\, m_2,\, \ldots,\, m_n)$$

This property means a server can compute on encrypted data **without ever seeing the plaintext** — a transformation once called "the holy grail of cryptography."

### Homomorphism Levels

| Term | Operations Supported | Example |
|------|---------------------|---------|
| **Partially Homomorphic (PHE)** | One operation (+ or ×) unlimited | RSA (×), Paillier (+) |
| **Somewhat Homomorphic (SHE)** | Both + and × limited number of times | Early lattice schemes |
| **Leveled FHE** | Both + and × up to depth $L$ | BFV, BGV, CKKS without bootstrapping |
| **Fully Homomorphic (FHE)** | Unlimited +, ×, any function | TFHE, CKKS+bootstrapping |

### The Conceptual Significance of FHE

The intuition behind FHE is worth dwelling on. Traditionally, encryption has been understood as a *barrier* to computation. Data is encrypted to hide it, but to actually process it a server must first decrypt — creating a brief but unavoidable window of exposure. FHE removes this trade-off entirely. Encrypted data can be processed in place, without any decryption step anywhere in the pipeline, and any party performing the computation gains zero knowledge of the underlying values.

Think of it like a special tamper-evident envelope. You can fold the envelope, merge it with other envelopes, and even count the combined contents — all without breaking the seal. When the sealed result finally reaches the person who holds the key, they open it and find the correctly computed answer.

This has profound consequences for trusted computation in untrusted environments. Cloud providers, machine-learning inference services, and medical analytics platforms can become "computationally blind" workers: they execute complex algorithms without ever learning what they are computing over. In federated learning specifically, a central aggregation server can combine gradient updates from dozens of hospitals' private training runs and produce a globally useful model update, while learning nothing about any patient's data — not because it has promised not to look, but because it is *mathematically impossible* for it to do so.

### Why It Took 31 Years

The idea of computing on encrypted data was asked explicitly by Rivest, Adleman, and Dertouzos in 1978 — the same year RSA was invented. It took 31 years and Craig Gentry's 2009 Stanford dissertation to give the first affirmative answer. Understanding *why* it was so hard illuminates how FHE works.

The core obstruction is **noise amplification**. Every lattice-based encryption scheme injects randomness into ciphertexts to make them indistinguishable from random. This noise is not passive — it grows when you compute. Adding two ciphertexts roughly doubles the noise. Multiplying two ciphertexts is far worse: it squares the noise. After even a handful of multiplications, the noise swamps the signal and decryption produces garbage. This is the wall that blocked all earlier proposals for computing on encrypted data.

Gentry's breakthrough was conceptually counterintuitive: **evaluate the decryption circuit itself, inside the encryption**. If you encrypt the secret key under a second encryption layer and then run the decryption algorithm homomorphically on a noisy ciphertext, the result is a freshly encrypted, less-noisy ciphertext that still holds the same plaintext value. This operation — **bootstrapping** — converts a noise-exhausted ciphertext back to a usable one, without ever revealing the plaintext to anyone. Since bootstrapping can be repeated arbitrarily many times, computation depth becomes unlimited, making the scheme truly "fully" homomorphic.

Every practical FHE system since 2009 is, at its core, a more efficient realization of this same fundamental idea. The history of the field is largely the history of making bootstrapping fast enough to be practical.

---

## 2. History and Milestones

| Year | Event |
|------|-------|
| **1978** | RSA discovered — multiplicatively homomorphic |
| **1999** | Paillier cryptosystem — additively homomorphic |
| **2009** | **Craig Gentry's breakthrough**: first FHE construction (IBM/Stanford dissertation) using ideal lattices + squashing + bootstrapping |
| **2010** | BV scheme — simplified Gentry using LWE |
| **2011** | BGV scheme (Brakerski-Gentry-Vaikuntanathan) — first leveled FHE without expensive bootstrapping at every step |
| **2012** | Brakerski's scale-invariant scheme → became BFV (Fan-Vercauteren 2012) |
| **2016** | CKKS scheme (Cheon-Kim-Kim-Song) — approximate arithmetic for real/complex numbers; critical for ML |
| **2016** | FHEW (Ducas-Micciancio) + TFHE (Chillotti et al.) — fast bootstrapping (<1ms per gate) |
| **2020** | TFHE library matures; Concrete library released by Zama.ai |
| **2021** | TenSEAL released (OpenMined) — Python wrapper for Microsoft SEAL |
| **2022** | HElib, OpenFHE, Concrete ML enter production use |
| **2023–2025** | FHE acceleration chips (Cornami, Fabric Cryptography); GPU-FHE libraries |

Gentry's 2009 construction was theoretically revolutionary but computationally impractical (~$10^9\times$ slowdown). Each generation has reduced this overhead by orders of magnitude. Modern TFHE achieves ~10,000× overhead over unencrypted computation for boolean circuits; CKKS achieves ~100–1000× for batched floating-point arithmetic.

---

## 3. Mathematical Foundations

### 3.1 Learning With Errors (LWE)

Before the formal definition, consider the intuition. You have a hidden secret vector $\mathbf{s}$ with $n$ components. An adversary is given many pairs $(\mathbf{a}_i, b_i)$ where $\mathbf{a}_i$ is a randomly chosen vector and $b_i = \langle \mathbf{a}_i, \mathbf{s} \rangle + e_i$ is the inner product with the secret, plus a small error $e_i$. Without the error term, this is a system of linear equations solvable in polynomial time by Gaussian elimination — trivially broken. With even a tiny error added to each sample, no efficient algorithm is known. The noisy version of linear algebra corresponds to finding a short vector in a high-dimensional geometric structure called a **lattice**, a problem that has resisted attack by classical and quantum computers alike for decades.

This surprising sensitivity to noise — the difference between trivially breakable and apparently unbreakable — is what makes LWE useful for cryptography. The same property that makes decryption hard without the key also provides the noise budget that FHE schemes use for computation.

The **LWE problem** (Regev, 2005) is the computational hardness assumption underlying most modern FHE schemes.

**LWE Distribution**: For secret vector $\mathbf{s} \in \mathbb{Z}_q^n$ and small error $e$ drawn from a Gaussian distribution $\chi$:

$$(\mathbf{a},\, b = \langle \mathbf{a}, \mathbf{s} \rangle + e \mod q)$$

Given many samples $(\mathbf{a}_i, b_i)$, it is computationally hard to find $\mathbf{s}$ (even for quantum computers, assuming appropriate parameters).

**Security**: LWE is as hard as worst-case lattice problems (Regev's reduction). Concretely, with $q \approx 2^{32}$, $n = 1024$, $\chi = \mathcal{N}(0, 3.2^2)$: ~128-bit classical security.

### 3.2 Ring-LWE (RLWE)

Plain LWE is provably secure but computationally expensive: each ciphertext carries one scalar value, and operations involve arithmetic over vectors with thousands of elements. The **ring** variant of LWE is the engineering breakthrough that makes FHE practical at scale.

The key observation is that polynomials of degree $N$ have $N$ coefficients. If arithmetic is performed in a ring of polynomials subject to certain structural constraints, then $N$ independent plaintext values can be packed into a single polynomial ciphertext and processed simultaneously. This is the FHE equivalent of SIMD (Single Instruction, Multiple Data) parallelism in CPU design — the same single polynomial addition processes all $N$ plaintext values at once. This "batching" or "slot packing" is what makes homomorphic gradient encryption across large neural network layers tractable.

The structural constraint used — the ring $\mathbb{Z}_q[X] / (X^N + 1)$ with $N$ a power of two — has the additional nice property that polynomial multiplication in this ring can be computed in $O(N \log N)$ using the Number Theoretic Transform (NTT), just as the Fast Fourier Transform makes polynomial multiplication efficient over the complex numbers.

**RLWE** replaces vectors with polynomials in the ring $R_q = \mathbb{Z}_q[X] / (X^N + 1)$, where $N$ is a power of 2. A single RLWE sample packs $N$ LWE samples, giving orders-of-magnitude efficiency gains through batching (SIMD-like parallelism).

An RLWE ciphertext encrypting plaintext polynomial $m(X)$ is a pair $(a, b) \in R_q^2$:

$$b = a \cdot s + e + \Delta \cdot m$$

where:
- $s \in R_q$ is the secret key polynomial
- $e \in R_q$ is a small error polynomial (coefficients $\ll q$)  
- $\Delta = \lfloor q / t \rfloor$ is a scaling factor (plaintext modulus $t$)
- $a \in R_q$ is uniformly random

**Decryption**: $m \approx \lfloor (b - a \cdot s) / \Delta \rfloor \mod t$

The error $e$ must remain small (not grow past $\Delta/2$) for correct decryption.

### 3.3 Polynomial Arithmetic and NTT

Polynomial multiplication is at the heart of every RLWE homomorphic operation. When you multiply two ciphertexts, you are effectively multiplying two degree-$N$ polynomials, which produces a degree-$2N$ result that must then be reduced back to degree $N$ modulo $(X^N + 1)$. The reason $X^N + 1$ is chosen as the reduction polynomial — rather than, say, $X^N - 1$ — is that it makes the ring structure more favorable for security proofs and NTT efficiency simultaneously.

Homomorphic operations on RLWE ciphertexts require polynomial multiplication in $R_q$. Naïve multiplication is $O(N^2)$; the **Number Theoretic Transform (NTT)** reduces this to $O(N \log N)$ — analogous to FFT but over finite fields.

For $N = 8192$: NTT costs ~$N \log_2 N \approx 107,000$ multiplications mod $q$. On modern CPUs with AVX-512, this takes ~microseconds, making RLWE operations practical.

### 3.4 Modulus Switching and Key Switching

**Modulus switching**: Rescale the ciphertext from modulus $q$ to a smaller modulus $q' < q$ to reduce error growth after multiplication. This is the mechanism that makes "leveled" FHE — each multiplicative level consumes one modulus.

$$\mathbf{ct}_{q'} = \left\lfloor \frac{q'}{q} \cdot \mathbf{ct}_q \right\rceil$$

**Key switching**: After homomorphic multiplication, the ciphertext is encrypted under $s^2$ (not $s$). Key switching converts it back to an encryption under $s$ using a relinearization key $\text{rlk}$.

---

## 4. FHE Scheme Taxonomy

### 4.1 BFV (Brakerski/Fan-Vercauteren)

- **Plaintext space**: $\mathbb{Z}_t[X]/(X^N+1)$ — exact integer polynomials
- **Operations**: Exact addition and multiplication (no approximation)
- **Batching**: Up to $N$ integer slots via CRT encoding
- **Use case**: Exact integer arithmetic (database queries, voting, integer ML)
- **Library**: Microsoft SEAL, OpenFHE
- **Noise growth**: $O(N)$ per multiplication

**Encoding for integers**: Chinese Remainder Theorem (CRT) packs $N$ independent integer values into one ciphertext. Each slot can store an integer mod $t$.

### 4.2 BGV (Brakerski-Gentry-Vaikuntanathan)

- Similar to BFV but uses a different noise management strategy (modulus switching vs scale invariance)
- Slightly more efficient for deep circuits
- **Library**: HElib, OpenFHE
- **Difference from BFV**: BGV performs modulus switching *before* multiplication to keep noise small; BFV does it after

### 4.3 CKKS (Cheon-Kim-Kim-Song)

- **Plaintext space**: $\mathbb{C}^{N/2}$ — approximate complex/real numbers
- **Key property**: Treats encryption error as **rounding error**, acceptable for ML
- **Operations**: Approximate addition and multiplication
- **Batching**: Up to $N/2$ complex (or $N$ real) slots via canonical embedding
- **Use case**: Machine learning, statistics, neural network inference
- **Library**: Microsoft SEAL (via TenSEAL), HElib, OpenFHE, Lattigo
- **Scaling factor** $\Delta$: Controls precision (larger $\Delta$ = more precision but consumes more modulus)

**Why CKKS for ML?** The core insight of CKKS is a philosophical reframing of the noise problem. In every previous FHE scheme, noise was the enemy — it had to be kept infinitesimally small and tightly controlled, or the result was corrupted. CKKS observes that in machine learning, this requirement is unnecessarily strict. Gradient values are already floating-point approximations; training is stochastic; inference results are probability estimates — not exact answers. An error of $10^{-9}$ in a gradient component is no different from a rounding error in the 10th decimal place. It is, for all practical purposes, zero.

By accepting that decrypted values will be approximate — that the scheme is performing "approximate arithmetic" rather than exact arithmetic — CKKS can relax the noise constraints dramatically. Instead of fighting the noise, it embraces it as just another tiny floating-point rounding error and designs the scaling factor $\Delta$ to ensure that rounding error is negligibly small compared to the precision the application actually needs. This makes CKKS far more efficient for real-valued data than exact schemes like BFV or BGV, which require very large moduli to maintain exact results throughout computation.

In gradient aggregation for federated learning, we need only addition and scalar multiplication — the simplest possible homomorphic operations. CKKS treats the inherent FHE noise as just another source of floating-point rounding error, enabling practical real-number computation at a small fraction of the cost of deeper FHE circuits.

### 4.4 TFHE (Fast Fully Homomorphic Encryption over the Torus)

- **Plaintext space**: Single bits or small integers (bootstrapped LWE)
- **Operations**: Any boolean gate ($\text{AND}$, $\text{OR}$, $\text{XOR}$, etc.), programmable bootstrapping
- **Key property**: Bootstrapping is fast (<1ms per gate) — can be done after *every* operation
- **Use case**: Arbitrary function evaluation, neural network inference with ReLU
- **Library**: TFHE-lib, Concrete (Zama.ai)
- **Bootstrapping cost**: ~10ms per bootstrapping on CPU (2023 hardware)

**Programmable Bootstrapping (PBS)**: TFHE's most powerful and distinctive feature. In standard FHE schemes, bootstrapping is a necessary evil — a slow, expensive operation to refresh a worn-out ciphertext. TFHE turns this on its head: bootstrapping becomes the primary computational primitive, and the fact that it also refreshes the noise is a bonus.

The mechanism works as follows. The "bootstrapping" in TFHE involves evaluating a test polynomial that is rotated in a ring accumulator by an amount determined by the encrypted input value. This rotation is itself computed homomorphically using the bootstrapping key. Geniusly, the test polynomial can be chosen to encode *any function* $f$ over the message space — so bootstrapping simultaneously refreshes the ciphertext noise and evaluates $f(v)$ for the encrypted input $v$. This means any function expressible as a look-up table (LUT) — including ReLU, sigmoid, any piecewise linear approximation, floor, comparison — costs exactly one bootstrapping operation, regardless of complexity. This makes TFHE's approach to non-linear functions radically different from CKKS, which requires expensive polynomial approximations with bounded depth.

### 4.5 FHEW (Fast Homomorphic Encryption from the Weak)

- Predecessor to TFHE; similar boolean-gate approach
- TFHE improved bootstrapping speed by ~2 orders of magnitude over FHEW
- Less commonly used today

### 4.6 NTRU-based Schemes

- Use NTRU lattice problems instead of LWE
- Potentially more efficient (smaller ciphertext sizes) but with different security assumptions
- Less widely deployed; OpenFHE includes some NTRU variants

---

## 5. Core Operations

### 5.1 Key Generation

All FHE schemes generate three key types:

| Key | Purpose | Who Holds It |
|-----|---------|-------------|
| **Secret key** $sk$ | Decryption | Client only — NEVER shared |
| **Public key** $pk$ | Encryption | Can be shared openly |
| **Evaluation key** $evk$ | Homomorphic operations | Shared with server |

Evaluation keys include:
- **Relinearization key** (rlk): for multiplication (reduces ciphertext size from 3 to 2 polynomials)
- **Galois keys** (galk): for rotation/SIMD slot permutations
- **Bootstrapping key** (bsk): for bootstrapping (TFHE only — per-bit bootstrapping key)

### 5.2 Encryption

$$\text{Enc}_{pk}(m) \rightarrow \mathbf{ct} = (c_0, c_1) \in R_q^2$$

The message $m$ is scaled by $\Delta$ and masked with random $a$ and noise $e$. Encryption is randomized — encrypting the same message twice gives different ciphertexts.

### 5.3 Homomorphic Addition

$$\text{Add}(\mathbf{ct}_1, \mathbf{ct}_2) = (c_0^{(1)} + c_0^{(2)},\, c_1^{(1)} + c_1^{(2)}) \mod q$$

Addition is cheap (coefficient-wise polynomial addition). Error grows additively: $e_{\text{out}} \approx e_1 + e_2$. This is why addition is "free" — it barely consumes the noise budget.

### 5.4 Homomorphic Multiplication

Multiplication is the expensive operation:

$$\text{Mul}(\mathbf{ct}_1, \mathbf{ct}_2) \rightarrow \text{(degree-4 polynomial)} \xrightarrow{\text{relinearize}} (c_0', c_1')$$

Error grows multiplicatively: $e_{\text{out}} \approx q \cdot e_1 \cdot e_2 / \Delta$, consuming one "level" of the modulus chain.

**CKKS rescaling**: After multiplication in CKKS, the scaling factor becomes $\Delta^2$. Rescaling divides by $\Delta$ (drops one modulus prime), restoring the scale to $\Delta$:

$$\mathbf{ct}_{\text{rescaled}} = \left\lfloor \mathbf{ct}_{\text{mult}} / p_i \right\rceil$$

where $p_i$ is the $i$-th prime in the modulus chain.

### 5.5 Rotation (SIMD over Slots)

With $N/2$ plaintext slots, a *rotation by $k$* cyclically shifts all slot values:

$$[m_0, m_1, \ldots, m_{N/2-1}] \xrightarrow{\text{rot}_k} [m_k, m_{k+1}, \ldots, m_{k-1}]$$

Rotations are used to implement reductions (like computing a sum over all slots) and matrix-vector products. Each rotation costs roughly the same as a multiplication and requires a Galois key.

### 5.6 Decryption

$$\text{Dec}_{sk}(\mathbf{ct}) = \lfloor (c_0 + c_1 \cdot sk) / \Delta \rceil \mod t$$

Decryption is fast (a single polynomial inner product) and only possible with the secret key.

---

## 6. Noise and Bootstrapping

### 6.1 The Noise Budget Problem

Every freshly encrypted ciphertext has a **noise budget** ($B$ bits). Each operation consumes budget:

| Operation | Budget Consumed |
|-----------|----------------|
| Addition | ~1 bit |
| Multiplication | ~30–60 bits (1 modulus prime) |
| Rotation | ~30–60 bits |

When budget reaches 0, decryption fails. For CKKS with a modulus chain of $L$ primes, you can perform at most $L$ multiplications before running out of levels.

**Typical budget (CKKS, Microsoft SEAL)**:

| `poly_modulus_degree` | Max coeff_mod bits | Multiplicative levels |
|-----------------------|-------------------|----------------------|
| 4096 | 109 | 1–2 |
| 8192 | 218 | 3–5 |
| 16384 | 438 | 10–12 |
| 32768 | 881 | 24–28 |

### 6.2 Bootstrapping

**Bootstrapping** is the key operation that makes FHE truly "fully" homomorphic — it refreshes the noise budget, allowing unlimited computation depth.

Conceptually, bootstrapping is a recursive trick. To refresh a ciphertext that has accumulated too much noise, you evaluate the *decryption function itself* inside the FHE scheme. You give the server an encryption of your secret key and the noisy ciphertext, and the server homomorphically runs the decryption algorithm on them. The output is a new ciphertext that encrypts the same value as the original, but has much less noise — because the homomorphic decryption circuit, run from scratch, starts with a fresh noise budget.

This works because decryption is a relatively simple computation (a polynomial inner product modulo the noise threshold). If the FHE scheme is powerful enough to evaluate its own decryption circuit within its noise budget — a property Gentry called "bootstrappable" — then bootstrapping is possible. The secret key encryption sent to the server does not compromise security because the server is already assumed to be honest-but-curious (it follows the protocol but tries to infer data), and the encrypted key reveals nothing without the key to decrypt the key, which the client does not share.

**Gentry's insight (2009)**: The decryption circuit $\text{Dec}_{sk}(\mathbf{ct})$ can itself be evaluated *homomorphically* on an encryption of $sk$, producing a "refreshed" ciphertext with a smaller error.

$$\mathbf{ct}_{\text{fresh}} = \text{Eval}(\text{Dec}, \text{Enc}_{pk}(sk), \mathbf{ct})$$

**Bootstrapping costs**:
- CKKS bootstrapping: ~1–10 seconds (CPU, 2024) for $N = 65536$
- TFHE bootstrapping: ~10ms per bit (CPU, 2024) — this is why TFHE evaluates programs gate-by-gate

**Trade-off**: Whether to bootstrap depends on circuit depth. For shallow ML inference (≤5 multiplications deep), bootstrapping is often unnecessary with $N = 16384$ and a long modulus chain.

---

## 7. Security Parameters

FHE security is based on the **hardness of RLWE** (in a chosen-ciphertext-secure sense after adding padding). The [HomomorphicEncryption.org Standard](https://homomorphicencryption.org/standard/) defines security levels:

| Security Level | LWE Dimension $n$ | $\log_2 q$ | Quantum Security |
|--------------|-------------------|-----------|-----------------|
| 128-bit | 1024 | 27 | ✅ Post-quantum |
| 128-bit | 4096 | 109 | ✅ Post-quantum |
| 192-bit | 8192 | 218 | ✅ Post-quantum |
| 256-bit | 16384 | 438 | ✅ Post-quantum |

**Post-quantum security**: RLWE-based FHE is believed to be secure against quantum computers (no known quantum speedup over classical attacks for lattice problems). This is a significant advantage over RSA and ECC which are broken by Shor's algorithm.

**Noise distribution**: The error $e$ is typically sampled from $\chi = \mathcal{N}(0, \sigma^2)$ with $\sigma = 3.2$ (the standard deviation recommended by the HE standard).

**Parameter selection rule of thumb**:
- Always use `poly_modulus_degree ≥ 8192` for 128-bit security with multi-level HE
- The product of all coefficient moduli primes must stay within the bound for your chosen $N$
- Each prime should be ~60 bits except the special "first" prime for CKKS

---

## 8. Scheme Comparison Table

| Property | BFV | BGV | CKKS | TFHE |
|----------|-----|-----|------|------|
| **Plaintext type** | Integers mod $t$ | Integers mod $t$ | Approx. reals/complex | Bits / small ints |
| **Arithmetic** | Exact | Exact | Approximate | Boolean / LUT |
| **Batching** | $N$ integer slots | $N$ integer slots | $N/2$ complex slots | 1 per ciphertext |
| **Addition cost** | Very low | Very low | Very low | ~10ms (bootstrap) |
| **Multiplication cost** | Medium | Medium | Medium | ~10ms (bootstrap) |
| **Non-linear functions** | Hard | Hard | Hard (polynomial approx.) | Native (PBS) |
| **Noise management** | Scale-invariant | Modulus switch | Rescaling | Bootstrap always |
| **Best for ML** | Integer models | Integer models | ✅ Gradient aggregation | ✅ Inference with activations |
| **Library** | SEAL, OpenFHE | HElib, OpenFHE | **TenSEAL**, SEAL, Lattigo | **Concrete ML**, TFHE-rs |
| **Bootstrapping speed** | Minutes | Minutes | ~1–5s | ~10ms/bit |
| **Ciphertext size** | $2N \cdot \lceil \log q \rceil$ bits | Same as BFV | Same as BFV | $N_{\text{LWE}} \cdot \lceil \log q \rceil$ bits |
| **Maturity** | Production | Production | Production | Production |

### When to Choose Which Scheme

The choice between FHE schemes depends on the data type, the operations needed, and where non-linearity appears in the computation.

For **exact integer computation** — database queries, voting systems, integer-valued ML features — **BFV or BGV** provides exact results with no rounding, using CRT slot packing for batch integer arithmetic.

For **approximate real-number computation** in ML applications — gradient aggregation, statistical analysis, neural network layers involving weighted sums — **CKKS** is the practical standard. It handles floating-point data natively, provides float64-grade precision, and batches thousands of values into a single ciphertext. TenSEAL is its recommended Python interface.

For **arbitrary function evaluation** on encrypted values — non-linear activations (ReLU, clipping, argmax), decision trees, comparisons — **TFHE** is the only scheme that handles these natively without deep polynomial approximation. Its Programmable Bootstrapping evaluates any look-up table in a single operation. Concrete ML is its recommended Python interface.

For workflows that require **both approximate arithmetic and arbitrary functions** — such as a neural network where linear layers use CKKS batching and non-linear activations use TFHE's PBS — a hybrid approach combining both schemes is theoretically possible, though complex to engineer and not currently implemented in this framework.

---

## 9. FHE in Federated Learning

In federated learning, FHE protects against the **honest-but-curious (HBC) server** threat: a server that follows the FL protocol but tries to infer private training data from the received gradient updates.

### 9.1 Why Gradient Privacy Matters

In standard federated learning, clients send gradient updates (differences in model weights) rather than raw data. This seems safe — the server never sees your training examples directly. However, research over the past decade has shown this intuition to be dangerously wrong.

Gradients are not opaque. They encode the structure of the data that produced them. Just as a shadow reveals the shape of an object, gradient vectors reveal patterns in the training batch. With enough mathematical machinery, an adversary observing the gradients can effectively reconstruct the original training samples.

Research has shown that gradients can leak significant private information:
- **Gradient inversion attacks** (Zhu et al., 2019; Geiping et al., 2020): reconstruct training images from image classification gradients
- **Model inversion**: recover statistical properties of private data
- **Membership inference**: determine if a sample was used in training

FHE prevents these attacks by ensuring the server never sees plaintext gradients.

### 9.2 FL with FHE — Protocol

The FHE-protected federated learning protocol proceeds as follows each round. First, every participating client trains on its local data and computes a gradient update $\Delta w_i$ — the difference between the locally updated model and the current global model. The client then encrypts this gradient vector using the shared public key, producing a ciphertext $\text{ct}_i = \text{Enc}(pk, \Delta w_i)$, and transmits that ciphertext to the server.

The server, which holds only the evaluation context (no secret key), receives all clients' ciphertexts and performs homomorphic aggregation. It computes the homomorphic sum $\text{ct}_\text{agg} = \sum_i \text{ct}_i$ and optionally scales by $1/n$ (a plaintext scalar multiply). At no point does the server decry pt anything — it operates entirely in ciphertext space and genuinely cannot invert any ciphertext to see a gradient value. The server then broadcasts the aggregated ciphertext $\text{ct}_\text{agg}$ back to all clients.

Each client decrypts with its secret key to recover $\Delta w_\text{global} = \text{Dec}(sk, \text{ct}_\text{agg})$ and applies the update to its local model.

FedAvg aggregation requires only **homomorphic addition** and **scalar multiplication** — both the cheapest possible homomorphic operations. No ciphertext-by-ciphertext multiplication is needed, making this effectively a "zero-level" FHE use case. This is why even relatively modest CKKS parameterisations suffice for FL gradient aggregation, and why the dominant cost is ciphertext size (bandwidth), not arithmetic complexity.

### 9.3 Key Management in FL

The standard approach is **shared secret key** (all clients share one key pair):
- All clients encrypt with the same `pk`
- Server performs homomorphic aggregation
- Clients decrypt the aggregate with `sk`
- **Risk**: Any client who defects exposes the key

Alternative: **Threshold FHE** — the secret key is distributed among $t$-of-$n$ clients; decryption requires cooperation of at least $t$ clients. Not yet implemented in this framework.

### 9.4 Bandwidth Overhead

The dominant cost of FHE in FL is **ciphertext expansion** — CKKS ciphertexts are much larger than the plaintext:

| Model | Plaintext size | CKKS ciphertext (N=8192) | Expansion |
|-------|--------------|--------------------------|-----------|
| Linear (100 params) | 0.4 KB | ~100 KB | 250× |
| Small MLP (1000 params) | 4 KB | ~1 MB | 250× |
| ResNet-18 (11M params) | 44 MB | ~11 GB | 250× |
| This FL model (~13 features) | 0.01 MB | ~244 MB | 24,000× |

The large expansion for our model occurs because we use a fresh ciphertext large enough for 8192-slot RLWE, but only pack 13 values into it — massive slot underutilization. In production systems, packing many gradient vectors into one ciphertext (batch encoding) would reduce this to ~250×.

### 9.5 Limitations of FHE in FL

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| Ciphertext expansion | High bandwidth | Batch packing, parameter compression |
| No non-linear server ops | Server cannot apply clipping/normalization homomorphically | Move clipping to clients |
| Slow for deep models | Training too slow for many FL rounds | Use for inference only, or TFHE |
| Key management complexity | Need secure key distribution | Threshold FHE, PKI |
| No Byzantine protection | HE doesn't prevent poisoned updates | Combine with ZKP (he_tenseal_zkp mode) |

---

## 10. Implementation in This Framework

This project uses two FHE libraries covering the two primary RLWE-based schemes:

### TenSEAL (CKKS) — `he_tenseal` mode

TenSEAL is used for the `he_tenseal` and `he_tenseal_zkp` modes. At the start of training, a shared CKKS context is generated with `poly_modulus_degree = 8192`, a four-prime modulus chain `[60, 40, 40, 60]`, and `global_scale = 2^40`. The client holds the full context (including the secret key) and the server receives a stripped copy with only the evaluation keys.

During each FL round, the client encrypts each model layer's gradient tensor as a flat CKKS vector and serializes it to bytes for transmission over Flower's gRPC channel. The server deserializes all clients' ciphertexts, performs homomorphic addition layer by layer, divides by the number of clients via a plaintext scalar multiply, and returns the aggregated ciphertext. Clients then deserialize, decrypt, and apply the aggregated gradient. At no point do any gradient values appear in cleartext on the server.

For full TenSEAL API documentation, parameter selection rationale, serialization details, key management, and CKKS encoding internals, see [FHE.md](FHE.md).

### Concrete ML (TFHE) — `he_concrete_tfhe` mode

Concrete ML is used for the `he_concrete_tfhe` and `he_concrete_tfhe_zkp` modes. Rather than encrypting gradient updates as floating-point CKKS ciphertexts, this mode quantizes the model's weights to 8-bit integers (int8) and transmits those integer values to the server. The server aggregates quantized integers directly (without FHE evaluation), and the clients dequantize the result.

The reason this mode is described as "TFHE mode" is that the underlying model is compiled to a TFHE circuit during the `compile()` call \u2014 Concrete ML creates the FHE execution infrastructure for eventually running encrypted inference. The int8 transmission during FL rounds is the quantized representation that would be used as input to that FHE circuit in a full encrypted-inference deployment. In the FL aggregation context specifically, the cryptographic protection comes from the discretization itself (granular gradient values that reveal less than float32) combined with ZKP integrity proofs in the `he_concrete_tfhe_zkp` variant.

For full Concrete ML API documentation, TFHE compilation pipeline details, quantization mechanics, Programmable Bootstrapping, and scheme internals, see [FHE.md](FHE.md).

### Mode Comparison in This Framework

| | `he_tenseal` | `he_concrete_tfhe` |
|-|-------------|-------------------|
| Scheme | CKKS | TFHE |
| Arithmetic | Approximate float | Quantized int8 |
| Upload size | 244.7 MB/round | 17.8 MB/round |
| Enc+Dec time | ~1.5s | ~5.3s |
| Accuracy | ~87.1% (=baseline) | ~84.8% (-2.5%) |
| Non-linear server ops | ❌ | ✅ (PBS) |
| Go dependency | ❌ | ❌ |

---

## 11. Performance Considerations

### 11.1 Operation Costs (CKKS, N=8192, CPU)

| Operation | Approximate Time |
|-----------|----------------|
| Key generation | ~0.5s |
| Encryption (8192 slots) | ~1ms |
| Decryption | ~0.5ms |
| Homomorphic addition | ~0.1ms |
| Homomorphic multiplication | ~5ms |
| Galois rotation | ~5ms |
| Serialization (one ciphertext) | ~1ms + I/O |

### 11.2 Optimization Techniques

**Batching**: Pack multiple gradient values into a single ciphertext's slots. With $N/2 = 4096$ slots and 13 gradient values, you could pack $\lfloor 4096/13 \rfloor = 315$ gradient vectors per ciphertext, reducing communication by 315×.

**Level planning**: Minimize multiplications in the circuit. Each multiplication costs one modulus prime (~30 bits). For gradient aggregation (just addition), no multiplications are needed — use a minimal modulus chain.

**SIMD rotations for aggregation**: Use tree-sum via rotations to aggregate all slots in $O(\log N)$ operations instead of $O(N)$ decryptions.

**Hardware acceleration**:
- **AVX-512**: 8× speedup for NTT operations (Intel Ice Lake+)
- **GPU**: 10–100× speedup for NTT (CUDA-based libraries like cuFHE, nuFHE)
- **FPGA/ASIC**: 100–10,000× speedup (experimental)

### 11.3 Memory Requirements

For $N = 8192$ and a 5-level modulus chain (each prime ~60 bits):
- **Ciphertext**: $2 \times N \times L \times \lceil 60/64 \rceil \times 8$ bytes ≈ $2 \times 8192 \times 5 \times 8 = 655$ KB per ciphertext
- **Public key**: ~1.3 MB
- **Relinearization key**: ~7.8 MB
- **Galois keys**: ~7.8 MB × (number of rotations needed)
- **Secret key**: ~655 KB

---

## 12. Limitations and Trade-offs

### 12.1 Computational Overhead

FHE remains 3–6 orders of magnitude slower than unencrypted computation (depending on scheme and circuit depth). This means:
- **Inference**: Feasible today for shallow networks (logistic regression, small MLP) — seconds to minutes
- **Training over encrypted data**: Generally impractical — too slow for FL training loops
- **Gradient aggregation only**: This framework's approach — encrypt for transport, not for computation at depth

### 12.2 Approximate Arithmetic (CKKS)

CKKS introduces rounding errors on the order of $\Delta^{-1} \approx 2^{-40}$ (with `global_scale = 2^40`). This is smaller than float32 precision ($2^{-23}$) — for gradient aggregation, it's completely negligible.

However, when composing many CKKS operations (e.g., 10+ multiplications), errors compound and can reach 1–5% of the true value. For FL gradient aggregation (just one addition and one scalar mul), this is not an issue.

### 12.3 No Circuit Hiding

FHE does not hide the *function* being computed. The server knows what operations it performs on ciphertexts. Only the *data* is hidden. For FL gradient aggregation, this is acceptable — the aggregation function (FedAvg) is public.

### 12.4 Chosen-Ciphertext Attacks

Basic schemes are only CPA-secure (secure against chosen-plaintext attacks). CCA2 security (chosen-ciphertext attacks) requires additional padding/augmentation. For most FL use cases, CPA is sufficient since the adversary (server) cannot ask clients to decrypt arbitrary ciphertexts.

---

## 13. Threshold FHE, Multi-Key Operations, and Selective Encryption

### 13.1 The Key-Sharing Problem in Standard FL+FHE

In this framework's standard HE protocol, all participating institutions share a single secret key $sk$. Any client holding $sk$ can decrypt any ciphertext — including other clients' gradient ciphertexts if they could intercept them. More critically, a single key compromise exposes the entire historical training transcript: every gradient ciphertext from every round becomes readable. In a multi-hospital consortium where each institution's data carries sensitive patient information, this key-sharing model creates a single point of catastrophic failure.

### 13.2 Threshold Homomorphic Encryption

**Threshold FHE** (Asharov et al., 2012; Boneh, Garg, Gorbunov, Kulkarni, Nikolaenko, Rudra, Wichs, 2018) distributes the decryption capability across $n$ parties such that any $t$ of them must cooperate to decrypt — but fewer than $t$ parties learn nothing.

**Key generation**: The algorithm generates a shared public key $pk$ and distributes secret key $sk$ as additive key shares $(sk_1, \ldots, sk_n)$ satisfying $sk = sk_1 + \cdots + sk_n \pmod{q}$ (in the polynomial ring $R_q$ for RLWE-based schemes). Each party $i$ holds only $sk_i$; no party holds $sk$ directly. Encryption under $pk$ works identically to standard CKKS.

**Threshold decryption** (for $t$-of-$n$ scheme):
1. Each party $i$ computes a *partial decryption* $d_i = \text{PartDec}(sk_i, \mathbf{ct})$ — a noisy partial decryption contribution.
2. Any $t$ of the $n$ parties broadcast their partial decryptions.
3. Any party (or an untrusted aggregator) combines $d_1, \ldots, d_t$ via $m = \text{Combine}(d_1, \ldots, d_t)$ to recover the plaintext $m$.
4. Fewer than $t$ partial decryptions reveal nothing, because each $d_i$ alone is computationally indistinguishable from random.

For CKKS-based threshold decryption, the partial decryption $d_i = sk_i \cdot c_1 + e_i'$ adds a small Smudging noise $e_i'$ to mask $sk_i$ before broadcasting. The combined decryption $\sum_i d_i = sk \cdot c_1 + \sum_i e_i'$ recovers the message with error bounded by $n \cdot \|e'\|$, which is absorbed into CKKS's inherent approximation error.

**FL deployment with threshold FHE**:
- All $n$ hospital clients jointly generate $(pk, sk_1, \ldots, sk_n)$ via a distributed key generation (DKG) protocol — requiring one round of communication between all clients.
- During each FL round, clients encrypt gradients under $pk$ exactly as in standard HE mode.
- After aggregation, the server distributes the aggregated ciphertext to all clients.
- Each client broadcasts its partial decryption $d_i$.
- Once $t$ partial decryptions are available, any client reconstructs the final aggregate.
- No single institution (or server) can decrypt without $t$ cooperating parties.

This eliminates the single-point-of-failure key compromise risk. Threshold FHE is not yet implemented in this framework but is the recommended upgrade path for production deployments.

### 13.3 Multi-Key Homomorphic Encryption

**Multi-Key FHE** (López-Alt, Tromer, Vaikuntanathan, 2012) is a more radical approach: each client encrypts under their own *independent* public key $pk_i$. The server can homomorphically aggregate ciphertexts encrypted under different keys:

$$c_{\text{agg}} = \text{MultiKeyEval}\!\left(\text{Avg},\ c_1^{pk_1}, c_2^{pk_2}, \ldots, c_n^{pk_n}\right)$$

The resulting $c_{\text{agg}}$ is encrypted under the *joint key* $(pk_1, \ldots, pk_n)$. Decryption requires that all $n$ parties contribute a partial decryption — a form of $n$-of-$n$ threshold. Each client's gradient is thus protected under their own exclusive key: only by cooperating with all other participants can the aggregate be decrypted.

**Security advantage**: Client $i$'s gradient is encrypted *solely* under $pk_i$. Unless all clients cooperate, no subset can decrypt any individual gradient. Compared to threshold FHE (where the shared $pk$ allows any party to encrypt a message that any $t$ holders can decrypt), multi-key FHE provides stronger isolation at the cost of higher computational overhead: evaluation key sizes and homomorphic operation costs scale linearly with the number of parties.

For the cross-silo scenario with $n = 3$–$10$ hospitals, multi-key CKKS (implemented in Microsoft SEAL's multi-party extension) is computationally feasible, with evaluation times increasing by a factor of roughly $n$ compared to single-key CKKS.

### 13.4 Selective and Sensitivity-Aware Encryption

Full-gradient encryption is computationally and bandwidth-intensive. A practical efficiency insight is that gradient coordinates are not equally privacy-sensitive: parameters of early feature-extraction layers (which learn general, dataset-agnostic representations) carry less identifiable information about training data than parameters of final classification layers (which are highly sensitive to the specific training distribution and can be inverted to reconstruct dataset-specific signals).

**Sensitivity maps** (Huang, Yang, et al., 2025; Yang, 2025) identify high-sensitivity gradient coordinates via spectral analysis of the gradient covariance matrix across training rounds. The top-$k$ coordinates by eigenvalue magnitude are designated "sensitive" and encrypted; the remaining coordinates are transmitted in plaintext. The resulting *partial ciphertext* transmission:

$$\text{PartialEnc}(\nabla) = \left(\underbrace{\text{CKKS.Enc}(\nabla_{\text{sensitive}})}_{\text{high-sensitivity subset, encrypted}},\quad \underbrace{\nabla_{\text{insensitive}}}_{\text{low-sensitivity subset, plaintext}}\right)$$

The server performs standard FedAvg addition on the plaintext component and homomorphic addition on the ciphertext component, then returns both. Each client decrypts the encrypted aggregate and concatenates with the plaintext aggregate.

**Efficiency**: For a typical ResNet-18 (11M parameters) where approximately 20–30% of gradients carry significant sensitivity, selective encryption reduces ciphertext upload volume by 70–80% and encryption time proportionally — achieving 3× end-to-end speedup versus full-gradient CKKS (Yang, 2025). For this framework's healthcare model (13 features), full encryption is already trivially efficient, so selective encryption provides marginal benefit; for larger deployed models it is a high-impact optimization.

**Security analysis**: Selective encryption weakens the protection of the plaintext-transmitted coordinates. An adversary (semi-honest server) observing the plaintext insensitive gradients can perform gradient inversion on those coordinates; only the encrypted sensitive coordinates are opaque. The security guarantee is therefore *partial* — protecting the high-information coordinates while accepting leakage from low-information ones. For deployments requiring complete gradient confidentiality, full CKKS encryption is required regardless of bandwidth cost.

---

## 14. Further Reading

### Foundational Papers

- **Gentry (2009)**: "A Fully Homomorphic Encryption Scheme" — [Stanford dissertation](https://crypto.stanford.edu/craig/craig-thesis.pdf)
- **BGV (2012)**: Brakerski, Gentry, Vaikuntanathan — "(Leveled) Fully Homomorphic Encryption without Bootstrapping" [ePrint 2011/277](https://eprint.iacr.org/2011/277.pdf)
- **CKKS (2017)**: Cheon, Kim, Kim, Song — "Homomorphic Encryption for Arithmetic of Approximate Numbers" [ePrint 2016/421](https://eprint.iacr.org/2016/421.pdf)
- **TFHE (2020)**: Chillotti, Gama, Georgieva, Izabachène — "TFHE: Fast Fully Homomorphic Encryption over the Torus" [Journal of Cryptology](https://link.springer.com/article/10.1007/s00145-019-09319-x)

### FHE in ML

- **CryptoNets (2016)**: Gilad-Bachrach et al. — First neural network inference over CKKS
- **HETAL (2023)**: Efficient privacy-preserving transfer learning with CKKS
- **Concrete ML**: Zama.ai's framework — [github.com/zama-ai/concrete-ml](https://github.com/zama-ai/concrete-ml)
- **Gradient inversion attacks**: Zhu et al. (2019) "Deep Leakage from Gradients" — motivation for FHE in FL

### Standards

- **HomomorphicEncryption.org Standard (2021)**: Parameter recommendations, security levels — [homomorphicencryption.org](https://homomorphicencryption.org)

### Libraries

| Library | Scheme | Language | Notes |
|---------|--------|----------|-------|
| Microsoft SEAL | BFV, CKKS | C++ / Python (via wrapper) | Industry standard |
| TenSEAL | CKKS, BFV | Python (wraps SEAL) | Used in this project |
| Concrete ML | TFHE | Python | Used in this project |
| HElib | BGV, CKKS | C++ | IBM, best for BGV |
| OpenFHE | BFV, BGV, CKKS, FHEW | C++ | Successor to PALISADE |
| Lattigo | BFV, BGV, CKKS, RLWE | Go | For Go services |
| TFHE-rs | TFHE | Rust | Zama's high-performance TFHE |

---

> **Detailed implementation guides**:
> - TenSEAL (CKKS) → [FHE.md](FHE.md)
> - Concrete ML (TFHE) → [FHE.md](FHE.md)
> - Framework overview → [README.md](README.md)


---

## TenSEAL & CKKS — Detailed Guide

> **Navigation**: [FHE.md](FHE.md) | [FHE.md](FHE.md) | [README.md](README.md)

TenSEAL is an open-source Python library by OpenMined that wraps **Microsoft SEAL** to provide a clean Python interface for CKKS and BFV homomorphic encryption. This guide covers the CKKS scheme in depth, TenSEAL's API, its use in this FL framework, and how it compares to Concrete ML.

Understanding TenSEAL requires understanding why CKKS exists as a scheme, what choices it makes and what it sacrifices, and how TenSEAL's design reflects those underlying mathematical constraints. The documentation below proceeds from conceptual foundations through mathematical detail to practical implementation.

### Table of Contents

1. [CKKS Scheme Deep Dive](#1-ckks-scheme-deep-dive)
2. [TenSEAL Architecture](#2-tenseal-architecture)
3. [Parameters Reference](#3-parameters-reference)
4. [TenSEAL API — Complete Reference](#4-tenseal-api--complete-reference)
5. [CKKS Encoding: How Numbers Become Polynomials](#5-ckks-encoding-how-numbers-become-polynomials)
6. [Noise Analysis and Precision](#6-noise-analysis-and-precision)
7. [Usage in This FL Framework](#7-usage-in-this-fl-framework)
8. [Key Management](#8-key-management)
9. [Performance Benchmarks](#9-performance-benchmarks)
10. [Troubleshooting](#10-troubleshooting)
11. [CKKS vs TFHE (TenSEAL vs Concrete)](#11-ckks-vs-tfhe-tenseal-vs-concrete)

---

### 1. CKKS Scheme Deep Dive

#### 1.1 Design Philosophy

CKKS begins with a deceptively simple philosophical shift: **approximate arithmetic is sufficient for machine learning**. In exact FHE schemes like BFV or BGV, a decrypted value must be bitwise-identical to the original plaintext. Any uncontrolled error means a corrupted result. This requirement imposes severe constraints on how much noise the scheme can tolerate, which in turn limits how much computation can be done before a ciphertext becomes unusable.

CKKS relaxes this requirement. It treats the inherent encryption noise not as a catastrophic failure to be avoided, but as a form of *controlled rounding error* — similar in nature to the unavoidable rounding error of IEEE 754 floating-point arithmetic. Just as a float32 value is already an approximation of an infinite-precision real number, a CKKS-encrypted value is an approximation whose error is bounded and predictable.

This philosophical shift has a profound practical consequence: CKKS can pack more computation into each "noise budget" than exact schemes, because it does not need to maintain exact correctness. The scaling factor $\Delta$ (global scale) plays the role of the number of significant figures — you choose how many bits of precision you need, and design the parameter set to maintain that precision through the required number of operations. For gradient aggregation in federated learning, float32 gradients carry 23 bits of precision, and CKKS with `global_scale = 2^40` provides 40 bits — more than sufficient, with precision to spare.

The second fundamental design choice in CKKS is **batching through polynomial slot encoding**. A single CKKS ciphertext is a polynomial in a ring of degree $N$, and it can simultaneously encrypt $N/2$ independent real or complex numbers in parallel — called "slots." All homomorphic operations act on all $N/2$ slots simultaneously, much like how a CPU's SIMD vector instruction processes 8 floats in a single clock cycle. This batching is what makes CKKS efficient for gradient aggregation: rather than encrypting each gradient scalar individually, an entire weight layer (containing potentially thousands of values) is packed into one or a few ciphertexts and transmitted and processed as a unit.

#### 1.2 The CKKS Plaintext Space

CKKS works over the polynomial ring:

$$R = \mathbb{Z}[X] / (X^N + 1), \quad N = 2^k \text{ (power of 2)}$$

The canonical embedding maps $N/2$ complex numbers to elements of this ring:

$$\sigma: \mathbb{C}^{N/2} \rightarrow R_{\mathbb{R}} \quad \text{via inverse DFT at roots of } X^N + 1$$

For real-only data (like gradients), we use the real part: $N$ real slots via mirrored complex encoding.

#### 1.3 Scaling Factor Δ

The central parameter in CKKS is the **scaling factor** $\Delta$ (called `global_scale` in TenSEAL). Before encoding, each value $m_i$ is multiplied by $\Delta$:

$$\hat{m}_i = \lfloor \Delta \cdot m_i \rceil \in \mathbb{Z}$$

This shifts fractional values into the integer polynomial domain. After decryption, dividing by $\Delta$ recovers the approximate real value. The choice of $\Delta$ controls precision:

| `global_scale` | Bits of precision | Decimal digits |
|---------------|------------------|---------------|
| $2^{20}$ | ~20 bits | ~6 digits |
| $2^{30}$ | ~30 bits | ~9 digits |
| $2^{40}$ | ~40 bits | ~12 digits (recommended) |
| $2^{50}$ | ~50 bits | ~15 digits |
| $2^{60}$ | ~60 bits | ~18 digits |

**This framework uses `global_scale = 2^40`** — more than sufficient for gradient aggregation (float32 gradients have 23 bits of mantissa).

#### 1.4 CKKS Ciphertext Structure

A CKKS ciphertext is a pair $(c_0, c_1) \in R_q^2$ where:

$$c_0 = -a \cdot s + \Delta \cdot m + e \mod q$$
$$c_1 = a$$

with:
- $s \in R_2$ — secret key (ternary, coefficients in $\{-1, 0, 1\}$)
- $a \in R_q$ — uniformly random polynomial  
- $e \sim \mathcal{N}(0, \sigma^2)^N$ — Gaussian noise ($\sigma = 3.2$)
- $q$ — the current ciphertext modulus (shrinks with each level used)

**Decryption**: $m \approx (c_0 + c_1 \cdot s) / \Delta \mod q$

#### 1.5 The Modulus Chain

Instead of a single large modulus $q$, CKKS uses a **modulus chain**: a product of distinct primes:

$$q = p_0 \cdot p_1 \cdot p_2 \cdots p_{L-1}$$

Each homomorphic multiplication consumes one prime level:
- Fresh ciphertext: uses all $L$ primes → maximum depth
- After 1 multiplication + rescale: uses $L-1$ primes
- After $k$ multiplications + rescales: uses $L-k$ primes
- Exhausted ciphertext (0 levels left): can only add/subtract, cannot multiply

In TenSEAL, `coeff_mod_bit_sizes` defines the bit-length of each prime in the chain.

**Example** for this framework:

```python
coeff_mod_bit_sizes = [60, 40, 40, 60]
##                      ^               ^
##                      |               |
##                   special prime  special prime
##                   (for key switch)
##                         40       40
##                      ↑      ↑
##                  level 1  level 2  ← multiplicative levels
```

The first and last primes are typically larger (60 bits), forming the "special prime" for key switching. The middle primes (here two 40-bit primes) give $L=2$ multiplicative levels — enough for gradient aggregation (which needs 0 multiplications).

#### 1.6 Homomorphic Operations in CKKS

**Addition**:
$$\text{ct}_1 + \text{ct}_2 = (c_0^{(1)} + c_0^{(2)},\ c_1^{(1)} + c_1^{(2)})$$
Error accumulates linearly. 2 additions ≈ 1 bit of extra noise.

**Multiplication** (followed by rescaling):
$$\text{ct}_1 \times \text{ct}_2 \xrightarrow{\text{tensor}} (d_0, d_1, d_2) \xrightarrow{\text{relinearize}} (c_0', c_1') \xrightarrow{\text{rescale}} \text{ct}_{\text{result}}$$

Rescaling drops one prime level and divides coefficients by $p_i$, restoring the scale from $\Delta^2$ to $\Delta$.

**Rotation** (cyclic shift of slots by $k$):
Implemented via Galois automorphisms: $X \mapsto X^{5^k \mod 2N}$.

Requires a **Galois key** for each distinct rotation step. In TenSEAL, `generate_galois_keys()` pre-computes keys for common rotations.

---

### 2. TenSEAL Architecture

TenSEAL is architecturally a **thin Python facade over Microsoft SEAL**, the industry's most widely deployed FHE library, developed by the Cryptography and Privacy Research group at Microsoft Research. The value TenSEAL adds is not new cryptographic functionality but rather usability: a Pythonic API, automatic serialization for network transport, and integration-friendly design.

The layers, from top to bottom, are:

- **Python Application**: your FL client/server code
- **TenSEAL Python Layer**: provides `ts.context()`, `ts.ckks_vector()`, and related classes
- **pybind11 bindings**: automatically generated C++ ↔ Python bridge — passes inputs down and returns results up with zero copying where possible
- **TenSEAL C++ Core**: handles serialization format, context management, and glue logic
- **Microsoft SEAL (C++)**: performs all actual cryptographic computation — key generation, NTT-based polynomial arithmetic, relinearization, Galois operations
- **NTT / AVX-512 arithmetic**: hardware-optimized number-theoretic transforms using SIMD instructions

The TenSEAL **context** object is the central abstraction. It is a self-contained bundle that carries the scheme parameters (polynomial degree, modulus chain, scale) and optionally the cryptographic keys. A context can be in two modes:

- **Full context** (containing the secret key): held by the client. Can encrypt, decrypt, and perform homomorphic operations.
- **Evaluation context** (without the secret key): sent to the server. Can encrypt (with the embedded public key) and perform all homomorphic operations, but cannot decrypt. Sharing this context with the server exposes no private information.

This two-mode design is central to the FL security model: clients generate a shared context, strip the secret key from it before sharing, and the server operates forever in evaluation mode. Decryption is only ever performed by clients, with their local copy of the full context.

---

### 3. Parameters Reference

#### 3.1 Core Parameters

| Parameter | TenSEAL Arg | Type | Description |
|-----------|-------------|------|-------------|
| Polynomial modulus degree | `poly_modulus_degree` | int (power of 2) | Ring dimension $N$. Larger = more slots + more security + slower |
| Coefficient modulus | `coeff_mod_bit_sizes` | list[int] | Bit lengths of each prime in the modulus chain |
| Global scale | `global_scale` | float | Scaling factor $\Delta$ for encoding |
| Scheme | `scheme` | `ts.SCHEME_TYPE.CKKS` | Use CKKS for real numbers |

#### 3.2 Security vs Capacity Trade-off

| $N$ | Slots | Security (log₂ q limit) | Max levels | Typical use |
|-----|-------|-------------------------|-----------|-------------|
| 4096 | 2048 | 109 bits | 1–2 | Toy/test |
| **8192** | **4096** | **218 bits** | **3–5** | **This framework** |
| 16384 | 8192 | 438 bits | 10–12 | Deep circuits |
| 32768 | 16384 | 881 bits | ≥24 | Very deep circuits |

#### 3.3 This Framework's Parameters

```python
## From `ppflx.keys` or programmatic example
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree = 8192,
    coeff_mod_bit_sizes = [60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()
context.generate_relin_keys()
```

**Why these parameters?**
- `N = 8192`: 4096 slots (more than enough for any gradient vector), 128-bit security
- `[60, 40, 40, 60]`: Sum = 200 bits ≤ 218 bit limit for $N=8192$; 2 multiplicative levels
- `global_scale = 2^40`: Float64-grade precision (40 bits > float32's 23-bit mantissa)
- Galois keys: Generated for potential gradient rotation/reduction
- Relin keys: Generated in case multiplication is needed

#### 3.4 Parameter Constraints

The sum of all primes in `coeff_mod_bit_sizes` must not exceed the security limit for the chosen $N$:

$$\sum_i \text{coeff\_mod\_bit\_sizes}[i] \leq \text{MaxCoeffModBits}(N)$$

Violating this causes a `RuntimeError: parameters are not valid`. 

For `[60, 40, 40, 60]` with $N=8192$: sum = 200 ≤ 218 ✅

#### 3.5 Scale Consistency

In CKKS, all ciphertexts in an addition must have the **same scale** within a tolerance. Mismatched scales cause the error: `"scale out of bounds"`.

After a multiplication, the scale becomes $\Delta^2$. After rescaling, it returns to $\Delta$. When adding:
- Ciphertext at level $l$ (scale $\Delta$) + Ciphertext at level $l$ (scale $\Delta$): ✅
- Ciphertext at level $l$ + Ciphertext at level $l-1$ (after one more rescale): ❌ different scales

This is managed in the framework by ensuring all client ciphertexts use the same number of levels before sending to the server.

---

### 4. TenSEAL API — Complete Reference

#### 4.1 Context Management

The framework stores contexts as raw bytes behind a short header (`ppflx.keys.he_tenseal`), never with pickle: unpickling a swapped key file can execute code.

```python
import tenseal as ts
from ppflx.keys.he_tenseal import read_context_bytes, write_context

## --- Creating a context ---
context = ts.context(
    scheme=ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40

## Generate all evaluation keys
context.generate_galois_keys()   # for rotations
context.generate_relin_keys()    # for multiplication

## --- Saving the full context (with secret key) ---
write_context("keys/he_tenseal/secret_context.bin", context.serialize(save_secret_key=True))

## --- Creating a server-side context (no secret key) ---
write_context("keys/he_tenseal/public_context.bin", context.serialize(save_secret_key=False))

## --- Loading a context ---
context = ts.context_from(read_context_bytes("keys/he_tenseal/secret_context.bin"))  # restores with secret key

## --- Checking context properties ---
print(context.is_private())    # True if has secret key
print(context.poly_modulus_degree())  # 8192
```

#### 4.2 CKKS Vector — Encrypt and Decrypt

```python
## Encrypt a list of floats
plain_values = [0.12, -0.34, 0.99, ...]   # any length ≤ N/2 = 4096
ct = ts.ckks_vector(context, plain_values)

## The ciphertext packs all values into one RLWE ciphertext
## ct has shape (len(plain_values),) logically

## Decrypt
decrypted = ct.decrypt()   # Returns list[float], approximately equal to plain_values

## Decrypt with explicit scale recovery
decrypted = ct.decrypt(context)   # same as above, context can be passed

## Roundtrip error check
import numpy as np
err = np.max(np.abs(np.array(decrypted) - np.array(plain_values)))
print(f"Max error: {err:.2e}")  # Typically ~1e-12 for global_scale=2^40
```

#### 4.3 Serialization (for network transport)

```python
## Encrypt
ct = ts.ckks_vector(context, gradients)

## Serialize to bytes (for gRPC / network transport)
ct_bytes = ct.serialize()          # bytes object (~several MB)

## Server-side deserialization
server_context = ts.context_from(server_ctx_bytes)   # no secret key
ct_received = ts.ckks_vector_from(server_context, ct_bytes)

## After aggregation, serialize the result
ct_agg_bytes = ct_agg.serialize()

## Client-side deserialization and decrypt
ct_agg = ts.ckks_vector_from(context, ct_agg_bytes)
result = ct_agg.decrypt()
```

#### 4.4 Homomorphic Arithmetic

```python
## Addition of two ciphertexts
ct_sum = ct1 + ct2            # or ct1.add(ct2)
ct_sum = ct1.add_(ct2)        # in-place

## Addition with a plaintext (no encryption needed for small constant)
ct_shifted = ct + 1.5         # adds 1.5 to each slot
ct_shifted = ct + [1.5, 2.0, ...]  # add a plaintext vector (same length)

## Scalar multiplication
ct_scaled = ct * 0.5          # multiply all slots by 0.5
ct_scaled = ct * [0.5, 1.0, ...]  # element-wise with plaintext vector

## Negation
ct_neg = -ct

## Sum of all slots (reduce to scalar)
scalar_ct = ct.sum()          # uses log2(N/2) rotations + additions

## Dot product with plaintext vector
dot_ct = ct.dot(plaintext_vector)

## Matrix-vector product (CKKS supports matrix encoding tricks)
## For a num_cols-wide matrix, use ts.enc_matmul_encoding or manual rotation-sum
```

#### 4.5 Advanced: Rotation

```python
## Rotate slots left by k positions
ct_rotated = ct.roll(k)       # TenSEAL API: negative = left, positive = right
## Requires Galois keys for step k to have been generated

## Example: compute sum using tree rotation
def homomorphic_sum_slots(ct, n_slots):
    """Sum all slots using log2(n_slots) rotations."""
    step = 1
    while step < n_slots:
        ct = ct + ct.roll(-step)
        step *= 2
    return ct
```

#### 4.6 CKKS Matrix-Vector Multiplication

For neural network layers, CKKS can evaluate encrypted matrix-vector products using the **diagonal method** (Halevi-Shoup, 2014):

```python
def ckks_matmul(context, matrix, ct_vector):
    """
    Multiply plaintext matrix by encrypted vector using diagonals.
    Time: O(n * log n) rotations for n×n matrix.
    """
    n = len(matrix)
    result = None
    for i in range(n):
        # Extract i-th diagonal of matrix
        diag = [matrix[(j + i) % n][j] for j in range(n)]
        # Multiply encrypted vector by plaintext diagonal
        term = ct_vector * diag
        # Rotate by i positions
        term = term.roll(-i) if i > 0 else term
        # Accumulate
        result = term if result is None else result + term
    return result
```

---

### 5. CKKS Encoding: How Numbers Become Polynomials

#### 5.1 The Conceptual Challenge of Encoding

The most conceptually striking aspect of CKKS — and RLWE-based FHE generally — is the encoding step: how do real-valued numbers (like neural network gradients) become polynomials, and why does arithmetic on polynomials correspond to arithmetic on those original numbers?

The answer lies in a classical mathematical correspondence between polynomials and their values at specific points, formalized in CKKS as the **canonical embedding**. A polynomial of degree $N$ is completely determined by its values at $N$ distinct points. Conversely, given $N$ values at $N$ known points, there is exactly one polynomial of degree less than $N$ that passes through all of them. This is the Lagrange interpolation theorem.

CKKS exploits this correspondence. The $N/2$ numbers you want to encrypt become the "values at specific points" — in particular, at $N/2$ of the $2N$-th roots of unity in the complex plane (roots of $X^N + 1$). The polynomial whose values at those points match your numbers is computed via the inverse FFT (or number-theoretic equivalent). That polynomial, with its coefficients scaled up by $\Delta$ and rounded to integers, is the CKKS plaintext polynomial. When you add two such polynomials, the result's values at those same points are the elementwise sums of the originals. This is why polynomial addition corresponds to slot-by-slot addition of the original numbers.

Multiplication of polynomials in the ring $R = \mathbb{Z}[X]/(X^N+1)$ corresponds, under the canonical embedding, to slot-wise multiplication of the encoded values. This is the mathematical engine that gives CKKS its SIMD semantics: one polynomial operation touches all $N/2$ slots simultaneously.

#### 5.2 The Canonical Embedding

CKKS uses the **canonical embedding** $\sigma: R \rightarrow \mathbb{C}^N$, mapping a polynomial $p(X) \in R$ to its evaluations at the $N$ primitive $2N$-th roots of unity:

$$\sigma(p) = (p(\zeta^1), p(\zeta^3), p(\zeta^5), \ldots, p(\zeta^{2N-1}))$$

where $\zeta = e^{i\pi/N}$.

For real-valued messages $m \in \mathbb{R}^{N/2}$, we use the conjugate-invariant subring — effectively mirroring the complex values so that half the coordinates are the conjugates of the other half, giving $N/2$ real slots.

#### 5.2 Encoding Steps

To encode $[m_0, m_1, \ldots, m_{k-1}]$ (with $k \leq N/2$):

1. **Pad**: if $k < N/2$, zero-pad to $N/2$ slots
2. **Scale**: multiply by $\Delta$ → $[\Delta m_0, \Delta m_1, \ldots]$ (integers)
3. **Inverse canonical embedding**: apply $\sigma^{-1}$ (essentially an inverse FFT at the 2N-th roots)
4. **Round**: round coefficients to nearest integers
5. **Result**: a polynomial $p(X) \in R$ with integer coefficients

The encoded polynomial $p(X)$ is the CKKS plaintext.

#### 5.3 Why Encoding Matters for Performance

The full $N/2 = 4096$ slots in a single ciphertext is the key efficiency lever. Packing $k$ gradient values into one ciphertext means one ciphertext instead of $k$ ciphertexts, and one pair of polynomial additions at the server instead of $k$ separate operations. This is a $k$-fold reduction in bandwidth and computation simultaneously.

It is worth understanding why this framework exhibits a 24,000× ciphertext expansion ratio — far larger than the "typical" 250× often cited for CKKS. The reason is slot underutilization. With $N/2 = 4096$ available slots and a healthcare model with only 13 gradient values, just 0.3% of the ciphertext's capacity is used. The remaining 4083 slots contain zeros. The ciphertext is the same size whether it carries 13 values or 4096 values, so the effective expansion per value is $4096/13 \approx 315$ times worse than optimal. In a production deployment, one would encode many gradient layers together — or even multiple clients' gradients simultaneously — into a single ciphertext, driving slot utilization to nearly 100% and reducing effective expansion to the baseline 250×.

---

### 6. Noise Analysis and Precision

#### 6.1 Error Sources in CKKS

| Source | Magnitude | Stage |
|--------|-----------|-------|
| **Encoding rounding** | $\leq 1/2$ (integer rounding) | At encode time |
| **Encryption noise** | $\sim N(0, \sigma^2)$, $\sigma=3.2$ | At encrypt time |
| **Addition noise** | Accumulates linearly | Per addition |
| **Multiplication noise** | ~$B^2/\Delta$ per multiplication | Per multiply |
| **Rescaling error** | ~$N \cdot \sigma^2$ | Per rescale |

#### 6.2 Total Error After Aggregation (FL use case)

For FL gradient aggregation with $n$ clients (all additions, no multiplications):

$$\text{Total error} \approx n \cdot (2\sigma\sqrt{N} / \Delta)$$

With $n=2$, $N=8192$, $\sigma=3.2$, $\Delta=2^{40}$:

$$\text{error} \approx 2 \cdot \frac{2 \times 3.2 \times \sqrt{8192}}{2^{40}} \approx 2 \cdot \frac{579}{10^{12}} \approx 10^{-9}$$

This is 10 orders of magnitude smaller than the gradient values themselves (typically $10^{-3}$ to $10^{-1}$). **CKKS precision is negligible for FL gradient aggregation.**

#### 6.3 Detecting Scale Issues

If you see:
```
RuntimeError: result ciphertext is transparent
RuntimeError: scale out of bounds
```

This typically means:
- Ciphertexts at different levels/scales were added
- The modulus chain was exhausted (too many multiplications)

**Fix**: Ensure all ciphertexts entering an aggregation step have used the same number of multiplicative levels. In this framework, clients never multiply ciphertexts — they only encrypt once and the server adds — so this is naturally avoided.

---

### 7. Usage in This FL Framework

#### 7.1 Key Generation (ppflx.keys CLI / programmatic example)

```bash
python -m ppflx.keys generate he_tenseal
## writes keys/he_tenseal/secret_context.bin (clients) and keys/he_tenseal/public_context.bin (server)
```

Programmatically:

```python
from ppflx.keys.he_tenseal import generate

generate(secret_path="keys/he_tenseal/secret_context.bin", public_path="keys/he_tenseal/public_context.bin")
```

Loaders refuse anything that isn't a key file of this format, including legacy pickle files.

#### 7.2 Client Side (`client.py`)

```python
## Load context with secret key
from ppflx.keys.he_tenseal import load_client

context = load_client("keys/he_tenseal/secret_context.bin")

## After local training — encrypt gradient update
def encrypt_parameters(state_dict, context):
    """Encrypt all model weight tensors."""
    encrypted_layers = {}
    for name, tensor in state_dict.items():
        flat = tensor.flatten().tolist()
        ct = ts.ckks_vector(context, flat)
        encrypted_layers[name] = ct.serialize()   # bytes
    return encrypted_layers

## Decrypting aggregated result
def decrypt_parameters(encrypted_dict, context, model_shapes):
    """Decrypt server's aggregated result."""
    state_dict = {}
    for name, ct_bytes in encrypted_dict.items():
        ct = ts.ckks_vector_from(context, ct_bytes)
        flat = ct.decrypt()
        shape = model_shapes[name]
        state_dict[name] = torch.tensor(flat[:math.prod(shape)]).reshape(shape)
    return state_dict
```

#### 7.3 Server Side (`server.py`)

```python
## Load evaluation context (NO secret key); refuses a file that contains one
from ppflx.keys.he_tenseal import load_server

server_context = load_server("keys/he_tenseal/public_context.bin")

def aggregate_he_tenseal(encrypted_updates: list[dict]) -> dict:
    """
    Aggregate encrypted gradient updates via homomorphic addition.
    Server never decrypts — performs FedAvg in ciphertext space.
    """
    n = len(encrypted_updates)
    aggregated = {}
    
    for layer_name in encrypted_updates[0].keys():
        # Deserialize all clients' ciphertexts for this layer
        cts = [
            ts.ckks_vector_from(server_context, update[layer_name])
            for update in encrypted_updates
        ]
        
        # Homomorphic addition (FedAvg numerator)
        ct_sum = cts[0]
        for ct in cts[1:]:
            ct_sum += ct   # += is in-place homomorphic addition
        
        # Scalar division by n (FedAvg denominator)
        ct_avg = ct_sum * (1.0 / n)
        
        aggregated[layer_name] = ct_avg.serialize()
    
    return aggregated
```

#### 7.4 Data Flow Diagram

```
CLIENT                          SERVER
  │                               │
  │  [train locally]              │
  │  gradient: [0.12, -0.34, ...] │
  │                               │
  │  ct = Enc(pk, gradient)       │
  │  ─── ct_bytes (244 MB) ──────►│
  │                               │  ct.sum + ct_other
  │                               │  ct_avg = ct_sum * (1/n)
  │                               │  [server never decrypts]
  │◄─── ct_avg_bytes (244 MB) ────│
  │                               │
  │  avg = Dec(sk, ct_avg)        │
  │  update model                 │
```

#### 7.5 Transport

TenSEAL ciphertexts are transmitted as serialized bytes (uint8 arrays) in the `ArrayRecord` of the ClientApp's train reply, which the SuperNode sends to the SuperLink over gRPC. The size for one ciphertext (N=8192, 4 primes, 13 values):

```
size ≈ 2 × N × Σ(coeff_mod_bit_sizes) / 8
     = 2 × 8192 × (60+40+40+60) / 8
     = 2 × 8192 × 25 bytes
     ≈ 409,600 bytes ≈ 400 KB per layer
```

With multiple layers, this is why total upload is ~244 MB/round — each layer is a separate ciphertext.

**Optimization**: Pack all layer parameters into one flat vector → one ciphertext. This reduces to ~400 KB total instead of 244 MB, at the cost of losing per-layer granularity.

---

### 8. Key Management

#### 8.1 Key Sizes

| Key | Size (N=8192, [60,40,40,60]) |
|-----|------------------------------|
| Secret key $sk$ | ~400 KB |
| Public key $pk$ | ~800 KB |
| Relinearization key | ~4.8 MB |
| Galois keys (all rotations) | ~100 MB |
| Galois keys (selected rotations) | ~5–20 MB |

**Galois keys** are the largest. TenSEAL's `generate_galois_keys()` generates keys for all $N/2$ rotations. If only specific rotations are needed (e.g., rotation by 1 for summation), generating only those reduces key size significantly:

```python
## Generate only Galois keys for powers-of-2 rotations (for tree-sum)
context.generate_galois_keys()   # all — safe but large
## (TenSEAL does not yet expose selective Galois key generation at Python level;
## selective generation is available in raw Microsoft SEAL C++ API)
```

#### 8.2 Security Properties

- **Secret key**: Must never leave the client. If compromised, all past ciphertexts can be decrypted.
- **Public key**: Safe to distribute openly. Used for encryption only.
- **Evaluation keys**: Safe to share with server. Do not enable decryption — only enable homomorphic operations.
- **Semantic security**: Two encryptions of the same message produce different ciphertexts (randomized encryption). Server cannot determine if two ciphertexts contain the same gradient.

#### 8.3 Forward Secrecy

Standard CKKS does not provide perfect forward secrecy (PFS). If the secret key is eventually compromised, past ciphertext transcripts can be decrypted. For PFS in FL, a new key pair should be generated each round — not implemented in this framework (one key for entire training).

---

### 9. Performance Benchmarks

#### 9.1 Operation Timings (Apple M1 Pro, TenSEAL 0.3.14, N=8192)

| Operation | Time |
|-----------|------|
| Key generation (all keys) | ~0.8s |
| Encrypt 4096 floats | ~1.2ms |
| Decrypt 4096 floats | ~0.6ms |
| Serialize ciphertext | ~0.2ms + I/O |
| Deserialize ciphertext | ~0.4ms + I/O |
| Homomorphic addition | ~0.8ms |
| Scalar multiplication | ~0.9ms |
| Homomorphic multiplication | ~8ms |
| Galois rotation by 1 | ~7ms |

#### 9.2 FL Round Timing Breakdown (healthcare, 2 clients)

| Phase | Time |
|-------|------|
| Local training | ~40s |
| Encrypt all layers (client) | ~1.2ms |
| Serialize + transmit | ~0.3s (local loopback) |
| Deserialize (server) | ~0.4ms |
| Homomorphic aggregation | ~1.6ms |
| Serialize + send back | ~0.3s |
| Deserialize (client) | ~0.4ms |
| Decrypt | ~0.6ms |
| **Total crypto overhead** | **~1.5s** |
| **Total round time** | **~41.5s** |

The 244 MB ciphertext transmission dominates over arithmetic when network is the bottleneck.

#### 9.3 Scaling with Model Size

| Model params | Plaintext | Ciphertext | Enc time | Transmit (1Gbps LAN) |
|-------------|-----------|------------|----------|---------------------|
| 13 (this FL model) | 0.1 KB | ~400 KB | 1.2ms | 3ms |
| 1,000 params | 8 KB | ~1 MB | 1.3ms | 8ms |
| 100,000 params | 800 KB | ~25 MB | 6ms | 200ms |
| 1M params | 8 MB | ~244 MB | 60ms | 2s |

With proper packing (full 4096-slot utilization), ciphertext overhead reduces from 24,000× to ~250×.

---

### 10. Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: scale out of bounds` | Ciphertexts at different scales added | Ensure all ciphertexts use same depth before aggregation |
| `RuntimeError: parameters are not valid` | `coeff_mod_bit_sizes` sum exceeds limit for N | Reduce bits or increase N |
| `RuntimeError: result ciphertext is transparent` | Encryption with wrong context, or zero noise | Use proper context; check context has public key |
| Decrypted values wildly wrong (not ~0 error) | Noise too large; levels exhausted | Reduce multiplicative depth; increase N or modulus |
| Serialization size unexpectedly large | All Galois keys included in context | Use `context.serialize(save_secret_key=False)` for server |
| `ValueError: size of vector is too large` | Input vector length > N/2 | Split into chunks of size N/2 |
| Memory error during key generation | Galois keys for large N require lots of RAM | Use smaller N or generate only needed rotation keys |
| `AttributeError: 'bytes' has no attribute 'decrypt'` | Forgot to deserialize bytes | Call `ts.ckks_vector_from(ctx, bytes)` first |

---

### 11. CKKS vs TFHE (TenSEAL vs Concrete)

CKKS (TenSEAL) and TFHE (Concrete ML) are both fully homomorphic encryption schemes, and both achieve 128-bit post-quantum security, but they represent fundamentally different design philosophies, different mathematical foundations, and different trade-off profiles. Choosing between them is not a matter of one being "better" — it is about matching the scheme's strengths to the problem's requirements.

#### 11.1 The Fundamental Philosophical Difference

**CKKS is a scheme designed for continuous, real-valued data.** Its foundational insight is that approximate arithmetic is indistinguishable from exact arithmetic in most machine learning contexts. A gradient value that decrypts to 0.1234567890 instead of 0.1234567891 introduces an error smaller than the gradient's own numerical noise floor from stochastic training. CKKS leans into this: it lets the noise grow in a controlled way, uses a large scaling factor to keep it negligible, and in exchange achieves exceptional efficiency for data that is naturally floating-point. The entire gradient vector for a model layer is packed into a single polynomial ciphertext and processed with one polynomial addition — regardless of whether that vector has 13 elements or 4000.

**TFHE is a scheme designed for arbitrary computation on discrete values.** Its foundational insight is that any computation can be decomposed into look-up table evaluations over quantized integers, and that bootstrapping — normally the most expensive operation in FHE — can simultaneously evaluate any such table for free. TFHE does not try to be efficient on floating-point data; instead, it first quantizes all data to integers, then evaluates the model circuit gate by gate (or look-up by look-up) in the encrypted domain. The power is generality: TFHE can evaluate ReLU, clipping, comparison, argmax, and essentially any function that can be expressed as a table over small integers. CKKS cannot do this natively.

These philosophical differences cascade into every observable property of the two schemes: ciphertext size, accuracy, bandwidth, server computation model, and the kinds of operations the server can perform.

#### 11.2 Batching and Ciphertext Size

One of the most practically significant differences is how the two schemes handle multiple values. CKKS, as an RLWE scheme, packs $N/2 = 4096$ values into a single polynomial ciphertext. The ciphertext's size is fixed by $N$ and the modulus chain, regardless of how many values are packed into it. For a model with thousands of parameters, CKKS may represent the entire layer in a handful of ciphertexts.

TFHE, in contrast, operates on integers one at a time (or one "word" at a time). Each encrypted value is a separate LWE sample. For a model with 13 gradient parameters at 8-bit quantization, you have 13 independent TFHE ciphertexts, each roughly 4–8 KB. This is far smaller per ciphertext than a CKKS polynomial (hundreds of KB), but CKKS was fitting 4096 values into those hundreds of KB, whereas TFHE fits one. For the specific gradient sizes in this framework, TFHE's per-parameter overhead works out to ~17.8 MB total versus CKKS's ~244 MB — TFHE is smaller because it avoids the massive slot underutilization caused by fitting only 13 values into a 4096-slot CKKS ciphertext.

#### 11.3 Precision and Accuracy

CKKS introduces only rounding error, on the order of $\Delta^{-1} \approx 2^{-40} \approx 10^{-12}$. For gradient aggregation, this is negligible — orders of magnitude smaller than float32 precision ($2^{-23} \approx 10^{-7}$) or typical gradient magnitudes ($10^{-3}$ to $10^{-1}$). CKKS produces results that are, for all practical purposes in FL, indistinguishable from unencrypted results. The healthcare model shows ~87.1% accuracy with CKKS versus ~87.3% baseline — a 0.2% difference attributable to floating-point randomness across runs, not CKKS error.

TFHE introduces quantization error, which is both more perceptible and conceptually different. Quantization maps floating-point gradients to the nearest integer on an 8-bit grid ($256$ discrete levels covering the observed range of values). This is a coarser approximation: a quantization error of up to $0.4\\%$ of the value range per gradient component accumulates over training rounds. The healthcare model shows ~84.8% accuracy with TFHE — a ~2.5% accuracy penalty relative to baseline. Whether this is acceptable depends entirely on the application. For many use cases, a 2.5% accuracy penalty in exchange for stronger privacy guarantees over arbitrary computation is a worthwhile trade.

#### 11.4 What the Server Can Compute

This is the most architecturally significant difference between the two schemes in a federated learning context.

With CKKS, the server can only perform **linear operations on ciphertexts**: addition, subtraction, and multiplication by plaintext scalars or vectors. It cannot compare ciphertext values, clip them, compute norms, or apply any non-linear function without the results being meaningless (because CKKS's non-linear function support requires polynomial approximation with bounded depth). In this framework, FedAvg aggregation (sum then divide) is entirely linear, so CKKS supports it perfectly. But if you wanted the server to apply gradient clipping — capping each gradient at some norm bound — you could not do it with CKKS on encrypted data without the client's help.

With TFHE, the server can perform **arbitrary functions on ciphertexts** via Programmable Bootstrapping. Any function expressible as a look-up table over integer inputs (which includes clipping, ReLU, argmax, secure comparison, etc.) can be evaluated in a single bootstrapping operation. This opens the door to server-side privacy-preserving gradient sanitization, anomaly detection on encrypted updates, or even full model inference on server-side encrypted data.

#### 11.5 Summary of Trade-offs

| Metric | CKKS / TenSEAL | TFHE / Concrete ML |
|--------|---------------|--------------------|
| Upload per FL round | 244.7 MB | 17.8 MB |
| Accuracy loss (healthcare) | ~0.2% | ~2.5% |
| Server computation model | Linear ops only | Arbitrary functions |
| Non-linear server ops | ❌ No | ✅ Yes (via PBS) |
| Enc + Dec time | ~1.5s | ~5.3s |
| Server aggregation time | ~1.6ms | ~500ms |
| Precision | Float-grade (~$10^{-12}$ error) | 8-bit quantized (~0.4% error) |
| Best suited for | Gradient aggregation | Full encrypted inference |

#### 11.6 When to Choose Which

**Prefer CKKS (TenSEAL)** when the primary goal is protecting gradient values during transmission and aggregation, accuracy is paramount, and the computation the server performs is limited to linear aggregation (FedAvg, weighted average). This is the ideal fit for FHE-protected FL where the server's role is simply to combine and re-distribute updates.

**Prefer TFHE (Concrete ML)** when you need the server to perform non-linear operations on encrypted updates (clipping, filtering, comparison), when bandwidth is constrained and the smaller ciphertext size matters, when you are building end-to-end encrypted inference rather than just encrypted aggregation, or when the model is already fundamentally discrete (decision trees, XGBoost, integer-weight models).

#### 11.7 Hybrid Approach

This framework's `he_tenseal_zkp` and `he_concrete_tfhe_zkp` modes layer ZKP proofs on top of HE, providing both confidentiality (HE prevents the server from seeing gradient values) and integrity (ZKP prevents clients from sending malformed or poisoned updates). The TFHE+ZKP mode is more bandwidth-efficient; the CKKS+ZKP mode has higher accuracy. See [ZKP.md](ZKP.md) for details.

---

> **Related documentation**:
> - [FHE.md](FHE.md) — Comprehensive FHE theory (all schemes)
> - [FHE.md](FHE.md) — Concrete ML / TFHE deep dive
> - [README.md](README.md) — All 10 modes compared


---

## Concrete ML & TFHE — Detailed Guide

> **Navigation**: [FHE.md](FHE.md) | [FHE.md](FHE.md) | [README.md](README.md)

Concrete ML is Zama.ai's Python library that compiles machine learning models to run under **TFHE** (Torus Fully Homomorphic Encryption). Unlike TenSEAL which wraps Microsoft SEAL's CKKS, Concrete ML uses a quantize-then-compile approach that translates an sklearn/PyTorch model into an optimized FHE boolean circuit. This guide covers the TFHE scheme in depth, Concrete ML's compilation pipeline, its use in this FL framework, and a detailed comparison with TenSEAL/CKKS.

### Table of Contents

1. [TFHE Scheme Deep Dive](#1-tfhe-scheme-deep-dive)
2. [Concrete ML Architecture](#2-concrete-ml-architecture)
3. [Quantization: Bridging Floats to FHE](#3-quantization-bridging-floats-to-fhe)
4. [Programmable Bootstrapping — The Core Innovation](#4-programmable-bootstrapping--the-core-innovation)
5. [Concrete ML API — Complete Reference](#5-concrete-ml-api--complete-reference)
6. [Supported Models and Operators](#6-supported-models-and-operators)
7. [Usage in This FL Framework](#7-usage-in-this-fl-framework)
8. [FHE Circuit Compilation Deep Dive](#8-fhe-circuit-compilation-deep-dive)
9. [Performance Benchmarks](#9-performance-benchmarks)
10. [Troubleshooting](#10-troubleshooting)
11. [TFHE vs CKKS: Detailed Comparison](#11-tfhe-vs-ckks-detailed-comparison)

---

### 1. TFHE Scheme Deep Dive

#### 1.1 Origins

TFHE (Torus Fully Homomorphic Encryption) was developed by Chillotti, Gama, Georgieva, and Izabàchène (2016–2020). It is an evolution of the FHEW scheme (Ducas-Micciancio, 2015), radically improving bootstrapping speed from seconds to milliseconds per gate — making practical per-gate bootstrapping possible.

The key conceptual departure of TFHE from earlier FHE schemes is in *what it optimizes for*. CKKS and BFV optimize for batching efficiency: pack thousands of values into one ciphertext and process them all with a single polynomial operation. TFHE takes the opposite approach: it optimizes for **function generality on single values**. Rather than asking "how can we pack more values into less space?" TFHE asks "how can we evaluate arbitrary functions over encrypted values as efficiently as possible?" The answer was to redesign bootstrapping from an expensive recovery operation into a cheap, general-purpose computation primitive.

The key mathematical insight: **work over the torus $\mathbb{T} = \mathbb{R}/\mathbb{Z}$** (real numbers modulo 1) rather than integer rings. At first this seems like an abstract choice, but it has a concrete engineering payoff. The torus has a natural cyclic structure that aligns perfectly with the "test polynomial rotation" mechanism used inside bootstrapping. When a bootstrapping operation rotates a polynomial in a ring accumulator by an amount determined by the encrypted input value, the torus structure ensures that this rotation correspondence is exact — the output coefficient precisely encodes the desired function output. Integer rings require more complex rescaling and modulus management to achieve the same. Torus arithmetic enables a particularly efficient "external product" between TRLWE and TRGSW ciphertexts that powers fast bootstrapping.

This led to a fundamental inversion of the TFHE design philosophy relative to earlier FHE: **make bootstrapping cheap enough to run after every single operation**, and then build all computation out of bootstrapped operations. Instead of treating bootstrapping as a last resort when the noise budget is almost exhausted, TFHE bootstraps continuously, keeping every ciphertext permanently fresh. This is conceptually similar to how garbage-collected languages (Python, Java) handle memory: instead of carefully managing when to free memory, you just allocate freely and let the runtime handle cleanup. The engineering cost is similar too: you pay a constant overhead per operation, but gain the freedom to compute without counting.

#### 1.2 LWE Over the Torus

A **TLWE** (Torus LWE) ciphertext encrypting bit $b \in \{0, 1\}$ is:

$$\mathbf{ct} = (\mathbf{a}, b_{\text{encrypted}}) \in \mathbb{T}_{\mathbb{Z}}^n \times \mathbb{T}$$

$$b_{\text{encrypted}} = \langle \mathbf{a}, \mathbf{s} \rangle + e + \frac{b}{2} \in \mathbb{T}$$

where:
- $\mathbf{a} \in \mathbb{T}_{\mathbb{Z}}^n$ — uniformly random (integer approximations of torus elements)
- $\mathbf{s} \in \{0, 1\}^n$ — binary secret key
- $e \in \mathbb{T}$ — small Gaussian error ($\sigma \approx 3.29 \times 10^{-10}$)
- $n$ — LWE dimension (~500–1000 for 128-bit security)

**Decryption**: Phase of $\mathbf{ct}$ is $\phi = b_{\text{enc}} - \langle \mathbf{a}, \mathbf{s} \rangle = b/2 + e$. Round to nearest $\{0, 1/2\}$ → recover $b$.

#### 1.3 TRLWE — Ring Variant for Efficiency

Like RLWE (used in CKKS), TRLWE works in a polynomial ring for batch efficiency:

$$R_{\mathbb{T}} = \mathbb{T}[X] / (X^N + 1)$$

A TRLWE ciphertext encrypts a polynomial $\mu(X) \in R_{\mathbb{T}}$ whose coefficients encode $N$ bits (or a small message modulation). This allows $N$ LWE samples to be packed into one TRLWE ciphertext.

#### 1.4 TRGSW — The Gadget Ciphertext

TFHE introduces **TRGSW** (Ring GSW) ciphertexts — a different encryption format used for the "bootstrapping key". TRGSW ciphertexts are larger but support an operation called the **external product**:

$$\mathbf{C} \boxdot \mathbf{ct}_{\text{TRLWE}} = \mathbf{C} \cdot \text{decomp}(\mathbf{ct}) \approx \mathbf{ct}_{\text{result}}$$

This is the primitive used inside bootstrapping — multiplying a TRLWE ciphertext by an encrypted selector bit to perform a conditional operation.

#### 1.5 Gate Bootstrapping

A **gate bootstrapping** operation takes one or two bit ciphertexts, evaluates a boolean gate (AND, OR, MUX, etc.), and simultaneously refreshes the noise:

$$\text{Gate}(\mathbf{ct}_1, \mathbf{ct}_2) \rightarrow \mathbf{ct}_{\text{out}} \quad \text{with fresh noise}$$

The bootstrapping works by evaluating the gate truth table using a **test polynomial** in a TRLWE ACC (accumulator) that is "rotated" by the LWE ciphertext value — the output coefficient encodes the gate output. This rotation is implemented via the external product, which is why TRGSW bootstrapping keys are needed.

**Gate homomorphic operations**: Each gate takes ~10ms on CPU (2024). Multi-gate nand (fanout): can implement ANY boolean circuit with this primitive.

#### 1.6 TFHE Message Space

TFHE can encode small integers, not just bits. With $p$ message bits in a ciphertext modulo $2^p$ (message space of size $p$ bits):

| Message bits $p$ | Values encoded | Bootstrapping cost |
|-----------------|---------------|-------------------|
| 1 bit | $\{0, 1\}$ | 1 PBS |
| 2 bits | $\{0, 1, 2, 3\}$ | 1 PBS |
| 4 bits | $\{0, ..., 15\}$ | 1–2 PBS |
| 8 bits | $\{0, ..., 255\}$ (int8) | 2–4 PBS |

Larger message spaces use the same number of bootstrapping operations as long as the total message width fits within the ciphertext's "precision" (determined by LWE modulus and noise floor).

---

### 2. Concrete ML Architecture

#### 2.1 Compilation Pipeline

Concrete ML's most distinctive feature is its **compilation pipeline**: you write a standard scikit-learn or PyTorch model, and Concrete ML automatically transforms it into an equivalent FHE circuit. This transformation is non-trivial — it involves quantizing floating-point representations, exporting the model's computation graph to an intermediate format (ONNX), compiling that graph into a sequence of TFHE operations, and generating the cryptographic keys needed to execute those operations.

The pipeline is opaque to the user by design — you do not hand-write FHE circuits; you write normal Python ML code and the compiler handles the translation. This is analogous to how a GPU shader compiler takes a high-level GLSL program and emits optimized GPU machine code: you describe *what* to compute, and the toolchain figures out *how* to compute it under the constraints of the target hardware (in this case, TFHE's PBS-based arithmetic).

Each stage of the pipeline serves a specific purpose. Quantization maps float32 model weights and activations to integer ranges that fit within TFHE's integer message space. ONNX export captures the computational graph as a sequence of operators (matrix multiply, relu, add, etc.) in a hardware-agnostic format. The Concrete compiler then maps each operator to the appropriate FHE primitive: linear operators accumulate in integer space (no bootstrapping needed), while non-linear operators (activations, rounding) each become one Programmable Bootstrapping call. Key generation produces the cryptographic material needed for the resulting FHE circuit.

The overall pipeline looks like this:
User's sklearn / PyTorch model (float32)
           │
           ▼
  ┌─────────────────────────┐
  │  1. QUANTIZATION        │  float → int (n_bits precision)
  │     Post-Training       │  learned min/max → scale + zero_point
  └─────────────────────────┘
           │
           ▼
  ┌─────────────────────────┐
  │  2. ONNX EXPORT         │  model → ONNX computation graph
  │     via torch.onnx      │
  └─────────────────────────┘
           │
           ▼
  ┌─────────────────────────┐
  │  3. CONCRETE COMPILER   │  ONNX → Concrete IR → FHE circuit
  │     (Rust + MLIR)       │  operator fusion, PBS insertion
  └─────────────────────────┘
           │
           ▼
  ┌─────────────────────────┐
  │  4. KEY GENERATION      │  LWE + TRLWE + TRGSW keys
  │     circuit.keygen()    │  bootstrapping keys (large: ~1 GB for big models)
  └─────────────────────────┘
           │
           ▼
  ┌─────────────────────────┐
  │  5. FHE EXECUTION       │  encrypted inference
  │     circuit.encrypt()   │  gate-by-gate evaluation with PBS
  │     circuit.run()       │
  │     circuit.decrypt()   │
  └─────────────────────────┘
```

#### 2.2 Key Abstractions

**`concrete.ml.sklearn`**: Drop-in sklearn-compatible classifiers and regressors that can be compiled to FHE:
- `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`
- `XGBClassifier`, `SGDClassifier`, `LinearSVC`
- Each has `compile(X_calib)` and `predict()` / `predict_fhe()` methods

**`concrete.ml.torch`**: For custom PyTorch models:
- `compile_torch_model(model, X_calib, n_bits=8)` — full compilation
- `ConvertOpToHybrid` — convert specific layers to FHE, leave others in plaintext

**`fhe.Circuit`**: The compiled FHE circuit object:
- `circuit.keygen()` — generate LWE/TRLWE/bootstrapping keys
- `circuit.encrypt(x)` — produce LWE ciphertext
- `circuit.run(ct)` — evaluate FHE circuit
- `circuit.decrypt(ct_out)` — decode output

---

### 3. Quantization: Bridging Floats to FHE

#### 3.1 Why Quantization is Required

TFHE operates on **integers** (bits or small integers modulo $2^p$). This is not a limitation of the scheme so much as a reflection of its design target: TFHE was built for boolean circuits, not continuous arithmetic. Its bootstrapping mechanism evaluates look-up tables over discrete integer domains. Real-valued model parameters and activations must be converted to integers before FHE encryption. This conversion process is called **quantization**.

Quantization is conceptually analogous to converting a high-resolution photograph to a lower bit-depth image. You take a continuous measurement (a float32 gradient value from $-3.2$ to $+3.2$, say) and round it to the nearest value on a finite discrete grid (0 through 255, for 8-bit). The mapping is reversible — within the resolution of the grid — and the error introduced depends on the grid spacing: coarser grids (fewer bits) mean larger errors, finer grids (more bits) mean smaller errors but larger FHE circuits.

This is the central accuracy/efficiency trade-off of TFHE for ML models: quantization is the price of admission for using TFHE's powerful probabilistic bootstrapping machinery, and the `n_bits` parameter is the knob that controls how much you pay. For most practical classifiers on tabular data, 8-bit quantization introduces accuracy losses of 1–3%, which is often acceptable given the privacy and computational benefits.

Concrete ML uses **uniform affine quantization**:

$$q = \text{round}\!\left(\frac{x - x_{\min}}{x_{\max} - x_{\min}} \cdot (2^b - 1)\right)$$

Equivalently: $q = \text{clip}\!\left(\text{round}(x / S + Z),\, 0,\, 2^b - 1\right)$

where $S$ is the **scale** (the float range represented by one integer step) and $Z$ is the **zero point** (the integer corresponding to float 0.0).

**Dequantization**: $\hat{x} = S \cdot (q - Z)$

#### 3.2 The `n_bits` Parameter

`n_bits` controls quantization precision. Higher values reduce quantization error at the cost of larger FHE circuits:

| `n_bits` | Integer range | Max quantization error | Bootstrapping overhead |
|----------|--------------|----------------------|----------------------|
| 1 | {0, 1} | ~50% of range | Minimal |
| 4 | {0, ..., 15} | ~6% of range | Low |
| **8** | **{0, ..., 255}** | **~0.4% of range** | **Moderate (default)** |
| 12 | {0, ..., 4095} | ~0.02% of range | High |
| 16 | {0, ..., 65535} | ~0.002% of range | Very high |

**This framework uses `n_bits = 8`** — a common trade-off achieving <1% quantization error with manageable circuit size.

#### 3.3 Calibration Data

Quantization scales are computed from a **calibration dataset** passed to `compile(X_calib)`. The calibration data represents the expected input distribution:

```python
## Calibration data MUST be representative of real inference inputs
model.compile(X_train[:1000])   # Use training data as calibration
```

If calibration data is unrepresentative (e.g., only zeros), scale parameters will be wrong and quantized values will overflow → FHE produces garbage output.

#### 3.4 Accuracy Impact

Quantization introduces error by rounding float values to integer grid:

| Model type | Accuracy loss at `n_bits=8` | Notes |
|-----------|---------------------------|-------|
| Logistic Regression | ~0–0.5% | Linear, very tolerant |
| Small MLP (2 layers) | ~1–3% | Depends on activation precision |
| Deep MLP (>10 layers) | ~3–8% | Error compounds per layer |
| XGBoost | ~0–2% | Tree structure is already discrete |
| CNNs | ~2–5% | Activation and conv rounding |

**In this framework**: healthcare dataset with logistic regression → ~2.5% accuracy drop (87.3% baseline → 84.8% TFHE). This is primarily from the 8-bit quantization of the gradient updates and model weights at the boundary of the linear model.

---

### 4. Programmable Bootstrapping — The Core Innovation

#### 4.1 Standard vs Programmable Bootstrapping

To appreciate Programmable Bootstrapping (PBS), start with the problem it solves. In conventional FHE schemes, non-linear functions like ReLU, sigmoid, clipping, or maximum are extremely expensive. CKKS approximates them as polynomial series (requiring many multiplicative levels). Gate-based FHE evaluates them as bit-by-bit boolean circuits (requiring hundreds of AND gates, each one a bootstrapping operation at the gate output).

PBS collapses this cost to a single operation regardless of the function's complexity. The key insight is that the TFHE bootstrapping mechanism already involves evaluating a polynomial at a position determined by the encrypted input value. If you design that polynomial to encode the function $f$ you want to evaluate — not just the identity or a decryption function — then the bootstrapping computes $f(v)$ and refreshes the noise in one step.

Concretely: the bootstrapping procedure maintains a ring polynomial accumulator that encodes all possible function outputs. As the algorithm processes the bits of the encrypted input, it effectively "rotates" the accumulator to the position corresponding to the input value. When done, the output coefficient of the accumulator is the encrypted value of $f(\text{input})$. The rotation is done using the bootstrapping key (TRGSW encryptions of secret key bits), and the entire process is a sequence of "blind rotations" — rotations by encrypted amounts. Changing the function $f$ is as simple as changing the test polynomial loaded into the accumulator at the start — no structural change to the circuit is needed.

This makes PBS the swiss-army knife of the TFHE world: one bootstrap handles decryption-based refresh AND any arbitrary single-output function evaluation. The cost, ~10ms per PBS on a modern CPU, dominates the computation time of any Concrete ML model, so the total inference time is roughly linear in the number of PBS operations required by the model.

**Standard (gate) bootstrapping** (FHEW): refreshes noise, evaluates one boolean gate.

**Programmable Bootstrapping (PBS)** (TFHE): refreshes noise AND simultaneously evaluates ANY function $f: \{0, ..., 2^p - 1\} \rightarrow \{0, ..., 2^p - 1\}$ — essentially a look-up table (LUT) of any size.

The function $f$ is encoded into the **test polynomial** stored in the TRLWE accumulator. The bootstrapping operation "selects" the output coefficient by rotating the accumulator by the encrypted input value.

#### 4.2 Mathematical Mechanism

Given an LWE ciphertext $\mathbf{ct}$ encrypting value $v \in \{0, ..., N-1\}$:

1. **Test polynomial** $T(X) = \sum_{i=0}^{N-1} f(i) \cdot X^i$ — encodes the LUT
2. **Accumulator** $\mathbf{ACC}_0 = \text{TRLWE}(T(X))$ — encrypts the test polynomial
3. **External product loop**: for each bit $s_j$ of the bootstrapping key,
   $$\mathbf{ACC}_{j+1} = (1 - \mathbf{ct}_j) \cdot \mathbf{ACC}_j + \mathbf{ct}_j \cdot X^{a_j} \cdot \mathbf{ACC}_j$$
   This "rotates" the accumulator by $v \cdot B_g^j$ total after all $l$ iterations
4. **Sample extraction**: extract the 0-th coefficient from $\mathbf{ACC}_l$ → $\text{LWE}(f(v))$

The result is a fresh LWE ciphertext encrypting $f(v)$ with reduced noise.

#### 4.3 PBS Cost vs Gate-by-Gate

| Approach | Cost per elementary operation | Can evaluate f: {0→15}? |
|----------|------------------------------|-------------------------|
| Gate bootstrapping (bit at a time) | 10ms | 4 bits → 4 gates → 40ms |
| **Programmable bootstrapping** | **10ms per table lookup** | **Yes, in one PBS (10ms)** |

With PBS, any function over $p$-bit integers takes just **one bootstrapping** regardless of complexity. This is revolutionary because it means:
- ReLU: 1 PBS
- MaxPool: 1 PBS per comparison
- Integer quantized GeLU: 1 PBS  
- Any piecewise linear function: 1 PBS

#### 4.4 Impact on Neural Network Inference

Without PBS, activations like ReLU would require hundreds of boolean gates per neuron. With PBS, each neuron's activation = 1 PBS (~10ms). For a 128-neuron hidden layer: 128 PBS operations run in ~1.28 seconds (serial) or much faster with GPU parallelism.

**Concrete ML's compiler automatically inserts PBS at every non-linear operation** (activations, clipping, rounding). Linear operations (matrix multiply) are accumulated in plaintext integer space, then PBS is applied at the quantized output.

---

### 5. Concrete ML API — Complete Reference

#### 5.1 sklearn-Compatible Interface

```python
from concrete.ml.sklearn import SGDClassifier, LogisticRegression
from concrete.ml.sklearn import RandomForestClassifier, XGBClassifier
import numpy as np

## === Training (identical to sklearn) ===
model = SGDClassifier(
    n_bits=8,            # quantization bits — key parameter
    max_iter=200,
    random_state=42
)
model.fit(X_train, y_train)

## === Compile to FHE circuit ===
## Calibration data should be a representative sample
model.compile(X_train[:500])   # sets quantization scales

## === Plaintext (simulated) prediction — fast, no FHE ===
y_pred_plain = model.predict(X_test)
print(f"Plaintext accuracy: {(y_pred_plain == y_test).mean():.3f}")

## === FHE prediction (encrypted) — slow but private ===
y_pred_fhe = model.predict(X_test, fhe="execute")
print(f"FHE accuracy: {(y_pred_fhe == y_test).mean():.3f}")

## === Simulate FHE (fast check for accuracy without crypto overhead) ===
y_pred_sim = model.predict(X_test, fhe="simulate")
## fhe="simulate" uses quantized arithmetic but skips actual TFHE encryption
## Useful for calibrating n_bits before running full FHE
```

#### 5.2 Compilation Options

```python
from concrete.ml.sklearn import LogisticRegression

model = LogisticRegression(n_bits=8)
model.fit(X_train, y_train)

## Basic compilation
model.compile(X_calib)

## Compilation with options
circuit = model.compile(
    X_calib,
    configuration=Configuration(
        enable_unsafe_features=True,      # allow unsafe opts for benchmarking
        use_insecure_key_cache=True,      # cache keys for repeated testing
        insecure_key_cache_location="./key_cache",
        show_graph=True,                  # print MLIR graph
        show_mlir=False,                  # print MLIR IR
        show_optimizer=False,             # print optimizer steps
        dump_artifacts_on_unexpected_failures=True,
    )
)
## circuit = model.fhe_circuit  (accessible after compile)

## Get circuit statistics
print(model.fhe_circuit.statistics)
## → {n_inputs: 1, n_1b_gates: 12, n_2b_gates: 1, ...}
```

#### 5.3 Manual Key Generation and Encryption

```python
## After compilation, manage keys explicitly
circuit = model.fhe_circuit

## Generate FHE keys (LWE key + evaluation/bootstrapping keys)
circuit.keygen()
## Keys stored in circuit — this can take 1–5 minutes for large models

## Serialize keys for reuse
key_bytes = circuit.serialize_lwe_secret_key()
with open("fhe_key.bin", "wb") as f:
    f.write(key_bytes)

## Quantize input (convert float → int)
x_q = model.quantize_input(X_test[:1])
## x_q is numpy array of integers

## Encrypt (produces encrypted LWE ciphertext)
x_enc = circuit.encrypt(x_q)

## Evaluate (FHE inference — runs PBS on server)
y_enc = circuit.run(x_enc)

## Decrypt
y_q = circuit.decrypt(y_enc)

## Dequantize (int → float class label)
y_pred = model.dequantize_output(y_q)
```

#### 5.4 Hybrid Mode (Partial FHE)

For models with both FHE-compatible and FHE-incompatible operations:

```python
from concrete.ml.deployment import FHEModelServer, FHEModelClient, FHEModelDev

## Developer: compile and export
dev = FHEModelDev("./fhe_model", model)
dev.save()
## Saves: serialized model, client.zip, server.zip

## Client: encrypt inputs
from concrete.ml.deployment import FHEModelClient
client = FHEModelClient("./fhe_model/client.zip", key_dir="./keys")
client.generate_private_and_evaluation_keys()
evaluation_keys = client.get_serialized_evaluation_keys()

x_enc = client.quantize_encrypt_serialize(X_test[:1])

## Server: run inference on encrypted data
from concrete.ml.deployment import FHEModelServer  
server = FHEModelServer("./fhe_model/server.zip")
server.load()
y_enc = server.run(x_enc, evaluation_keys)

## Client: decrypt
y_pred = client.deserialize_decrypt_dequantize(y_enc)
```

#### 5.5 Custom PyTorch Model Compilation

```python
import torch
import torch.nn as nn
from concrete.ml.torch.compile import compile_torch_model

class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SmallMLP()
## ... train model ...

## Compile to FHE
fhe_circuit = compile_torch_model(
    model,
    torch_inputset=X_train_tensor[:500],   # calibration data
    n_bits=8,
    rounding_threshold_bits=6,             # additional rounding for speed
    p_error=0.01,                           # acceptable PBS error probability
)

fhe_circuit.keygen()
x_enc = fhe_circuit.encrypt(X_test_np[:1])
y_enc = fhe_circuit.run(x_enc)
y_pred = fhe_circuit.decrypt(y_enc)
```

---

### 6. Supported Models and Operators

#### 6.1 sklearn Models

| Model | FHE Support | Notes |
|-------|-------------|-------|
| `LogisticRegression` | ✅ Full | Best accuracy, low circuit depth |
| `LinearSVC` | ✅ Full | Linear, efficient |
| `SGDClassifier` | ✅ Full | Flexible loss functions |
| `DecisionTreeClassifier` | ✅ Full | Tree evaluation via PBS |
| `RandomForestClassifier` | ✅ Full | Multiple trees, higher overhead |
| `GradientBoostingClassifier` | ✅ Full | XGBoost-style |
| `XGBClassifier` | ✅ Full (via wrapping) | |
| `KNeighborsClassifier` | ⚠️ Partial | Distance computation expensive |
| `SVC` (kernel) | ❌ No | Kernel functions incompatible |
| Neural networks (sklearn) | ❌ No | Use torch path |

#### 6.2 PyTorch Operators Supported in FHE

| Operator | FHE Support | Notes |
|----------|-------------|-------|
| `Linear` / `Conv2d` | ✅ | As matrix multiply |
| `ReLU` | ✅ | Via PBS |
| `LeakyReLU` | ✅ | Via PBS |
| `Sigmoid` | ✅ | PBS with polynomial approximation |
| `Tanh` | ✅ | PBS with polynomial approximation |
| `GELU` | ✅ | Via PBS |
| `BatchNorm` | ⚠️ | Only inference-mode (constant fold) |
| `MaxPool` | ✅ | Via comparison PBS |
| `Softmax` | ⚠️ | Approximate; complex circuit |
| `Attention` | ❌ | Too deep for practical FHE yet |
| `Dropout` | N/A | Not used at inference |
| `Embedding` | ⚠️ | Large LUT |

#### 6.3 Precision Requirements

Each operator consumes "precision bits" from the total noise budget:

```
Input (8-bit) → Linear → Accumulation (wider precision) 
 → Quantize/Rescale → PBS → Output (8-bit)
```

If the accumulation overflows (too many inputs or too wide weights), quantization error increases. The `n_bits` parameter and `rounding_threshold_bits` control where rounding is inserted to prevent overflow.

---

### 7. Usage in This FL Framework

#### 7.1 The `he_concrete_tfhe` Mode

This framework uses Concrete ML for the `he_concrete_tfhe` mode. Unlike TenSEAL which encrypts gradient vectors for transport, Concrete ML compiles the **entire model to an FHE circuit** so that:
1. The client quantizes its local model weights to int8
2. Weights are encrypted as LWE ciphertexts
3. The server performs FHE-based weight aggregation
4. The aggregated encrypted model is sent back

#### 7.2 Model Definition and Training

In `core/model_builder.py`:

```python
from concrete.ml.sklearn import SGDClassifier

def build_concrete_model(n_bits=8):
    """Build a Concrete ML compatible classifier."""
    return SGDClassifier(
        n_bits=n_bits,
        loss='log_loss',        # logistic regression with SGD
        max_iter=100,
        random_state=42,
        learning_rate='optimal',
        alpha=0.01  # L2 regularization
    )
```

#### 7.3 Compile and Key Generation

```python
from concrete.ml.sklearn import SGDClassifier

def compile_concrete_model(model, X_calib):
    """Compile model to FHE circuit and generate keys."""
    # Compile
    model.compile(X_calib)
    circuit = model.fhe_circuit
    
    # Generate keys (LWE + evaluation keys)
    circuit.keygen()
    
    # The evaluation key (needed by server) is embedded in circuit
    # Serialize for storage
    with open("concrete_circuit.bin", "wb") as f:
        f.write(circuit.serialize())
    
    return circuit
```

#### 7.4 Encrypted Gradient Transport

```python
## client.py — TFHE mode

def encrypt_weights_concrete(model):
    """
    Quantize model weights to int8 and encrypt.
    Returns serialized encrypted ciphertext bytes.
    """
    circuit = model.fhe_circuit
    
    # Extract weight vector
    weights = model.coef_.flatten().astype(np.float32)
    
    # Quantize (float → int8)
    weights_q = model.quantize_input(weights.reshape(1, -1))
    
    # Encrypt as LWE ciphertext
    ct = circuit.encrypt(weights_q)
    
    return ct   # or serialize for transport
```

```python
## server.py — TFHE aggregation

def aggregate_tfhe(encrypted_weights_list, n_clients, circuit):
    """
    Aggregate encrypted int8 weights.
    TFHE doesn't support homomorphic addition directly on LWE ciphertexts
    without a full circuit — so we sum in quantized integer space.
    """
    # In this framework's implementation, the server receives 
    # already-quantized (but NOT encrypted) int8 weights for aggregation.
    # Full TFHE aggregation would require an FHE circuit for the summation.
    
    # Decode quantized weights (server has evaluation key, not secret key)
    sum_q = np.zeros_like(encrypted_weights_list[0])
    for w_q in encrypted_weights_list:
        sum_q = sum_q + w_q   # integer addition (no encryption here)
    
    avg_q = (sum_q / n_clients).astype(np.int8)
    return avg_q
```

> **Implementation note**: True TFHE aggregation of encrypted gradient values requires a custom FHE circuit for the summation step. This framework's `he_concrete_tfhe` mode uses quantized (int8) weight transport rather than ciphertext-level aggregation — providing the quantization-privacy trade-off without the full key-exchange complexity of multi-party TFHE aggregation. The primary benefit is compact, quantized gradient representation (17.8 MB vs 244.7 MB for CKKS).

#### 7.5 Data Flow for `he_concrete_tfhe`

```
CLIENT                              SERVER
  │                                   │
  │  [train locally]                   │
  │  weights: float32 (13 params)      │
  │                                   │
  │  quantize → int8 weights           │
  │  (17.8 MB/round — compressed)     │
  │  ─────── int8 weights ──────────► │
  │                                   │  sum_q = Σ w_q_i
  │                                   │  avg_q = sum_q / n
  │◄──────── avg_q int8 ──────────────│
  │                                   │
  │  dequantize → float weights        │
  │  update model                     │
```

The 17.8 MB vs 0.01 MB for baseline comes from transmitting quantized weight matrices for all layers — still much smaller than CKKS ciphertexts (244.7 MB).

---

### 8. FHE Circuit Compilation Deep Dive

#### 8.1 ONNX Graph Translation

Concrete ML exports PyTorch / sklearn models to ONNX (Open Neural Network Exchange), then processes the ONNX computation graph:

1. **Operator lowering**: Each ONNX operator (MatMul, Relu, etc.) → Concrete IR operator
2. **Precision propagation**: Determine required bit-width at each tensor
3. **PBS insertion**: Insert bootstrapping at each non-linear op (where input precision > 1 bit and output requires fresh noise)
4. **Tiling**: Large matrix operations split into tile-level operations that fit in TFHE's arithmetic width
5. **Code generation**: MLIR → low-level Concrete instruction stream

#### 8.2 Circuit Statistics

After compilation, inspect the FHE circuit cost:

```python
stats = model.fhe_circuit.statistics
print(stats)
## {
##   'n_inputs': 1,
##   'n_outputs': 1,
##   'n_1b_gates': 0,         # 1-bit boolean gates
##   'n_2b_gates': 0,         # 2-bit table lookups
##   'n_3b_gates': 0,         # 3-bit
##   'n_4b_gates': 0,         #  ...
##   'n_8b_gates': 1,         # 8-bit PBS operations  ← most expensive
##   'total_n_bootstrapings': 13,    # total PBS count
##   'p_error': 0.01,         # per-PBS error probability
##   'global_p_error': 0.12,  # probability ≥1 error in circuit
## }
```

**Inference time estimation**: `total_n_bootstrapings × ~10ms` (CPU) or `÷ parallelism_factor` (GPU/multi-core).

#### 8.3 p_error Parameter

Each PBS has a probability $p$ of producing an incorrect result (from residual noise exceeding the decoding threshold). The `p_error` parameter controls this:

| `p_error` | Circuit error probability (13 PBS) | Speed |
|-----------|-------------------------------------|-------|
| 0.001 | ~1.3% | Slowest (more precision needed) |
| 0.01 | ~12% | Default |
| 0.05 | ~49% | Faster |
| 0.1 | ~72% | Very fast but unreliable |

`global_p_error` accounts for compound failure: $p_{\text{global}} = 1 - (1 - p)^{n_{\text{PBS}}}$.

Choose `p_error` based on your reliability requirements. For healthcare classification, `p_error=0.001` is recommended for production.

#### 8.4 Key Sizes (Concrete ML / TFHE)

| Key | Approximate Size | Notes |
|-----|----------------|-------|
| LWE secret key | ~4 KB | Client holds only |
| Evaluation key (bootstrap) | ~300–600 MB | Server needs this |
| LWE public key | ~32 KB | For encryption |

The evaluation key (bootstrapping key) is large because it contains TRGSW encryptions of all $n_{\text{LWE}}$ bits of the secret key. This is sent from the client to the server once, then reused for all inferences.

---

### 9. Performance Benchmarks

#### 9.1 Single Inference Timing (CPU, Intel i9-12900K)

| Model | n_bits | PBS count | Inference time (FHE) | Plaintext speedup |
|-------|--------|-----------|--------------------|--------------------|
| Logistic Regression (13 features) | 8 | 13 | ~1.3s | 1000× |
| Logistic Regression (30 features) | 8 | 30 | ~3s | 1000× |
| Small MLP (2 layers, 64 neurons) | 8 | ~200 | ~20s | 5000× |
| Decision Tree (depth 3) | 8 | ~7 | ~0.7s | 500× |
| XGBoost (10 trees, depth 3) | 8 | ~70 | ~7s | 3000× |

#### 9.2 FL Round Timing (healthcare, 2 clients, Apple M1)

| Phase | Time |
|-------|------|
| Local training (plaintext) | ~40s |
| Quantize weights | ~0.01s |
| Serialize + transmit int8 (~17.8 MB) | ~1.5s (local) |
| Server: int8 aggregation | ~0.05ms |
| Serialize + send back | ~1.5s |
| Dequantize | ~0.01s |
| **Total crypto overhead** | **~5.3s** |
| **Total round time** | **~45.3s** |

#### 9.3 Bandwidth Comparison

| Mode | Weight format | Size per round |
|------|--------------|---------------|
| baseline | float32 | 0.01 MB |
| he_tenseal | CKKS ciphertext | 244.7 MB |
| **he_concrete_tfhe** | **int8 quantized** | **17.8 MB** |
| he_concrete_tfhe_zkp | int8 + ZKP proof | 17.8 MB + 0.2 KB |

The 17.8 MB for Concrete is 14× smaller than TenSEAL and 1780× larger than baseline. The size comes from int8 weights for all layers serialized together.

#### 9.4 GPU Acceleration (Concrete ML experimental)

Concrete ML has experimental GPU support via CUDA:

```python
from concrete.ml.sklearn import SGDClassifier

model = SGDClassifier(n_bits=8)
model.fit(X_train, y_train)

## GPU-accelerated compilation (experimental, requires CUDA)
model.compile(X_calib, device="cuda")

## GPU inference (all PBS computed in parallel)
y_pred = model.predict(X_test, fhe="execute", device="cuda")
```

GPU speedup: ~10–50× for PBS-heavy models (parallelism over multiple samples).

---

### 10. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `AccumulatorTooLargeError` | Dot product overflow in quantized arithmetic | Reduce `n_bits`, reduce hidden layer width, or use `rounding_threshold_bits` |
| `ValueError: Model not compiled` | Calling predict with `fhe="execute"` before `compile()` | Always call `model.compile(X_calib)` and then `circuit.keygen()` |
| Low FHE accuracy vs plaintext | `n_bits` too low → large quantization error | Increase `n_bits` (8→10→12); check calibration data representativeness |
| `global_p_error` > 0.5 | Too many PBS with loose `p_error` | Decrease `p_error` to 0.001; use fewer layers |
| Key generation takes >10 min | Large bootstrapping keys for big model | Normal for large models; cache keys with `use_insecure_key_cache=True` |
| Calibration warning: `extrapolation` | Test inputs outside calibration range | Expand calibration data; normalize inputs before compile |
| `ModuleNotFoundError: concrete` | Concrete ML not installed | `pip install concrete-ml` |
| `ImportError: libgomp` | Missing OpenMP runtime (Linux) | `apt install libgomp1` or use Docker image |
| FHE inference crashes (segfault) | Concrete compiler bug | Pin to concrete-ml==1.x.y ; report to Zama GitHub |

---

### 11. TFHE vs CKKS: Detailed Comparison

#### 11.1 Fundamental Differences

At the deepest level, TFHE and CKKS differ not just in performance characteristics but in their model of how encryption and computation interact. CKKS is built on the principle that encrypting many values together in a polynomial and operating on that polynomial is efficient. TFHE is built on the principle that bootstrapping is cheap enough to be the atomic unit of computation. These different foundations lead to systems that are complementary rather than competing: CKKS excels where TFHE struggles, and vice versa. The table below captures key dimensions of this difference, with the conceptual trade-offs discussed in detail in Section 11.4.

| Dimension | TFHE (Concrete ML) | CKKS (TenSEAL) |
|-----------|-------------------|----------------|
| **Core scheme** | Torus LWE + bootstrapping | RLWE polynomial ring |
| **Plaintext type** | Small integers (via quantization) | Approximate reals |
| **Arithmetic precision** | Quantized (lossy) | Near-float64 (lossless for ML) |
| **Non-linear functions** | Native via PBS | Polynomial approximation only |
| **Bootstrapping frequency** | Every non-linear op | Optional (leveled usually sufficient) |
| **Bootstrapping cost** | ~10ms per PBS (fast) | ~1–5s per bootstrap (slow) |
| **Batch size** | 1 sample at a time | N/2 = 4096 values in one ciphertext |
| **Ciphertext size** | ~4 KB per LWE sample | ~400 KB per CKKS ciphertext |
| **Total gradient size** | 17.8 MB (int8 all params) | 244.7 MB (CKKS all layers) |

#### 11.2 Security Comparison

Both schemes achieve **128-bit classical security** under the standard parameter settings recommended by homomorphicencryption.org.

| Security Property | TFHE | CKKS |
|------------------|------|------|
| Underlying hardness | LWE (Decision) | RLWE (Decision) |
| Quantum security | ✅ Post-quantum | ✅ Post-quantum |
| IND-CPA | ✅ | ✅ |
| IND-CCA | ❌ (standard) | ❌ (standard) |
| Commitment security | ❌ | ❌ |
| Parameter standard | HE.org / Zama | HE.org / SEAL |

> **Critical CKKS note**: CKKS is *not* IND-CPA secure in the traditional sense when the decryption result is revealed (Li & Micciancio, 2021 attack). The approximate decryption leaks information about the noise, which can be used to recover the secret key. TenSEAL addresses this by adding 40 bits of extra noise. This does NOT affect this framework's use case (gradient aggregation, not inference with many decryption queries).

#### 11.3 Use Case Suitability

| Use Case | TFHE (Concrete) | CKKS (TenSEAL) | Recommendation |
|----------|----------------|----------------|----------------|
| FL gradient aggregation (additions only) | ⚠️ Overkill | ✅ Ideal | TenSEAL |
| FL with server-side clipping | ✅ (PBS) | ❌ | Concrete |
| Encrypted inference (entire model) | ✅ | ⚠️ (polynomial approx. only) | Concrete |
| Integer models (XGBoost, trees) | ✅ | ⚠️ | Concrete |
| Statistical analysis (mean, variance) | ⚠️ (quantization) | ✅ | TenSEAL |
| Large model, many layers | ⚠️ (slow PBS) | ✅ (batch) | TenSEAL |
| Bandwidth-sensitive deployment | ✅ (smaller CT) | ❌ (large CT) | Concrete |
| Regulatory compliance (exact DP) | ❌ | ❌ | Neither (use DP mode) |

#### 11.4 The Core Conceptual Difference

Understanding the TFHE/Concrete vs CKKS/TenSEAL distinction requires going beyond the benchmark numbers to the underlying design philosophies.

**CKKS is a vector-oriented, approximate-arithmetic scheme.** It treats encrypted data as a vector of real floating-point numbers, encrypted in bulk into a single polynomial ciphertext. All homomorphic operations are linear algebra — polynomial addition and multiplication — applied to the entire vector simultaneously. CKKS is naturally suited to gradient aggregation because gradient aggregation *is* linear algebra: sum the gradient vectors, divide by the count. CKKS does this in one polynomial addition (one ring operation per layer), with precision that exceeds float32 and latency measured in milliseconds. Its limitation is equally fundamental: CKKS cannot natively evaluate non-linear functions on ciphertexts. Any non-linear function must be approximated by a polynomial, which requires multiplicative depth, which consumes the modulus chain. For shallow polynomials (degree 3–5), this is fine. For arbitrary functions like ReLU or argmax, polynomial approximation either requires impractical depth or introduces significant error.

**TFHE is a gate-oriented, discrete-computation scheme.** It treats encrypted values as discrete integers processed one at a time. Its foundational operation — Programmable Bootstrapping — evaluates any function over a small integer domain in a single operation. This makes TFHE naturally suited to model inference: linear layer outputs are accumulated in quantized integer space, and at each non-linear layer, PBS evaluates the activation function in one step. TFHE pays for this flexibility with quantization error (converting floats to integers loses precision) and with per-value ciphertext overhead (no free batching across values). It is best understood not as "CKKS with less precision" but as a fundamentally different tool: a secure general-purpose processor for integer circuits, rather than a secure SIMD co-processor for floating-point linear algebra.

The practical consequence in this FL framework: CKKS/TenSEAL is the right choice when the FL aggregation server's role is limited to summing and averaging gradient updates — operations that are linear and well-matched to CKKS's strengths. TFHE/Concrete is the right choice when the server needs to apply non-linear operations (clipping, anomaly detection, comparison) to gradient updates, or when the entire inference pipeline (not just aggregation) needs to run in the encrypted domain.

#### 11.5 Decision Guide

For **FL gradient aggregation** as a primary use case — where clients train locally and the server simply sums encrypted updates and returns the average — **CKKS/TenSEAL** (`he_tenseal`) is the right default choice. It requires no accuracy compromise, no quantization step, and the aggregation computation (addition + scalar multiply) is perfectly matched to CKKS's capabilities.

If you need **server-side non-linear aggregation** — such as geometric median, coordinate-wise clipping, or any comparison-based robust aggregation — switch to **TFHE/Concrete** (`he_concrete_tfhe`). Only TFHE can evaluate these functions on encrypted updates without revealing the gradients to the server.

If **bandwidth** is a primary constraint — for example in mobile FL, IoT scenarios, or cross-datacenter FL with expensive links — **TFHE/Concrete** produces ciphertexts roughly 14x smaller than CKKS for this framework's model size. This advantage grows when slot underutilization is high (small models relative to CKKS polynomial size).

If **maximum accuracy** is required with zero tolerance for quantization-induced error — clinical applications where 2.5% accuracy loss is unacceptable, or models where 8-bit quantization causes significant degradation — **CKKS/TenSEAL** is the only FHE option. CKKS's only accuracy loss is rounding at the $10^{-12}$ level, which is negligible for any practical ML application.

For **full encrypted model inference** (not just aggregation, but running predictions on encrypted patient data without the server seeing the inputs) — **TFHE/Concrete** is designed for this. Concrete ML's compilation pipeline can convert an entire sklearn or PyTorch model into an FHE circuit that runs inference on encrypted inputs end-to-end.

Both schemes can be combined with **ZKP proofs** for integrity guarantees. The `he_tenseal_zkp` mode provides higher accuracy; the `he_concrete_tfhe_zkp` mode provides smaller ciphertext sizes. The ZKP overhead is similar in both cases. See [ZKP.md](ZKP.md) for details on the proof generation and verification flow.

---

> **Related documentation**:
> - [FHE.md](FHE.md) — Comprehensive FHE theory and scheme comparison
> - [FHE.md](FHE.md) — TenSEAL / CKKS deep dive
> - [README.md](README.md) — All 10 FL modes benchmarked
> - [Zama Concrete ML docs](https://docs.zama.ai/concrete-ml) — official reference

---

## HE Environment Variables Reference

All HE behaviour is controlled by environment variables so configurations can be changed without editing code or re-generating keys. Variables apply to both TenSEAL (CKKS) and Concrete TFHE backends unless noted.

### Quick reference

| Variable | Default | Values | Applies to |
|----------|---------|--------|-----------|
| `FL_ENCRYPT_LAYERS` | `ALL` | CSV layer names or `ALL` | TenSEAL |
| `FL_CONCRETE_TFHE_BIT_WIDTH` | `14` | Integer 2–16 | Concrete TFHE only |
| `FL_CONCRETE_TFHE_ADAPTIVE_QUANT` | `0` | `0`, `1` | Concrete TFHE only |

---

### `FL_ENCRYPT_LAYERS`

**Default:** `ALL`
**Values:** Comma-separated layer names, or `ALL`

Determines which model layers are encrypted before upload to the server.

| Value | Privacy | Bandwidth | Latency |
|-------|---------|-----------|---------|
| `ALL` (default) | Full gradient privacy — every layer is ciphertext | Maximum (all layers expanded by HE overhead) | Maximum encryption time |
| `model.0.weight,model.0.bias` | First layer encrypted; remaining layers in plaintext | Expansion only for those layers | Selective — just the first layer |
| Single layer e.g. `model.0.weight` | Weight matrix encrypted; bias in plaintext | Smallest encrypted payload | Fastest partial encryption |

> **Security note**: Encrypting only some layers leaks the plaintext gradient values of the unencrypted layers to the server. This is acceptable when those layers carry low information (e.g., scalar bias terms) but is not full gradient confidentiality. Use `ALL` for complete protection.

```bash
## Default: encrypt everything (full privacy, maximum bandwidth)
export FL_ENCRYPT_LAYERS=ALL

## First dense layer only: the other layers are sent in plaintext
export FL_ENCRYPT_LAYERS=model.0.weight,model.0.bias

## Encrypt only the weight matrix, skip bias
export FL_ENCRYPT_LAYERS=model.0.weight
```

**Impact on results:**
- TenSEAL: each encrypted layer adds its CKKS ciphertext expansion to the upload, so encrypting fewer layers cuts bandwidth roughly in proportion to their share of the parameters. Names that are not layers of the model are an error, so a misspelled layer can't silently leave the model in plaintext.
- Concrete TFHE: smaller ciphertexts (~17 MB per layer), but encryption time scales linearly with the number of encrypted layers.
- Accuracy: unaffected — HE encryption is exact (CKKS) or lossless at the given bit width (TFHE).

---

### `FL_CONCRETE_TFHE_BIT_WIDTH`

**Default:** `14`
**Values:** Integer 2–16

Quantization bit width used when converting floating-point gradient weights to integers for TFHE encryption. Applies to the `he_concrete_tfhe` and `he_concrete_tfhe_zkp` modes.

| Bit width | Accuracy impact | Ciphertext size | Encryption time |
|-----------|----------------|----------------|----------------|
| `8` | ~2–3% accuracy drop vs. baseline | Smallest | Fastest |
| `14` (default) | Negligible (<0.1% drop) | Moderate | Moderate |
| `16` | Lossless (within float32 precision) | Largest | Slowest |

The quantization rounding error scales as approximately $2^{-b}$ where $b$ is the bit width. At 14 bits the error is $\approx 6 \times 10^{-5}$, below the noise floor of typical FL gradient updates.

```bash
## Fast testing — acceptable accuracy loss for development
export FL_CONCRETE_TFHE_BIT_WIDTH=8

## Default — good accuracy / speed balance (recommended)
export FL_CONCRETE_TFHE_BIT_WIDTH=14

## Maximum accuracy — production medical / finance use cases
export FL_CONCRETE_TFHE_BIT_WIDTH=16
```

---

### `FL_CONCRETE_TFHE_ADAPTIVE_QUANT`

**Default:** `0`
**Values:** `0` (off) or `1` (on)

Toggles adaptive quantization in Concrete ML. When enabled, the quantization scale is chosen per-layer based on the observed value distribution, rather than using a single global scale across all layers.

| Value | Behaviour | When to use |
|-------|-----------|-------------|
| `0` (default) | Fixed global scale across all layers | Consistent, reproducible; best when layer weight magnitudes are similar |
| `1` | Per-layer scale fitted to distribution | Recovers ~0.5–1% accuracy on models with widely varying weight magnitudes across layers; adds ~5–10% overhead |

```bash
## Disabled (default)
export FL_CONCRETE_TFHE_ADAPTIVE_QUANT=0

## Enable for heterogeneous models where layer magnitudes differ significantly
export FL_CONCRETE_TFHE_ADAPTIVE_QUANT=1
```
