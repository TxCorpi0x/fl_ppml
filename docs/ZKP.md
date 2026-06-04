# Zero-Knowledge Proofs — Deep Conceptual Guide

> **Navigation**: [README.md](README.md) | [ZKP.md](ZKP.md) | [ZKP.md](ZKP.md) | [FHE.md](FHE.md)

## Table of Contents

1. [What is a Zero-Knowledge Proof?](#1-what-is-a-zero-knowledge-proof)
2. [The Three Defining Properties](#2-the-three-defining-properties)
3. [Historical Context and Intellectual Significance](#3-historical-context-and-intellectual-significance)
4. [Mathematical Foundations](#4-mathematical-foundations)
5. [Proof System Taxonomy](#5-proof-system-taxonomy)
6. [ZKP in Machine Learning](#6-zkp-in-machine-learning)
7. [ZKP in Federated Learning](#7-zkp-in-federated-learning)
8. [Pedersen Commitments — Deep Dive](#8-pedersen-commitments--deep-dive)
9. [gnark and Groth16 zk-SNARKs — Deep Dive](#9-gnark-and-groth16-zk-snarks--deep-dive)
10. [Pedersen vs Groth16: A Conceptual Comparison](#10-pedersen-vs-groth16-a-conceptual-comparison)
11. [Security Model and Limitations](#11-security-model-and-limitations)
12. [Further Reading](#12-further-reading)

---

## 1. What is a Zero-Knowledge Proof?

A **zero-knowledge proof** (ZKP) is a cryptographic protocol between two parties — a *prover* and a *verifier* — in which the prover convinces the verifier that a certain statement is true, while the verifier learns nothing beyond the bare fact of the statement's truth. No evidence, no witnesses, no intermediate values, no auxiliary information leaks out of the proof. The verifier is convinced, but enlightened by nothing.

To understand why this is non-trivial, consider the contrast with conventional proofs. A mathematical proof of a theorem reveals the *argument* — the chain of logical steps that produces the conclusion. Anyone who reads it gains understanding. A court witness who proves they were at a particular location at a particular time typically reveals that location to the court, to the opposing counsel, and to the public record. In ordinary epistemology, knowledge flows alongside proof: you prove something by sharing the evidence that makes it true, and the evidence teaches something.

Zero-knowledge proofs break this coupling. The prover possesses a *witness* — a private piece of information that makes the statement true — and generates a proof that cryptographically commits to the statement's validity without transmitting, hinting at, or statistically leaking any information about the witness itself. The verifier can check the proof efficiently and conclude, with overwhelming probability, that the statement is true and that the prover genuinely knows a valid witness — yet learns nothing about what that witness is.

The canonical example used to build the intuition is the **Ali Baba cave** (Quisquater et al., 1989). Imagine a circular cave with a single entrance and a door in the middle guarded by a password. Peggy (the prover) claims to know the password; Victor (the verifier) wants to be convinced without learning the password himself. They arrange a protocol: Victor waits at the entrance while Peggy walks into the cave, randomly choosing to enter from the left or right path. Victor then shouts either "come out from the left" or "come out from the right." If Peggy knows the password, she can always comply — she uses the door when necessary. If she does not know the password, she can only comply if Victor's instruction matches the path she originally entered. By repeating this many times, the probability that Peggy consistently guesses correctly by luck becomes negligible, while Victor learns nothing about the password beyond the fact that Peggy knows it.

This cave analogy captures the core intuition: the prover demonstrates knowledge by correctly responding to challenges that would be impossible to consistently meet without the private knowledge, while the challenges and responses themselves contain no information about what that knowledge is.

---

## 2. The Three Defining Properties

Every zero-knowledge proof system is formally characterized by three properties. These are not engineering goals but mathematical definitions that must be precisely satisfied for a protocol to constitute a genuine zero-knowledge proof.

### 2.1 Completeness

*If the statement is true and both parties follow the protocol honestly, the verifier accepts the proof with probability 1 (or overwhelming probability close to 1).*

Completeness is the correctness requirement. It ensures that an honest prover who genuinely knows the witness can always generate a proof that the verifier accepts. Without completeness, a proof system would be useless — even legitimate provers could fail to convince verifiers. A system is *perfectly complete* if acceptance probability is exactly 1, and *statistically complete* if it is $1 - \text{negl}(\lambda)$ for a security parameter $\lambda$.

### 2.2 Soundness

*If the statement is false, no cheating prover (however computationally powerful) can convince the honest verifier to accept, except with negligible probability.*

Soundness is the security requirement. It ensures the proof system cannot be gamed — a prover who does not actually know a valid witness cannot fabricate a convincing proof. The **soundness error** $\delta$ is the probability that an invalid proof passes verification. For a proof system to be useful, $\delta$ must be negligibly small (typically $2^{-128}$).

**Knowledge soundness** is a stronger variant: not only must the statement be true, but there must exist an *extractor* algorithm that, given black-box oracle access to a cheating prover, can efficiently extract a valid witness. If the prover convinces the verifier, the prover must "know" a witness in a computationally meaningful sense. This stronger property is required for zk-SNARKs and is what makes them useful for proving program execution rather than just the existence of a solution.

### 2.3 Zero-Knowledge

*The verifier learns nothing from the proof beyond the truth of the statement — formally, there exists a polynomial-time simulator that can generate transcripts indistinguishable from real proof transcripts without access to the witness.*

The zero-knowledge property is defined via the **simulator paradigm**. Consider a hypothetical "fake" prover that does not know the witness but has access to a "magic" ability: it can see the verifier's random challenges before sending its messages. Using this ability, the simulator can generate proof transcripts that are computationally (or statistically, or perfectly) indistinguishable from real ones. Since the simulator produces these transcripts without any witness, and they look identical to real proofs, it follows that the real proofs leak nothing about the witness — any information the verifier could extract from a real proof, it could also produce itself via the simulator.

There are graded levels of zero-knowledge:
- **Perfect zero-knowledge**: the simulated transcripts are identically distributed to real ones
- **Statistical zero-knowledge**: the distributions are statistically indistinguishable (differ by a negligible amount in total variation distance)
- **Computational zero-knowledge**: the distributions are computationally indistinguishable — no polynomial-time adversary can distinguish them with non-negligible advantage

For practical cryptographic applications, computational zero-knowledge under standard hardness assumptions is sufficient.

---

## 3. Historical Context and Intellectual Significance

Zero-knowledge proofs were introduced by Goldwasser, Micali, and Rackoff in their 1985 paper "The Knowledge Complexity of Interactive Proof Systems," which won the Gödel Prize in 1993 and the ACM Turing Award in 2012. The paper introduced not only zero-knowledge proofs but the complexity classes IP (Interactive Proofs) and, implicitly, the notion of "knowledge complexity" — a quantitative measure of how much knowledge a proof system reveals. The result that knowledge complexity zero is achievable — that some statements can be proven while revealing zero additional knowledge — was a foundational surprise.

The theoretical importance of zero-knowledge proofs extends well beyond their practical applications. The subsequent work of Gmali, Goldreich, Wigderson (1987) showed that every language in **NP** has a zero-knowledge proof system — an astonishing result meaning that any mathematical statement whose correctness can be efficiently verified can also be proven in zero-knowledge. This result, achieved by showing how to prove graph 3-colorability in zero-knowledge (and reducing all NP problems to it), established that zero-knowledge proofs are not a niche curiosity but a foundational tool of computational complexity theory.

For practical cryptography, the critical development was the **Fiat-Shamir heuristic** (1986), which showed how to convert interactive zero-knowledge protocols into non-interactive ones by replacing the verifier's random challenges with the output of a cryptographic hash function. Non-interactivity is essential for public deployment: the prover generates a single self-contained proof object, sends it to any number of verifiers, and each verifier checks it independently without any back-and-forth protocol. This transformation made zero-knowledge proofs logistically practical.

The modern era of zero-knowledge proof deployment began in earnest around 2010–2016, driven by the invention of **zk-SNARKs** (Succinct Non-interactive Arguments of Knowledge) that achieved extremely short proofs and constant-time verification. The landmark theoretical work by Groth (2010), Bitansky et al. (2012), and ultimately Groth (2016) produced the Groth16 protocol still in widespread production use today. Simultaneously, the Ethereum community's interest in blockchain-based computation created a powerful practical incentive for efficient ZKP systems, leading to an explosion of library development, protocol research, and production deployment that continues today.

What makes this history striking is that zero-knowledge proofs went from being a purely theoretical curiosity — a definition in a complexity theory paper — to a deployed production technology in major blockchain systems, federated learning frameworks, and privacy-preserving computation platforms within roughly 30 years. The pace of development accelerated dramatically once practical efficiency was achieved.

---

## 4. Mathematical Foundations

### 4.1 Languages, Witnesses, and NP

The formal setting for zero-knowledge proofs is the theory of computational complexity, specifically the class NP. An **NP language** $L$ is a set of strings such that membership in $L$ can be *verified* efficiently (in polynomial time) given a short *witness*. Formally:

$$L \in \text{NP} \iff \exists \text{ poly-time verifier } V \text{ s.t. } x \in L \iff \exists w : |w| \leq \text{poly}(|x|) \land V(x, w) = 1$$

In this framework, $x$ is the **statement** (also called the instance or public input) and $w$ is the **witness** (also called the secret input or private input). The zero-knowledge proof allows the prover (who knows $w$) to convince the verifier (who knows only $x$) that $x \in L$, while revealing nothing about $w$.

For ZKP in machine learning, the statement might be: "There exists a gradient vector $w$ whose $\ell_2$ norm does not exceed the bound $B$ and whose MiMC hash equals the committed value $C$." The witness is the actual gradient vector. The verifier checks the proof commits to a valid gradient without learning the gradient values.

### 4.2 Commitment Schemes

A **commitment scheme** is a fundamental building block of zero-knowledge proof systems. It allows a prover to "commit" to a value — essentially locking it in a sealed envelope — while keeping the value hidden from the verifier. Later, the prover can "open" the commitment to reveal the value, and the verifier can check that the revealed value matches the original commitment.

A commitment scheme $\text{Commit}(m, r) \to C$ takes a message $m$ and a random blinding factor $r$, and produces a commitment $C$. It must satisfy two properties:

**Hiding**: The commitment $C$ reveals no information about $m$. Even an adversary who sees $C$ and knows the distribution of possible messages cannot determine $m$. This is analogous to the sealed envelope: you cannot see through it.

**Binding**: Once the commitment $C$ is published, the committer cannot "change their mind" — they cannot find a different value $m'$ and blinding factor $r'$ such that $\text{Commit}(m', r') = C$. The commitment binds the prover to their original choice. This is analogous to the envelope being tamper-evident: breaking the seal would be detectable.

The tension between these properties is fundamental. Perfect hiding requires that $C$ be independent of $m$, but perfect binding requires that $C$ uniquely determines $m$. These two conditions cannot simultaneously hold with perfect security (this impossibility is proven by information-theoretic arguments), so practical commitment schemes achieve one perfectly and the other computationally — that is, secure against polynomial-time adversaries under hardness assumptions.

### 4.3 Arithmetic Circuits and R1CS

The most practical realization of NP computations for ZKP purposes uses **arithmetic circuits** — directed acyclic graphs where wires carry values from a finite field $\mathbb{F}_p$, and gates compute addition and multiplication over that field. Any computation — including neural network inference, hash functions, norm computations — can be expressed as an arithmetic circuit over an appropriate field.

The **Rank-1 Constraint System (R1CS)** is an equivalent formulation that is particularly amenable to zk-SNARK construction. An R1CS instance consists of three matrices $A, B, C \in \mathbb{F}_p^{m \times n}$. A witness vector $\mathbf{z} \in \mathbb{F}_p^n$ satisfies the R1CS if:

$$A\mathbf{z} \circ B\mathbf{z} = C\mathbf{z}$$

where $\circ$ denotes the Hadamard (componentwise) product. Each row of this matrix equation represents a single multiplication gate constraint of the form $(a_i \cdot \mathbf{z})(b_i \cdot \mathbf{z}) = c_i \cdot \mathbf{z}$. The total number of rows (constraints) is the "size" of the circuit and directly determines the computational cost of proving. In this framework, the gradient norm-bounding circuit produces approximately 32,000 R1CS constraints for 100 gradient coordinates.

### 4.4 Bilinear Pairings and Elliptic Curves

Modern zk-SNARKs like Groth16 are built on **bilinear pairings** — a special algebraic structure on elliptic curves. A bilinear pairing is a map $e: \mathbb{G}_1 \times \mathbb{G}_2 \to \mathbb{G}_T$ between three groups of prime order $p$, satisfying:

$$e(aP, bQ) = e(P, Q)^{ab} \quad \forall P \in \mathbb{G}_1, Q \in \mathbb{G}_2, a, b \in \mathbb{Z}_p$$

This bilinearity is the algebraic miracle that makes constant-size proofs possible. A prover can commit to a polynomial of arbitrary degree using a single group element, and verifiers can check polynomial identities using pairing equations without learning the polynomial itself. The **BN254** elliptic curve (the Barreto-Naehrig curve at the 254-bit prime), used by gnark in this framework, provides approximately 128 bits of classical security while admitting efficient pairing computation.

The security of pairing-based schemes rests on assumptions like the **Computational Diffie-Hellman** (CDH) assumption in $\mathbb{G}_1$ and $\mathbb{G}_2$, and the **Bilinear Diffie-Hellman** (BDH) assumption in the pairing group. These are believed hard classically but are not post-quantum secure — a sufficiently large quantum computer running Shor's algorithm could break them.

### 4.5 ZK-Friendly Hash Functions

Standard cryptographic hash functions (SHA-256, SHA-3, Blake2) are designed for efficient computation in software. However, inside an arithmetic circuit over $\mathbb{F}_p$, bitwise operations (XOR, AND, shifts) are extremely expensive — each bit operation may cost dozens of field multiplications. A 256-bit SHA-256 computation may require 20,000–50,000 R1CS constraints, making it impractical as a commitment function inside a ZKP circuit.

**ZK-friendly hash functions** are designed specifically for efficient representation as arithmetic circuits. **MiMC** (Minimal Multiplicative Complexity, Albrecht et al., 2016) hashes data using a series of field multiplications and additions, avoiding bitwise operations entirely. A single MiMC round computes $(x + c_i)^3$ over the field, and 110–220 such rounds provide cryptographic security. The total cost is approximately 300–500 R1CS constraints per MiMC invocation — roughly 100 times fewer than SHA-256.

MiMC is the commitment hash used in this framework's gnark circuit. For each sampled gradient coordinate, the coordinate value is quantized to a field element and passed through MiMC, building up a hash chain. The final hash output is the public commitment value $C$ included in the proof. Verifying that the prover knows the preimage of $C$ establishes that they are committing to a specific, fixed gradient vector.

---

## 5. Proof System Taxonomy

Understanding where Groth16 and Pedersen commitments fit requires a map of the proof system landscape.

### 5.1 Interactive Proof Protocols (Sigma Protocols)

The earliest and simplest zero-knowledge proof protocols are **sigma protocols** — three-message interactive protocols characterized by the $\Sigma$-shaped flow of messages:

1. **Commitment** ($a$): The prover generates a random nonce, applies a function to it, and sends the result to the verifier
2. **Challenge** ($e$): The verifier sends a uniformly random challenge from a challenge space
3. **Response** ($z$): The prover computes a response using both the nonce and their private witness

The classic example is the **Schnorr identification protocol**, which proves knowledge of a discrete logarithm. If a prover knows $x$ such that $g^x = y$ (where $g$ is a group generator and $y$ is the public value), they choose random $r$, send $a = g^r$, receive challenge $e$, and respond with $z = r + xe$. A verifier checks $g^z = a \cdot y^e$. An honest prover always passes; a cheating prover without knowledge of $x$ can only pass if they predicted $e$ before choosing $a$, which they cannot do if $e$ is truly random and independent of $a$.

Sigma protocols are efficient and elegant but are *interactive* — they require the verifier to be online and participating during the proof. This makes them unsuitable for asynchronous or one-to-many proof scenarios.

### 5.2 The Fiat-Shamir Transform

The **Fiat-Shamir heuristic** (1986) converts any sigma protocol into a **non-interactive** zero-knowledge proof by replacing the verifier's random challenge with the output of a cryptographic hash function applied to all prior messages. Instead of waiting for a random challenge from the verifier, the prover computes:

$$e = H(a, x)$$

where $H$ is a hash function (modeled as a random oracle), $a$ is the prover's commitment message, and $x$ is the statement being proven. This produces a self-contained proof $\pi = (a, z)$ that anyone can verify by recomputing $e = H(a, x)$ and checking the response equation.

The security of Fiat-Shamir proofs relies on the **random oracle model** — a theoretical idealization in which the hash function outputs truly random values for every distinct input. In practice, SHA-256 or BLAKE2 are used and the random oracle assumption is accepted as a pragmatic approximation. Fiat-Shamir makes sigma protocols non-interactive and widely deployable, and is the basis for many signature schemes (Schnorr signatures) and modern proof systems.

### 5.3 zk-SNARKs (Succinct Non-interactive Arguments of Knowledge)

**zk-SNARKs** are proof systems satisfying demanding efficiency requirements simultaneously:

- **Succinct**: The proof size is sublinear in the witness size — typically $O(1)$ or $O(\log n)$ field elements, regardless of the circuit size $n$
- **Non-interactive**: The proof is a single message requiring no interaction
- **Argument**: Soundness holds computationally (against bounded adversaries), not information-theoretically
- **Knowledge**: The prover must "know" a witness — knowledge extractability holds

A zk-SNARK produced by Groth16 for a circuit with $10^6$ constraints consists of exactly 3 elliptic curve points (about 192 bytes), and verification requires exactly 3 pairing operations regardless of circuit size. This constant-size, constant-time property is what makes SNARKs so valuable for applications where verification is a bottleneck.

### 5.4 zk-STARKs and Alternative Systems

**zk-STARKs** (Scalable Transparent ARguments of Knowledge) avoid pairings entirely, using hash functions and algebraic intermediate representations. Their key advantages are transparency (no trusted setup required) and post-quantum security (security based only on collision resistance of hash functions). Their disadvantages are larger proof sizes (tens of kilobytes versus hundreds of bytes) and slower verification. For this framework's FL use case, Groth16's smaller proofs and faster verification are preferable; STARKs would be preferable in a setting where trusted setup is infeasible and post-quantum security is required.

Other notable proof systems include **PLONK** (a universal and updatable SRS system), **Bulletproofs** (no trusted setup, linear verification), and **Nova** (recursive folding for incremental computation). The ZKP field is rapidly evolving, with new systems achieving better trade-offs in proof size, prover time, verifier time, and trust assumptions.

---

## 6. ZKP in Machine Learning

Zero-knowledge proofs appear in machine learning in several distinct roles, each addressing a different aspect of the trust gap between data owners, model trainers, and model users.

### 6.1 Verifiable Inference

The **verifiable inference** problem asks: given a public model and a public input, can a model provider prove that the declared output is the genuine output of running the model on the input — without the verifier re-running the entire model? This is relevant when inference is computationally expensive (large language models, high-resolution image classifiers) and users want to verify results without redundant computation. A ZKP proof of model execution provides cryptographic assurance that the server ran the claimed model, not a simpler or biased alternative.

Verifiable inference is technically challenging because neural network forward passes involve non-linear operations (ReLU, softmax, normalization) that are expensive to encode in arithmetic circuits. Recent work (zkml, EZKL, Gizatech) has demonstrated zk-SNARKs for ResNet-class models, though proving times remain high (minutes to hours for large models).

### 6.2 Verifiable Training

**Verifiable training** proofs convince a verifier that a model was trained on a specific dataset, following a specific training procedure, for a certain number of steps. This is relevant for regulatory compliance (a medical AI must be trained on approved data), for model auditing (proving a model was not fine-tuned on malicious data), and for federated learning (proving that local training was performed faithfully).

Verifiable training is substantially harder than verifiable inference because training involves gradient descent over thousands of steps, resulting in circuits with billions of constraints — currently at the frontier of ZKP research.

### 6.3 Privacy-Preserving Training Certificates

A more tractable variant is the **training certificate** approach: rather than proving the entire training procedure, prove structural properties of the model update — that the update's norm is bounded, that it is consistent with a committed input distribution, or that it does not contain outliers. This selective approach targets the most security-relevant properties while remaining computationally feasible. This is precisely the approach taken by this framework's gnark ZKP implementation.

### 6.4 Membership Proof and Exclusion Proofs

Zero-knowledge proofs can establish that a data point satisfies some predicate (is in a certain range, belongs to a certain category) without revealing the data point itself. In ML contexts, this enables clients to prove their training data meets quality or regulatory criteria without exposing private training records. Such proofs are beginning to appear in medical AI contexts where training data provenance is subject to audit requirements.

---

## 7. ZKP in Federated Learning

### 7.1 The Integrity Problem in Federated Learning

Federated learning distributes model training across many clients, each of which trains locally and contributes gradient updates. The aggregation server combines these updates (typically via FedAvg) to produce a global model. This architecture creates a fundamental trust asymmetry: the server must aggregate contributions from clients it cannot directly observe, and any single client can unilaterally undermine the system.

**Gradient poisoning** is the primary attack vector. A malicious client — one of the participating clients controlled by an adversary — can submit a gradient update that is not the product of honest training. It might submit a gradient crafted to steer the global model toward a backdoor behavior (triggering misclassification on inputs with a specific pattern), to reduce the model's accuracy on a targeted subpopulation, or simply to corrupt the model by injecting arbitrarily large gradient values. Crucially, gradient poisoning does not require breaking any cryptographic primitive — the attacker simply sends a crafted message, and the server, not knowing the true gradient, cannot distinguish it from a legitimate one.

Homomorphic encryption, as used in the `he_tenseal` and `he_concrete_tfhe` modes, addresses the confidentiality dimension of FL security: it prevents the server from reading gradient values. But it does nothing about integrity. A malicious client that encrypts a poisoned gradient produces an encrypted poisoned gradient. The server aggregates it without ever detecting the poison. Encryption and integrity are orthogonal security properties, and FHE addresses only the former.

### 7.2 How ZKP Addresses Integrity

Zero-knowledge proofs address the integrity problem by requiring each client to produce a cryptographic proof alongside its gradient update — a proof that the submitted gradient satisfies certain structural constraints. The server verifies this proof before including the gradient in aggregation. A poisoned gradient that violates the proven constraints causes the proof to fail. A poisoned gradient that somehow satisfies the constraints (within the norm bound and with a valid hash preimage) is still accepted, but the norm bound constraint significantly limits the damage such an update can cause.

The key insight is that the proof can simultaneously: (a) commit the client to a specific gradient vector, so they cannot change it after seeing other clients' updates; (b) prove that the gradient's $\ell_2$ norm is bounded, preventing outlier attacks; and (c) reveal nothing about the gradient's actual values to the server, preserving privacy.

This is the elegant combination that makes ZKP valuable in FL: it enforces integrity constraints on encrypted or private data without requiring the verifier to see the data.

### 7.3 The ZKP-FL Protocol in This Framework

At each training round in the `zkp_sampled` mode, the following sequence occurs for each client. The client completes local training and computes its gradient update vector $\Delta w$. Rather than proving properties of the full gradient (potentially millions of parameters), the client samples a fixed-size subset of $k = 100$ coordinates, where the sampling is deterministic — seeded by the round number and client identifier — so the server can verify which coordinates were included. This sampling is the key efficiency lever: proving 100 coordinates takes approximately 22 seconds, while proving millions would take hours.

For the sampled coordinates, the client computes a **MiMC hash commitment** — a field-element $C$ encoding the hash of the sampled values in a ZK-friendly arithmetic form. The client then calls the gnark proof service, which executes the Groth16 prover on the gradient circuit. This circuit encodes two claims simultaneously: first, that the prover knows a preimage of $C$ (the sampled gradients hash to $C$); second, that the $\ell_2$ norm of those gradients does not exceed the declared bound (norm boundedness). The output is a Groth16 proof $\pi$ of approximately 192 bytes — constant size regardless of how many constraints the circuit contains.

The client transmits its gradient update $\Delta w$, the commitment $C$, and the proof $\pi$ to the server. The server calls the gnark verification endpoint for each client's proof. Verification requires checking three pairing equations on the BN254 curve — a constant-time operation taking approximately 80 milliseconds regardless of the original circuit size. The server rejects any client whose proof fails verification and proceeds to FedAvg aggregation over only the verified updates.

### 7.4 Security Properties of the Protocol

The norm-bounding ZKP provides the following formal guarantees. First, **commitment binding**: a client cannot change their gradient after committing to $C$, because finding a second preimage for MiMC requires breaking its collision resistance. This prevents "last-look" attacks where a client waits to see partial aggregation results and adjusts its update accordingly.

Second, **norm soundness**: the Groth16 proof's soundness guarantee ensures that no computationally bounded prover can generate a valid proof for a gradient that violates the norm bound. The norm bound prevents a single malicious client from dominating the global update — if each honest client's gradient has norm at most $B$, the malicious client also cannot contribute more than $B$ per coordinate. For $n$ honest clients and 1 malicious client, the malicious client's contribution to the aggregate is bounded to $1/n$ of the honest contribution (in norm).

Third, **zero-knowledge**: the proof reveals nothing about the gradient values beyond what the commitment and proof imply. The server learns that the norm is bounded — a single bit of information — and that the gradient commits to $C$. It does not learn the actual gradient direction, magnitude distribution, or any other property. This zero-knowledge property is what allows combining ZKP with plaintext gradient transmission: in the `zkp_sampled` mode, the server sees the gradients in plaintext but the ZKP still ensures their authenticity. In the `he_tenseal_zkp` and `he_concrete_tfhe_zkp` modes, ZKP is layered on top of homomorphic encryption — the server sees only encrypted gradients and a proof that those gradients (whatever they are) satisfy the norm bound.

### 7.5 Sampling and Statistical Security

The use of sampling introduces a probabilistic security dimension. The adversary knows their full gradient update but does not know which $k$ coordinates will be sampled before submitting the proof (because the sampling seed incorporates the round number and other values determined after the adversary commits). If the malicious gradient violates the norm bound in some coordinates but satisfies it in others, they cannot selectively corrupt only the non-sampled coordinates without risk: with probability $k/n_{\text{params}}$, at least one corrupted coordinate will be in the sample, causing proof failure.

For a gradient with $10^5$ parameters and $k = 100$ sampled, the probability that a uniform poisoning attack corrupts at least one sampled coordinate is $1 - (1 - 100/100000)^{n_{\text{poisoned}}} \approx 1 - e^{-n_{\text{poisoned}}/1000}$. Poisoning 1,000 coordinates gives 63% detection probability; poisoning 5,000 gives 99.3%. This statistical guarantee is not as strong as full-gradient proving, but is practically sufficient and makes the approach computationally tractable.

---

## 8. Pedersen Commitments — Deep Dive

### 8.1 Conceptual Foundation

**Pedersen commitments** (Pedersen, 1991) are one of the simplest and most elegant constructions in commitment scheme theory. They are unconditionally (perfectly) hiding and computationally binding, based on the hardness of the discrete logarithm problem. Their construction predates zero-knowledge proofs by design motivation but fits naturally into the ZKP framework as a commitment primitive.

The conceptual model of a Pedersen commitment is that of a **locked safe with two dials**. The first dial, turned to position $m$ (the message), sets the content. The second dial, turned to a random position $r$ (the blinding factor), locks the safe. Knowing which position $r$ opens to which content is the "discrete log" — and if discrete log is hard, even someone who sees the safe ($C$) cannot determine $m$ without knowing $r$.

### 8.2 Mathematical Construction

Let $\mathbb{G}$ be a cyclic group of prime order $p$ in which the discrete logarithm problem is hard. Let $g$ and $h$ be two independently chosen generators of $\mathbb{G}$, such that the discrete relationship $\log_g h$ is unknown. This independence condition — that no one knows the discrete log of $h$ base $g$ — is essential to the binding property.

To commit to a message $m \in \mathbb{Z}_p$:

1. Choose a uniformly random blinding factor $r \leftarrow \mathbb{Z}_p$
2. Compute the commitment $C = g^m \cdot h^r \in \mathbb{G}$

To open the commitment, reveal $(m, r)$. The verifier checks that $g^m \cdot h^r = C$.

**Hiding**: For any two messages $m_0, m_1$ and any commitment $C$, there exists a unique $r_0$ such that $C = g^{m_0} h^{r_0}$ and a unique $r_1$ such that $C = g^{m_1} h^{r_1}$. Both are equally likely — the commitment $C$ is uniformly distributed in $\mathbb{G}$ regardless of the message $m$ (as long as $r$ is random). The hiding is therefore **perfect**: even an infinitely powerful adversary learns nothing about $m$ from $C$.

**Binding**: Suppose a cheating committer can produce two valid openings $(m, r)$ and $(m', r')$ for the same commitment $C$ (with $m \neq m'$). Then $g^m h^r = g^{m'} h^{r'}$, which gives $g^{m - m'} = h^{r' - r}$, meaning $\log_g h = (m - m')(r' - r)^{-1} \mod p$ — the cheater has computed the discrete log of $h$ base $g$. If discrete log is hard, this contradicts the binding property. Therefore, the binding is **computational**: it holds against polynomial-time adversaries but could in principle be broken by a sufficiently powerful (e.g., quantum) prover.

### 8.3 Homomorphic Properties

Pedersen commitments have a beautiful **additive homomorphism** property: the commitment of a sum is the product of commitments.

$$\text{Commit}(m_1, r_1) \cdot \text{Commit}(m_2, r_2) = g^{m_1} h^{r_1} \cdot g^{m_2} h^{r_2} = g^{m_1 + m_2} h^{r_1 + r_2} = \text{Commit}(m_1 + m_2, r_1 + r_2)$$

This homomorphism allows verification of linear properties of committed values. For example, a verifier can check that the sum of committed values equals a known public sum, without learning the individual values. In federated learning, this would allow a server to verify that the sum of clients' committed gradient updates equals a declared total, providing aggregate integrity without individual exposure.

### 8.4 Schnorr Proofs of Knowledge on Pedersen Commitments

Once a value is committed, a sigma protocol can prove properties of the committed value in zero-knowledge. The **Schnorr protocol for discrete log** can be adapted to prove knowledge of the opening $(m, r)$ of commitment $C$ without revealing either $m$ or $r$.

The protocol works as follows. The prover chooses random values $a_m, a_r \leftarrow \mathbb{Z}_p$ and computes an "announcement" $A = g^{a_m} h^{a_r}$. The verifier sends a random challenge $e$. The prover responds with $z_m = a_m + em$ and $z_r = a_r + er$ (both computations in $\mathbb{Z}_p$). The verifier accepts if $g^{z_m} h^{z_r} = A \cdot C^e$. An honest prover always satisfies this equation; a cheating prover who does not know $(m, r)$ cannot construct a consistent response to a random challenge.

Applying the Fiat-Shamir transform makes this non-interactive: the challenge $e = H(C, A, x)$ is derived from the hash of the commitment, announcement, and statement. The resulting proof $\pi = (A, z_m, z_r)$ is a non-interactive zero-knowledge proof of knowledge of the Pedersen opening — three group elements or two field elements, compact and efficient.

### 8.5 Pedersen in This Framework — Stub Status

> **⚠️ The `pedersen` backend is an unfinished stub and provides no Byzantine-fault protection.** It exists solely to let the ZKP code paths be exercised in environments where the Go gnark binary cannot be built (e.g. sandboxed CI). Do not use it in any deployment that requires real integrity guarantees.

The framework's Pedersen backend computes commitments over a 2048-bit safe-prime group $\mathbb{G} = \langle g \rangle \subset \mathbb{Z}_p^*$ (`fl/core/zkp.py`). However, the implementation stops there. Three components that would be required for a functional ZKP pipeline are deliberately absent:

**1. Proof of knowledge (Schnorr / Fiat-Shamir).** The Schnorr protocol described in §8.4 — commit, challenge, respond — is not implemented. No `prove_knowledge()` or `verify_knowledge()` function exists. Commitment objects are computed in-process and then discarded; nothing is sent to the server.

**2. Norm-bound proof.** Even a complete Schnorr proof-of-knowledge only shows the client *knows* the gradient that produced the commitment. It does not show the gradient's L2 norm is within any agreed bound. To prove a norm bound over Pedersen commitments, a Bulletproofs-style inner-product argument would be required. Bulletproofs are considerably more complex to implement correctly and, in pure Python, would be slower than the gnark Groth16 path while providing weaker security guarantees.

**3. Wire transport.** Because proofs do not exist, nothing is serialised into the Flower metrics dictionary. The server receives empty proof payloads and falls back to plain FedAvg — identical to the baseline mode in terms of Byzantine-fault tolerance.

The theoretical limitations of a fully-implemented Pedersen path are also significant. Proving knowledge of a gradient value is necessary but not sufficient for FL security — the norm constraint is what prevents gradient poisoning attacks. Additionally, a 2048-bit discrete-logarithm group provides approximately 112 bits of security, below the 128-bit target.

### 8.6 Why Pedersen is Being Superseded

The transition from Pedersen to Groth16 in this framework reflects the broader evolution of the ZKP field. Pedersen commitments are elegant and efficient for proving knowledge of values, but they are commitment schemes, not general-purpose proof systems. Expressing the norm constraint $\|w\|_2^2 \leq B$ over Pedersen commitments requires custom non-standard extensions (range proofs, arithmetic proof protocols) that become complex and expensive to implement and verify. Groth16, by contrast, accepts arbitrary arithmetic circuits — expressing the norm constraint is simply a matter of writing the corresponding R1CS constraints, which the gnark compiler handles automatically.

---

## 9. gnark and Groth16 zk-SNARKs — Deep Dive

### 9.1 The Groth16 Protocol

**Groth16** (Groth, 2016) is the most widely deployed zk-SNARK protocol in production use. It produces constant-size proofs (3 elliptic curve points) and requires $O(n)$ pairing operations for setup but only 3 pairings for verification, regardless of the circuit's $n$ constraints. These properties make it the preferred proof system for applications where many proofs are verified by the same verifier.

The protocol builds on the **Quadratic Arithmetic Program (QAP)** representation of arithmetic circuits. Given an R1CS with $m$ constraints and $n$ variables, there exist polynomials $u_i(X), v_i(X), w_i(X)$ for $i = 0, \ldots, n$ such that the R1CS is satisfiable if and only if:

$$\left(\sum_{i=0}^{n} a_i \cdot u_i(X)\right) \cdot \left(\sum_{i=0}^{n} a_i \cdot v_i(X)\right) - \left(\sum_{i=0}^{n} a_i \cdot w_i(X)\right) = H(X) \cdot Z(X)$$

for some polynomial $H(X)$ and the "vanishing polynomial" $Z(X) = \prod_{i=1}^{m}(X - \omega^i)$ (whose roots are the constraint indices). The prover's private witness $\{a_i\}$ satisfies this polynomial identity; anyone who can construct $H(X)$ such that this holds knows a satisfying assignment.

Groth16 turns this polynomial identity into a pairing-based cryptographic proof. During a one-time **trusted setup**, trapdoor values $(\alpha, \beta, \gamma, \delta, \tau)$ are sampled from $\mathbb{Z}_p$ and then destroyed. The setup produces a **Structured Reference String (SRS)** — a public collection of elliptic curve points encoding powers of $\tau$: $\{[1]_1, [\tau]_1, [\tau^2]_1, \ldots, [\tau^m]_1\}$ in $\mathbb{G}_1$ and $\mathbb{G}_2$. The SRS enables the prover to commit to polynomials without revealing them (by evaluating them at the encrypted point $\tau$ inside the curve), but the SRS alone cannot be used to recover $\tau$ — that information was destroyed.

The prover, given the witness, evaluates $H(\tau)$ inside the curve using the SRS, producing commitments to the prover polynomials. The proof $\pi = (A, B, C) \in \mathbb{G}_1 \times \mathbb{G}_2 \times \mathbb{G}_1$ is three elliptic curve points computed from the witness and the SRS. The verifier checks the proof using three pairing equations that, if satisfied, certify the polynomial identity holds at $\tau$ with overwhelming probability (by the Schwartz-Zippel lemma).

### 9.2 The Trusted Setup and Its Implications

The trusted setup is Groth16's most controversial aspect. During setup, secret scalars $(\alpha, \beta, \gamma, \delta, \tau)$ are chosen and their encoded powers embedded in the SRS. If any participant in the setup retains these values — the "toxic waste" — they can forge proofs for arbitrary false statements. The SRS is "trustworthy" only if the toxic waste is genuinely destroyed.

In production deployments (Zcash, Ethereum), the trusted setup is conducted as a **Powers-of-Tau multi-party computation ceremony** (MPC). Many participants each contribute randomness, and the SRS is the accumulated product of all contributions. The toxic waste is only computable if *all* participants are malicious — if even one participant is honest and destroys their contribution, the toxic waste is unrecoverable. The 2022 Hermez Ceremony involved over 3,000 participants.

In this framework, the SRS is generated locally by gnark for each circuit definition. This is acceptable for research and experimental purposes — the threat model does not include adversaries attempting to forge ZKP proofs by compromising the trusted setup. For a production healthcare FL deployment, a proper ceremony should be conducted. This is acknowledged as a known limitation in the framework's security documentation.

### 9.3 The Gradient Circuit

The gnark circuit implemented in this framework encodes two claims that together constitute a meaningful integrity guarantee for FL gradient updates.

The first claim is **hash commitment**: the sampled gradient coordinates, when quantized to field elements (by multiplying by a fixed scale factor and rounding) and hashed through a MiMC hash chain, produce the publicly declared commitment value $C$. This proves the client is not changing their gradient after the fact and is committed to the specific values they submitted.

The second claim is **norm boundedness**: the weighted sum of squared quantized coordinates does not exceed the declared bound $B^2 \cdot \text{scale}^2$. Working in the quantized domain (integers scaled by $10^6$) avoids floating-point ambiguity inside the finite-field circuit. The norm bound is a sum-of-squares constraint over the circuit wires, which is naturally expressed as a fixed number of multiplication and addition gates.

What makes this circuit design interesting from a ZKP perspective is the **combination** of these two constraints in one proof. Prior to gnark, implementing this combination required either specialized protocols (Bulletproofs for range proofs, separate commitment schemes) or accepting that norm proofs are not binding to specific gradient values. By encoding both as R1CS constraints in a single Groth16 circuit, the prover produces a single 192-byte proof that simultaneously certifies both properties.

### 9.4 gnark as a Proof System Library

**gnark** (Consensys, 2022) is a Go library for writing, compiling, and proving arithmetic circuits. It provides a high-level circuit definition language in Go that compiles down to R1CS, supports Groth16 and PLONK backends, and is optimized for performance on modern hardware. The choice of Go over Rust (which hosts competitive libraries like bellman, arkworks) reflects a pragmatic engineering decision: Go's simpler deployment story, easier HTTP service construction, and good performance on Apple Silicon made it the best fit for this framework's needs.

The gnark proof service in this framework operates as a stateless HTTP microservice. The circuit definition (gradient coordinates, scale, norm bound) is compiled once at startup and cached. Subsequent prove requests use the precompiled SRS and proving key without recompilation. Verification uses the correspondingly cached verification key, which is a small fixed-size object (~1.2 KB) regardless of circuit size. This is architecturally clean: the gnark service is a cryptographic co-processor, and the Python FL framework calls it via HTTP without managing any Go or cryptographic code directly.

### 9.5 Why Groth16 and not Other Systems

Groth16's choice over alternatives reflects this framework's requirements. **Bulletproofs** would eliminate the trusted setup and provide range proofs natively, but verification time is $O(n)$ in the number of constraints — for 32,000 constraints, this would be significantly slower than Groth16's constant-time verification. In FL, the server verifies proofs from all clients at each round, so verification speed is precious.

**PLONK** (the other backend gnark supports) achieves a *universal* SRS that works for all circuits below a size threshold, avoiding per-circuit setup. However, PLONK's proof size is larger and proving time is somewhat slower than optimized Groth16. For a system with a fixed circuit definition (the gradient norm circuit is not changing between FL rounds), Groth16's per-circuit setup is acceptable and its performance advantage justifies it.

**zk-STARKs** would provide post-quantum security and eliminate trusted setup, but their proof sizes (tens of kilobytes) are several orders of magnitude larger than Groth16's 192 bytes. For FL systems transmitting proofs over potentially bandwidth-constrained links, this overhead would be significant. The post-quantum argument for STARKs is also less urgent in the FL context because the gradient norm bound provides its security guarantee in real-time — a quantum adversary who breaks the proof long after training completes cannot retroactively un-bound the gradients that were aggregated during training.

---

## 10. Pedersen vs Groth16: A Conceptual Comparison

Pedersen commitments and Groth16 proofs represent two fundamentally different points in the design space of cryptographic proof systems. Understanding their conceptual differences clarifies why gnark was adopted as the primary backend and Pedersen retained only as a legacy alternative.

### 10.1 Generality of the Proven Statement

A Pedersen commitment proves one thing: the committer knows a value $m$ and a blinding factor $r$ such that $C = g^m h^r$. This is a proof of *knowledge of an opening* — a specific, narrow statement about the commitment relation. Proving any additional property — that $m$ is within a range, that $m$ satisfies a linear constraint, that $m$ is consistent with a hash of other values — requires building additional protocol machinery on top of the base commitment scheme. Range proofs over Pedersen commitments exist (Bulletproofs, Borromean ring signatures) but are non-trivially complex and each targets a specific type of constraint.

A Groth16 proof proves an arbitrary NP statement: "there exists a witness satisfying this arithmetic circuit." The arithmetic circuit can encode any combination of addition, multiplication, comparison, hash evaluation, or any other polynomial-time computation. The proof generation and verification mechanisms are the same regardless of what the circuit computes — only the circuit definition changes. This generality is Groth16's defining advantage: adding the norm constraint to the gradient commitment required writing approximately 100 additional lines of Go circuit code, not designing a new cryptographic protocol.

### 10.2 Proof Size and Verification Cost

Pedersen's proof via Schnorr (non-interactive, Fiat-Shamir) for committing to a single gradient coordinate produces a proof of approximately 64–128 bytes — two or four field elements. For 100 sampled coordinates, the total proof size is 6,400–12,800 bytes. Verification requires checking one group equation per coordinate, scaling linearly with the number of coordinates proved.

Groth16 produces a proof of exactly 192 bytes (three BN254 curve points) regardless of whether the circuit has 100 constraints or 100,000. Verification requires checking exactly three pairing equations, a constant-time operation taking approximately 80 milliseconds. As the circuit grows more complex (more gradient coordinates, more hash rounds, additional constraints), the proof size and verification time do not change. This asymptotic profile is ideal for the server-side verification bottleneck in FL: as the client's local model grows, proof verification overhead at the server remains fixed.

The tradeoff is proving time: Groth16 takes approximately 22 seconds per 100-coordinate proof (Apple M1), while Schnorr-based Pedersen proving takes under 1 second. This proving overhead is the primary cost of using Groth16 in FL, adding roughly 22–45 seconds per client per round.

### 10.3 Trust and Setup Assumptions

Pedersen commitments require no trusted setup. The generators $g$ and $h$ are chosen randomly (from a hash-to-curve procedure), and security holds under the sole assumption that the discrete log problem is hard in $\mathbb{G}$. Any party can independently verify the generators were chosen without knowledge of their discrete relationship.

Groth16 requires a trusted setup: the SRS must be generated by a trustworthy party or multi-party process. If the setup is compromised, all proofs using that SRS can be forged. This is a systemic risk that does not exist for Pedersen: a compromised Pedersen setup (if someone discovers $\log_g h$) merely allows the committer to equivocate — to change the committed value after the fact — affecting only individual commitments, not all proofs system-wide.

In the context of this framework — a research prototype used for experimental evaluation — the trusted setup assumption is acceptable. In a production deployment with financial or medical consequences, the trusted setup would require a proper MPC ceremony.

### 10.4 Security Strength

Both Pedersen (over 2048-bit groups) and Groth16 (over BN254) provide classical security, though at different levels: approximately 112 bits and 128 bits respectively. Neither is post-quantum secure. An adversary with a large-scale quantum computer could break both systems using Shor's algorithm for discrete logs.

The practical consequence is that both systems should be considered computationally binding only against current classical adversaries, with an understood upgrade path (to post-quantum ZKP systems like zk-STARKs or lattice-based commitments) if quantum threats materialize.

### 10.5 Expressive Power Summary

| Dimension               | Pedersen + Schnorr       | Groth16 (gnark)             |
|-------------------------|--------------------------|------------------------------|
| Proven statement        | Knowledge of commitment opening | Arbitrary arithmetic circuit |
| Norm constraint support | ❌ Requires separate range proof | ✅ Native R1CS constraint |
| Proof size              | Scales with number of values | Constant (192 bytes) |
| Verification cost       | Linear in values proved  | Constant (3 pairings, ~80ms) |
| Trusted setup           | None required            | Required (per circuit) |
| Proving time            | ~1s (Schnorr)            | ~22s (Groth16, 100 coords) |
| Security level          | ~112-bit (2048-bit DL)   | ~128-bit (BN254)             |
| Post-quantum security   | ❌ No                    | ❌ No                        |
| Go/C++ dependency       | None (Python-native)     | gnark Go service required    |

---

## 11. Security Model and Limitations

### 11.1 What ZKP Guarantees (and Does Not)

The ZKP protocol in this framework provides precisely defined guarantees that must not be overstated.

The norm-bounding proof guarantees that the proved gradient vector has $\ell_2$ norm at most $B$ in the sampled coordinates. It does not guarantee that the unsampled coordinates are also bounded — a sophisticated adversary could corrupt only coordinates outside the sampled region. The statistical detection probability depends on the number of corrupted coordinates and the sample size, as analyzed in Section 7.5.

The commitment proof guarantees binding — the client cannot change the submitted gradient after publishing the commitment $C$. It does not guarantee that the committed gradient was produced by honest training on the client's declared dataset. A client can run arbitrary computation, as long as the resulting gradient satisfies the norm constraint. ZKP cannot distinguish a gradient from genuine training from one manufactured to be adversarially useful while satisfying the norm bound.

No proof of *data quality* or *data provenance* is provided. The ZKP proves structural properties of the gradient, not the integrity of the data that produced it. This is a fundamental limitation: encoding "trained on a clean, representative dataset" as an arithmetic circuit constraint is an unsolved research problem.

### 11.2 Combining ZKP and FHE

The `he_tenseal_zkp` and `he_concrete_tfhe_zkp` modes layer ZKP on top of homomorphic encryption, providing both confidentiality and integrity. This combination is more powerful than either alone but introduces a subtle complication: the ZKP proof is generated on the *plaintext* gradient (the prover must know the gradient values to compute the proof) and transmitted alongside the encrypted gradient. The server can verify the proof's validity without decrypting the gradient, because the proof and the commitment $C$ are sufficient for verification.

This architecture means the order of operations is: train → sample coordinates → compute MiMC commitment in plaintext → generate ZKP proof → encrypt gradient → transmit (encrypted gradient, commitment $C$, proof $\pi$). The server verifies $\pi$ and $C$ match (via the gnark verifier), confirms the integrity properties, then aggregates encrypted gradients homomorphically. Neither the plaintext gradient nor any information about it (beyond the norm bound) reaches the server.

### 11.3 The Non-Post-Quantum Caveat

Both gnark (Groth16 on BN254) and Pedersen (discrete log in a prime group) are broken by quantum computers running Shor's algorithm. This is a known limitation that aligns this framework with current production ZKP standards (Ethereum, Zcash) which also use BN254. The quantum timeline for large-scale quantum computation remains uncertain, but regulatory frameworks (NIST post-quantum standards, 2024) are beginning to require migration plans.

Post-quantum ZKP alternatives include zk-STARKs (hash-based, quantum-resistant) and lattice-based commitment schemes. Neither is yet as mature or efficient as pairing-based SNARKs for general-purpose circuit proving, but rapid progress in this area is expected over the next 5–10 years.

### 11.4 Composition with Differential Privacy

The `dp` mode uses Opacus-based differential privacy (DP-SGD with Gaussian noise) to provide membership privacy — preventing the server from inferring whether a specific individual's data contributed to the model. This is a complementary guarantee to ZKP's integrity protection. The combined modes (`he_tenseal_zkp`, `he_concrete_tfhe_zkp`) do not yet include DP, though combining all three mechanisms is theoretically sound: DP operates at the training level, HE operates at the transmission level, and ZKP operates at the integrity verification level. All three can coexist without interaction.

---

## 12. Recursive Proofs, Polynomial Commitments, and Post-Quantum ZKP

### 12.1 Incrementally Verifiable Computation and NOVA

A fundamental scalability limitation of monolithic zk-SNARKs for ML applications is circuit size: proving an entire neural network training run as a single Groth16 circuit would require billions of R1CS constraints, making proof generation time prohibitive (days to weeks). **Incrementally Verifiable Computation (IVC)** resolves this by proving iterated computations step by step, producing a *constant-size* proof regardless of the number of steps.

Formally, IVC enables proofs of the statement: "Starting from state $z_0$, applying function $F$ exactly $n$ times yields state $z_n = F^n(z_0)$." Each step's proof $\pi_i$ certifies that $z_i = F(z_{i-1})$ *and* that $\pi_{i-1}$ is a valid IVC proof for the previous $i-1$ steps. The recursive structure means that the prover at step $i$ must "prove the verifier" — execute the ZKP verification of $\pi_{i-1}$ inside the arithmetic circuit for step $i$. The final proof $\pi_n$ has constant size (independent of $n$) and is verified in constant time.

The naïve implementation — encoding a full SNARK verifier circuit inside each step's circuit — is expensive. The verifier of Groth16 requires $O(|x|)$ pairing operations (where $|x|$ is the number of public inputs), adding thousands of R1CS constraints per step. **NOVA** (Kothapalli, Setty, Tzialla, 2022) achieves dramatically more efficient IVC via *folding schemes*.

**Folding** replaces full proof verification with a cheaper *accumulation* operation. Instead of verifying $\pi_{i-1}$ completely inside step $i$'s circuit, NOVA accumulates the claim "all previous steps are correct" into a *relaxed R1CS* instance — a relaxed version of the constraint system that admits an error term. The accumulation is very cheap (two multi-scalar multiplications in $\mathbb{G}_1$), and the final proof verifies the accumulated relaxed instance plus one standard SNARK proof for the last step. This reduces IVC overhead from $O(\text{circuit size})$ per step to a constant.

**SuperNova** and **HyperNova** extend NOVA to *non-uniform* computation — where the function $F_i$ at each step may differ — and to multi-folding with hypercube structures, enabling efficient proving for heterogeneous training loops (varying batch sizes, adaptive learning rates, conditional computation).

**IVC in federated learning**: VerifBFL (Bellachia et al., 2025) is the first FL system to use NOVA-based IVC for local training proofs. A client running $T$ local gradient descent steps generates one Groth16 proof committing to the entire $T$-step trajectory, with size independent of $T$. Proof generation for complete local training runs in under 81 seconds.

For this framework, IVC represents the upgrade path from the current single-round gradient sampling approach: rather than proving 100 sampled gradient coordinates from one round, a client could prove all $E$ epochs of local training using IVC, providing a much stronger integrity guarantee that the gradient was produced by genuine optimization rather than crafted by an adversarial but norm-bounded update.

---

### 12.2 KZG Polynomial Commitments

**KZG commitments** (Kate, Zaverucha, Goldberg, 2010) are polynomial commitment schemes underlying all PLONK-family proof systems and recently adopted as the basis for Ethereum's danksharding blob commitments. KZG allows a prover to commit to a polynomial $f(X)$ of degree $d$ as a single elliptic curve point and later prove evaluations $f(z) = y$ at any point $z$ with a single-curve-point proof.

**Construction**: In a trusted setup, scalar powers $\tau^0, \ldots, \tau^d$ are encoded as elliptic curve points $[1]_1, [\tau]_1, \ldots, [\tau^d]_1$ (the "powers of tau" SRS). To commit to $f(X) = \sum_{i=0}^d c_i X^i$, compute:

$$C = \sum_{i=0}^d c_i [\tau^i]_1 = [f(\tau)]_1$$

This single group element binds the committer to $f$ — changing any coefficient would produce a different curve point, and computing $\tau$ from the SRS requires solving the discrete logarithm.

**Evaluation proof**: To prove $f(z) = y$, note that if $f(z) = y$, then the polynomial $q(X) = (f(X) - y)/(X - z)$ is a polynomial of degree $d-1$ (no remainder). The prover computes $W = [q(\tau)]_1$ using the SRS, and the verifier checks via bilinear pairing:

$$e\!\left(C - [y]_1,\; [1]_2\right) = e\!\left(W,\; [\tau]_2 - [z]_2\right)$$

This single pairing equation certifies $f(z) = y$ in constant time with a one-group-element proof.

**KZG in ZKP for FL**: Rather than committing to gradient values via a hash function (as in this framework's MiMC-based circuit), a KZG-based approach could commit to the entire gradient vector as a polynomial (coefficient $i$ = gradient coordinate $i$), then selectively open evaluations requested by the server — one opening per sampled coordinate, each provable with one curve point. Compared to the current MiMC circuit:

| Property | MiMC Hash (current) | KZG polynomial commitment |
|---|---|---|
| Commitment size | ~32 bytes (field element) | ~48 bytes (G1 point) |
| Proof size per coordinate | Via Groth16 circuit (192 bytes total) | 48 bytes per evaluation |
| Prover time (100 coords) | ~22s (Groth16) | ~1ms per evaluation |
| Verifier time | ~80ms (3 pairings) | ~5ms per evaluation (1 pairing) |
| Trusted setup required | Yes (per circuit) | Yes (universal, reusable) |
| Binding on full gradient | ❌ (only sampled coords) | ✅ (entire polynomial) |

The critical advantage of KZG is that the commitment is to the *entire gradient polynomial*, so any randomly chosen coordinate can be opened on demand — the server can adaptively choose which coordinates to audit, whereas the current sampling must be fixed before proof generation. This dynamic auditing capability provides stronger statistical security against selective poisoning attacks.

---

### 12.3 Post-Quantum Zero-Knowledge Proofs

The Groth16 proof system (and all PLONK/KZG variants) relies on the hardness of the elliptic curve discrete logarithm problem (ECDLP). Shor's quantum algorithm solves ECDLP in polynomial time on a sufficiently large quantum computer (~4,000 logical qubits for BN254). This places all pairing-based ZKP systems in the same post-quantum threat class as RSA and classical Diffie-Hellman.

Unlike the HE components (CKKS, TFHE), which are based on LWE and believed quantum-secure, the ZKP layer of this framework's combined `he_*_zkp` modes would be broken by a quantum-capable adversary — allowing proof forgery (Byzantine clients could submit arbitrary gradients with valid-looking proofs) while gradient confidentiality (HE) remains intact.

Two main directions provide post-quantum ZKP:

**Hash-based proofs — zk-STARKs**: FRI (Fast Reed-Solomon Interactive Oracle Proof of Proximity) underpins STARKs (Ben-Sasson, Bentov, Horesh, Riabzev, 2018). Soundness reduces to collision resistance of the underlying hash function — believed post-quantum (Grover's algorithm provides only a quadratic speedup for preimage search, not collision finding, and 256-bit hashes retain at least 128-bit quantum security). STARKs require *no trusted setup* (the prover uses a public random oracle rather than an SRS) — eliminating the trusted setup vulnerability entirely.

| Property | Groth16 | zk-STARK |
|---|---|---|
| Post-quantum secure | ❌ (ECDLP) | ✅ (hash collision resistance) |
| Trusted setup | Required (per-circuit) | None |
| Proof size | 192 bytes | 50–200 KB |
| Prover time (100-coord circuit) | ~22s | ~5–30s (implementation-dependent) |
| Verifier time | ~80ms (3 pairings) | ~5–50ms (hash verification) |
| On-chain verification gas | ~600K gas (pairing) | ~3–10M gas (hash tree) |
| Tooling maturity | High (gnark, bellman) | Medium (StarkWare, Polygon zkEVM) |

The primary STARK trade-off is proof size: 50–200 KB versus 192 bytes for Groth16. For FL, where proofs are transmitted with gradient updates over institutional networks, STARK proof size adds meaningful overhead but is not prohibitive in the cross-silo setting.

**Lattice-based ZKPs**: Systems like **Ligero** (Ames, Hazay, Ishai, Venkitasubramaniam, 2017), **Brakedown** (Golovnev et al., 2023), and the **Shortest-Vector Proof** (SVP) paradigm provide ZKPs based on LWE or SIS (Short Integer Solution) — the same hardness assumptions as CKKS and TFHE. This offers *uniform* post-quantum security across both the HE and ZKP components of the full privacy stack. Proof sizes remain larger than pairing-based systems (~hundreds of kilobytes) and efficiency lags current pairing-based SNARKs, but the shared hardness assumption simplifies security arguments.

**Migration path for this framework**: The gnark service's modular design allows backend substitution. Replacing the Groth16 backend with a STARK backend (e.g., gnark's STARK support, or a Go STARK library) requires updating the gnark Go circuit and service, with no changes to the Python FL layer that calls the HTTP endpoints. The framework's plugin architecture isolates the ZKP backend from the training and aggregation logic.

---

## 13. Further Reading

### Foundational ZKP Papers

- **Goldwasser, Micali, Rackoff (1985)**: "The Knowledge Complexity of Interactive Proof Systems" — the founding paper defining zero-knowledge proofs. [ECRYPT reprint](https://people.csail.mit.edu/silvio/Selected%20Scientific%20Papers/Proof%20Systems/The_Knowledge_Complexity_Of_Interactive_Proof_Systems.pdf)
- **Goldreich, Micali, Wigderson (1987)**: "Proofs That Yield Nothing But Their Validity" — every NP language has a ZKP system. [IACR ePrint](https://www.wisdom.weizmann.ac.il/~oded/gmw1.html)
- **Fiat, Shamir (1986)**: "How to Prove Yourself: Practical Solutions to Identification and Signature Problems" — the Fiat-Shamir transform making ZKP non-interactive.
- **Pedersen (1991)**: "Non-Interactive and Information-Theoretic Secure Verifiable Secret Sharing" — introducing Pedersen commitments.

### zk-SNARK Papers

- **Groth (2016)**: "On the Size of Pairing-Based Non-Interactive Arguments" — the Groth16 protocol. [ePrint 2016/260](https://eprint.iacr.org/2016/260.pdf)
- **Parno, Howell, Gentry, Raykova (2013)**: "Pinocchio: Nearly Practical Verifiable Computation" — early SNARK system, predecessor to Groth16.
- **Bowe, Gabizon, Miers (2017)**: "Scalable Multi-party Computation for zk-SNARK Parameters" — multi-party setup ceremonies.
- **PLONK (2019)**: Gabizon, Williamson, Ciobotaru — universal SRS SNARKs. [ePrint 2019/953](https://eprint.iacr.org/2019/953.pdf)

### ZKP-Friendly Primitives

- **MiMC (Albrecht et al., 2016)**: "MiMC: Efficient Encryption and Cryptographic Hashing with Minimal Multiplicative Complexity" — the hash function used in this framework's circuit. [ePrint 2016/492](https://eprint.iacr.org/2016/492.pdf)
- **Poseidon (2019)**: Grassi et al. — a more recent ZK-friendly hash with lower R1CS cost. [ePrint 2019/458](https://eprint.iacr.org/2019/458.pdf)

### ZKP in Machine Learning

- **ZKML (2023)**: "Scaling Up Trustless DNN Inference with Zero-Knowledge Proofs" — verifiable neural network inference.
- **EZKL (2023)**: Open-source framework for converting neural networks to ZKP circuits. [github.com/zkonduit/ezkl](https://github.com/zkonduit/ezkl)

### ZKP in Federated Learning

- **Guo et al. (2021)**: "VKFL: Verifiable Knowledge Federated Learning" — ZKP for FL gradient integrity.
- **Zhao et al. (2022)**: "ZKFed: Secure Federated Learning with ZK-Proofs" — norm-bounded gradient proofs in FL.

### Libraries

| Library | Language | Backend | Notes |
|---------|----------|---------|-------|
| gnark | Go | Groth16, PLONK | Used in this framework |
| bellman | Rust | Groth16 | Zcash's original SNARK library |
| arkworks | Rust | Groth16, PLONK, Marlin | Modular, academic research |
| snarkjs | JavaScript | Groth16, PLONK | Browser-compatible |
| Bulletproofs | Rust | Range proofs | No trusted setup |
| EZKL | Python/Rust | PLONK | Neural network to ZKP |

---

> **Detailed implementation guides**:
> - [ZKP.md](ZKP.md) — practical FL integration guide (setup, API, configuration)
> - [ZKP.md](ZKP.md) — gnark service reference (HTTP API, circuit design, benchmarks)
> - [FHE.md](FHE.md) — homomorphic encryption (the complementary confidentiality mechanism)
> - [README.md](README.md) — all 10 modes with benchmarks


---

## Zero-Knowledge Proof Guide

> **Primary backend: gnark (Groth16 zk-SNARK)** — See [ZKP.md](ZKP.md) for full gnark setup.  
> Legacy Pedersen backend documented in the [appendix](#appendix-legacy-pedersen-backend) below.  
> Navigation: [README.md](README.md)

### Table of Contents
1. [What is ZKP in this framework?](#what-is-zkp-in-this-framework)
2. [Threat Model](#threat-model)
3. [Quick Start — gnark Backend](#quick-start--gnark-backend)
4. [How zkp_sampled Works in FL](#how-zkp_sampled-works-in-fl)
5. [Python API Reference](#python-api-reference)
6. [Circuit Design](#circuit-design)
7. [Performance Profile](#performance-profile)
8. [Configuration Options](#configuration-options)
9. [Troubleshooting](#troubleshooting)
10. [Appendix: Legacy Pedersen Backend](#appendix-legacy-pedersen-backend)

---

### What is ZKP in this framework?

This framework implements **Groth16 zk-SNARKs** via [gnark](https://github.com/consensys/gnark) (Go) to let federated learning clients prove the *integrity* of their submitted gradient updates — without revealing the raw values.

Each participating client:
1. Trains a local model on its private dataset
2. Computes a **Groth16 proof** that the gradient update satisfies two properties:
   - The update is committed to a specific value (MiMC hash matches)
   - The update's L2 norm is within a declared bound (Byzantine fault detection)
3. Sends the proof alongside its model update
4. The aggregation server verifies all proofs before aggregating

**What gnark proves per round**: For a sampled subset of `k` gradient coordinates, there exists a preimage (the actual gradient) whose MiMC hash equals the submitted commitment AND whose squared L2 norm ≤ `max_norm_sq × scale²`. A forged or poisoned gradient that passes norm-bounding would have to break ~128-bit security.

### Threat Model

| Adversary | Mode | Protection |
|-----------|------|-----------|
| HBC server (reads gradients) | `he_tenseal` / `he_concrete_tfhe` | ✅ Encryption hides values |
| Byzantine client (poisoned gradients) | `zkp_sampled` | ✅ ZKP proof prevents invalid updates |
| HBC server **+** Byzantine client | `he_tenseal_zkp` / `he_concrete_tfhe_zkp` | ✅ Both |
| Linkage/membership inference | `dp` | ✅ Differential privacy |

ZKP does **not** protect the gradient's confidentiality (values are visible to the server). Combine with HE for full protection.

---

### Quick Start — gnark Backend

#### 1. Prerequisites

- Go 1.21+ installed: `go version`
- gnark service binary compiled (do once):

```bash
cd fl_ppml/zkp_gnark_service
go mod tidy
go build -o gnark_service main.go
```

#### 2. Start the gnark Service

```bash
## Start in background (listens on :9000)
./gnark_service &
echo "gnark service PID: $!"

## Verify it's running
curl -s http://localhost:9000/health | python -m json.tool
## Expected: {"status": "ok", "circuit": "loaded"}
```

#### 3. Run ZKP Mode

```bash
cd fl_ppml

## Single ZKP mode
python -m compare.runner --dataset healthcare --modes zkp_sampled

## Or ZKP + encryption combinations
python -m compare.runner --dataset healthcare --modes he_tenseal_zkp
python -m compare.runner --dataset healthcare --modes he_concrete_tfhe_zkp
```

---

### How zkp_sampled Works in FL

#### Round Lifecycle (per client, per round)

```
Client trains locally
        |
        v
Sample k=100 gradient coordinates (deterministic seed from round+client_id)
        |
        v
Compute MiMC hash of sampled gradients → commitment C
        |
        v
POST /prove  →  gnark service generates Groth16 proof (π)
        |
        v
Send (gradients, C, π) to server
        |
        v
Server: POST /verify for each client proof
        |
        v
Only verified clients' gradients are FedAvg-aggregated
```

#### Why Sample Instead of All Gradients?

A neural network may have millions of parameters. Proving all coordinates would make each round take hours. Sampling 100 coordinates:
- Still catches poisoned gradients (attacker must corrupt all sampled coords probabilistically)
- Limits proving time to ~22s per client per round
- Reduces gnark circuit size from millions to 100 constraints

**Security**: If an attacker cannot predict which 100 coordinates will be sampled (the seed is derived from round and client ID), they cannot selectively poison non-sampled coordinates while passing proof verification.

#### Proof Verification in Aggregation

`server.py` aggregate step:

```python
from core.security import verify_gnark_proofs

def aggregate_fit(self, server_round, results, failures):
    verified_results = []
    for client, fit_res in results:
        proof_data = fit_res.metrics.get("zkp_proof")
        commitment = fit_res.metrics.get("zkp_commitment")
        if verify_gnark_proofs([proof_data], [commitment]):
            verified_results.append((client, fit_res))
        else:
            logger.warning(f"Client {client} rejected: invalid ZKP proof")
    return super().aggregate_fit(server_round, verified_results, failures)
```

---

### Python API Reference

#### `core/security.py`

```python
from core.security import generate_gnark_proofs, verify_gnark_proofs

## Client side: generate proofs for sampled gradient coordinates
proofs = generate_gnark_proofs(
    gradients: list[float],    # sampled gradient coordinates
    max_norm: float = 10.0,    # declared L2 norm bound
    gnark_url: str = "http://localhost:9000"
) -> list[dict]
## Returns: [{"proof": "<hex>", "commitment": "<hex>", "public_inputs": [...]}]

## Server side: verify proofs from all clients
is_valid = verify_gnark_proofs(
    proofs: list[dict],        # proof objects from generate_gnark_proofs
    commitments: list[str],    # hex-encoded MiMC commitments
    gnark_url: str = "http://localhost:9000"
) -> bool
## Returns True only if ALL proofs verify correctly
```

#### gnark HTTP API

The gnark service exposes three endpoints:

```
POST http://localhost:9000/prove
Content-Type: application/json

{
  "gradients": [0.12, -0.34, ...],   // k float64 values
  "max_norm_sq": 100.0,              // max_norm^2 * scale^2
  "scale": 1000                      // quantization scale
}

→ {"proof": "<hex>", "public_inputs": ["<commitment_hex>", "<bound_hex>"]}
```

```
POST http://localhost:9000/verify
Content-Type: application/json

{
  "proof": "<hex>",
  "public_inputs": ["<commitment_hex>", "<bound_hex>"]
}

→ {"valid": true}
```

```
POST http://localhost:9000/verify_light
Same as /verify but uses a cached verification key for ~3× faster parallel verification
```

---

### Circuit Design

#### MiMC Hash Commitment

```go
// In gnark circuit (Go):
func (c *GradientCircuit) Define(api frontend.API) error {
    // 1. MiMC hash of quantized gradients
    mimc, _ := mimc.NewMiMC(api)
    for _, g := range c.Gradients {
        quantized := api.Mul(g, c.Scale)
        mimc.Write(quantized)
    }
    hash := mimc.Sum()
    api.AssertIsEqual(hash, c.Commitment)

    // 2. L2 norm bound check
    sumSq := frontend.Variable(0)
    for _, g := range c.Gradients {
        q := api.Mul(g, c.Scale)
        sumSq = api.Add(sumSq, api.Mul(q, q))
    }
    api.AssertIsLessOrEqual(sumSq, c.MaxNormSq)
    return nil
}
```

**Why MiMC?**: MiMC is a ZK-friendly hash (uses field arithmetic). Standard hashes like SHA-256 cost >20,000 R1CS constraints; MiMC costs ~300 per evaluation, making it practical for gradient commitment.

**Quantization**: Float64 gradients are multiplied by `scale=1000` and cast to `int64` before hashing. This maps fractional values to integers that the finite-field circuit can process. The norm bound is correspondingly scaled: `bound_sq = max_norm_sq × scale²`.

#### Circuit Statistics

| Parameter | Value |
|-----------|-------|
| Gradient coordinates per proof | 100 (sampled) |
| R1CS constraints | ~32,000 |
| Proving key size | ~4.8 MB |
| Verification key size | ~1.2 KB |
| Trusted setup | Groth16 universal (BN254 curve) |

---

### Performance Profile

| Metric | Value | Notes |
|--------|-------|-------|
| Proof generation | ~22.4s per client per round | Go, Apple M1 |
| Proof verification | ~0.08s per proof | Fast — uses cached vk |
| Proof size | ~192 bytes | Groth16 compressed |
| Commitment size | 32 bytes | BN254 field element |
| gnark service startup | ~0.3s | circuit loaded at startup |
| Client communication overhead | +224 bytes/round | proof + commitment |

**Bottleneck**: Groth16 proving time. For `n` clients and `r` rounds: total ZKP time ≈ `n × r × 22.4s`. This is why `zkp_sampled` has ~11.6× wall-clock overhead vs baseline for 2 clients, 3 rounds (≈ 2×3×22.4s = 134s extra).

---

### Configuration Options

#### Gnark Service

```bash
## Default port
./gnark_service --port 9000

## Custom port
./gnark_service --port 9001
export GNARK_SERVICE_URL="http://localhost:9001"
```

#### Python Client

```python
## environment variable override
export GNARK_SERVICE_URL="http://localhost:9000"

## Or pass directly
proofs = generate_gnark_proofs(
    gradients=sample,
    max_norm=10.0,
    gnark_url="http://my-server:9000"
)
```

#### Sampling Configuration

In `core/security.py`, edit:

```python
ZKP_SAMPLE_SIZE = 100       # number of gradient coords to sample per proof
ZKP_MAX_NORM = 10.0         # default L2 norm bound
ZKP_SCALE = 1000            # quantization scale (int64 representation)
```

---

### Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `Connection refused :9000` | gnark service not started | `cd zkp_gnark_service && ./gnark_service &` |
| `go build` fails | Missing gnark dependency | `cd zkp_gnark_service && go mod tidy && go build` |
| `{"valid": false}` from /verify | Wrong public inputs passed | Ensure commitment from /prove is passed verbatim |
| Proof generation takes >60s | Slow CPU / M-chip Rosetta | Use native arm64 Go binary: `GOARCH=arm64 go build` |
| `proof_verification = 0.0` in benchmark | Using Pedersen backend | Ensure `FL_ZKP_BACKEND=gnark` env var is set |
| Circuit mismatch error | Stale proving key | Delete `zkp_gnark_service/*.key` and rebuild |

---

### Appendix: Legacy Pedersen Backend

The original ZKP implementation used **Pedersen commitments** (discrete-log-based, in Python). This is significantly lighter-weight but provides weaker guarantees:

- No norm-bounding circuit — only a hash commitment
- No succinct verification — verifier must re-derive commitments
- Cryptographic security based on discrete log in a 2048-bit group

#### Setup (Legacy)

```bash
python -m fl.keys generate zkp --output zkp_params.pkl --bit_length 2048
```

#### Running (Legacy)

```bash
## Set backend to pedersen explicitly
export FL_ZKP_BACKEND=pedersen

python simulation.py simulation --zkp --zkp_params zkp_params.pkl \
  --data_path ./data/ --dataset cifar --number_clients 2 \
  --rounds 2 --max_epochs 1 --benchmark
```

#### How Pedersen Works

Client: `C = g^m * h^r mod p`
- `m` = gradient value (scaled to integer)
- `r` = random blinding factor
- `g`, `h` = public generators
- `p` = large safe prime

Server receives `C` and later asks client to open it. Server cannot determine `m` from `C` alone (hiding). Client cannot change `m` after committing (binding).

**Compared to gnark**:

| | gnark (Groth16) | Pedersen |
|--|----------------|---------|
| Security | 128-bit (BN254) | 112-bit (2048-bit DL) |
| Norm bound proof | ✅ Yes (circuit) | ❌ No |
| Proof size | 192 bytes | ~512 bytes commitment |
| Proving time | ~22s | ~0.3s |
| Go dependency | Required | None |
| Verifier cost | O(1) (pairing) | O(n) (rerandomize) |

**Recommendation**: Use gnark for production. Use Pedersen only for development/testing where the Go service is unavailable.

---

### Further Reading

- [ZKP.md](ZKP.md) — comprehensive gnark service documentation
- [README.md](README.md) — all 10 modes compared
- [gnark documentation](https://docs.gnark.consensys.io/)
- [Groth16 paper](https://eprint.iacr.org/2016/260.pdf) — Jens Groth, 2016
- [MiMC paper](https://eprint.iacr.org/2016/492.pdf) — Albrecht et al., 2016


---

## gnark Zero-Knowledge Proof Guide

**The authoritative reference for the gnark Groth16 proof service in this framework.**

> This document consolidates all gnark-related documentation. See [ZKP.md](ZKP.md) for a one-page cheat sheet.

---

### Table of Contents

1. [What is gnark and Why We Use It](#1-what-is-gnark-and-why-we-use-it)
2. [Architecture](#2-architecture)
3. [Building the Service](#3-building-the-service)
4. [Running the Service](#4-running-the-service)
5. [HTTP API Reference](#5-http-api-reference)
6. [Python Client Integration](#6-python-client-integration)
7. [How ZKP Plugs into the FL Loop](#7-how-zkp-plugs-into-the-fl-loop)
8. [Circuit Design: What Is Being Proved](#8-circuit-design-what-is-being-proved)
9. [Performance Characteristics](#9-performance-characteristics)
10. [Using gnark Modes in the Framework](#10-using-gnark-modes-in-the-framework)
11. [Dataset-Specific Notes](#11-dataset-specific-notes)
12. [Security Model](#12-security-model)
13. [Configuration Reference](#13-configuration-reference)
14. [Troubleshooting](#14-troubleshooting)
15. [gnark vs. Legacy Pedersen Backend](#15-gnark-vs-legacy-pedersen-backend)

---

### 1. What is gnark and Why We Use It

#### gnark

[gnark](https://github.com/consensys/gnark) is an open-source Go library by Consensys for building and proving **zk-SNARK** circuits. This framework uses gnark for its **Groth16** backend — the most widely deployed zk-SNARK protocol, known for:

- **Constant proof size**: ~128–650 bytes regardless of witness complexity
- **Constant verification time**: ~2–15 ms, a fixed number of pairing operations on BN254
- **Non-interactivity**: One proof message, no back-and-forth protocol
- **Zero knowledge**: Prover reveals nothing about private inputs beyond the claim being proven

The proof service binary (`gnark_service`) compiles to `~18 MB` and runs as a lightweight HTTP server exposing a JSON API.

#### Why ZKP in Federated Learning?

In standard FL, the server aggregates gradient updates assuming all clients are honest. A **Byzantine adversary** — a malicious client — can submit any gradient: inflated norms, backdoor-poisoned updates, or manufactured updates to bias the global model. Homomorphic encryption (HE) protects *confidentiality* from the server, but does nothing to stop a malicious client from encrypting a poisoned gradient.

**ZKP addresses this orthogonal threat**: clients generate cryptographic proofs that their model update satisfies structural constraints (bounded gradient norm, hash commitment to parameters) *before* the server aggregates. The server verifies the proof — a fast operation — before including the update. A poisoned update that violates the proven constraints will cause proof verification to fail.

| Threat | HE Address? | ZKP Addresses? |
|--------|-------------|---------------|
| Honest-but-curious server reading gradients | ✓ | ✗ |
| Malicious server manipulating aggregation | ✗ | ✓ (zkFL variant) |
| Malicious client poisoning with valid-looking gradients | ✗ | ✓ (norm bound proof) |
| Membership inference on aggregated model | ✗ | ✗ |

---

### 2. Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    FL Training Round                           │
│                                                                │
│  Client k                                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. Local training → Δw_k (gradient update)              │  │
│  │ 2. Quantize weights to int64 (scale = 1,000,000)        │  │
│  │ 3. HTTP POST /prove → gnark service                      │  │
│  │    Request: {layer_name, weights_b64, shape, bound_sq}   │  │
│  │    Response: {proof_b64, hash_hex}                       │  │
│  │ 4. Attach proofs to FitRes metrics                       │  │
│  │ 5. Send Δw_k (plaintext or encrypted) + proofs to server │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  Server                                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 6. For each client: HTTP POST /verify → gnark service    │  │
│  │    Request: {layer_name, proof_b64, hash_hex, bound_sq}  │  │
│  │    Response: {verified: true/false}                      │  │
│  │ 7. Reject clients with failed proofs                     │  │
│  │ 8. FedAvg aggregate over verified clients only           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  gnark Service (Go, port 9000) — shared by client and server   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Circuit: MiMC hash + sum-of-squares norm bound           │  │
│  │ Backend: Groth16 on BN254 curve                          │  │
│  │ Setup: Per-shape circuit cache (compiled once, reused)   │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

#### Component Overview

| Component | Language | Location | Role |
|-----------|----------|----------|------|
| `gnark_service` (binary) | Go 1.21+ | `zkp_gnark_service/` | Proof generation + verification HTTP server |
| `core/zkp.py` | Python | `core/zkp.py` | Python client wrapping the HTTP API |
| `compare/experiment.py` | Python | `compare/experiment.py` | Auto-starts gnark service for ZKP modes |
| gnark v0.10.0 | Go | (dependency) | zk-SNARK circuit library |
| gnark-crypto v0.12.2 | Go | (dependency) | BN254 elliptic curve, pairings |

---

### 3. Building the Service

#### Prerequisites

- **Go 1.21+**: https://go.dev/dl/  
- **Verify**: `go version` (should print `go version go1.21` or newer)

#### Build Steps

```bash
## Navigate to the service directory
cd fl_ppml/zkp_gnark_service

## Download Go dependencies (~50 MB, cached after first run)
go mod download

## Compile the service binary
go build -o gnark_service main.go

## Verify the binary
ls -lh gnark_service
## Expected: -rwxr-xr-x ... 18M gnark_service
```

**Platform notes:**
- **macOS Apple Silicon**: Builds natively as `arm64`
- **macOS Intel**: Builds as `amd64`
- **Linux**: `go build -o gnark_service main.go` (same command)
- **Windows**: `go build -o gnark_service.exe main.go`

> **First compile is slow (~2 min)** due to gnark's circuit compilation. Subsequent builds are fast.

#### Docker Build (Alternative)

```bash
cd fl_ppml/zkp_gnark_service
docker build -t fhe-zkp-service:latest .

docker run -p 9000:9000 \
  -e ZKP_SERVICE_PORT=9000 \
  --name zkp-service \
  fhe-zkp-service:latest
```

---

### 4. Running the Service

#### Local Deployment

```bash
cd fl_ppml/zkp_gnark_service

## Start with defaults (listens on :9000)
./gnark_service
```

Expected startup output:
```
2026/03/02 10:23:45 Initializing gnark proof service...
2026/03/02 10:23:45 Service listening on :9000
2026/03/02 10:23:45 Available endpoints:
  POST /prove         - Generate Groth16 proof for layer weights
  POST /verify        - Verify a proof against public values
  POST /verify_light  - Lightweight verification (hash-only check)
  GET  /health        - Service health check
```

#### Background Process (recommended for FL runs)

```bash
## Start in background, redirect logs
./gnark_service > /tmp/gnark_service.log 2>&1 &

## Save the PID so you can stop it later
ZKP_PID=$!
echo "gnark service PID: $ZKP_PID"

## Verify it's alive
curl -s http://127.0.0.1:9000/health
## Expected: {"status": "ok", "service": "gnark-zkp"}

## Stop when done
kill $ZKP_PID
```

#### Auto-Start via Framework

The framework (`compare/experiment.py`) automatically starts and stops the gnark service for ZKP-containing modes (`zkp_sampled`, `he_tenseal_zkp`, `he_concrete_tfhe_zkp`). You only need to start it manually if running individual scripts.

#### Verify Service is Running

```bash
## Health check
curl http://127.0.0.1:9000/health

## Quick proof smoke-test
curl -s -X POST http://127.0.0.1:9000/prove \
  -H "Content-Type: application/json" \
  -d '{
    "layer_name": "smoke_test",
    "weights_b64": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
    "shape": [5],
    "scale": "1000000",
    "bound_sq": "10000000000"
  }' | python3 -m json.tool
```

Expected response:
```json
{
  "proof_b64": "oX6A...[base64 bytes]...",
  "hash_hex": "a1b2c3d4...",
  "verified": false
}
```

---

### 5. HTTP API Reference

All endpoints accept and return `application/json`. The service runs on `http://127.0.0.1:9000` by default.

#### `GET /health`

Service health check. Returns immediately.

**Response:**
```json
{
  "status": "ok",
  "service": "gnark-zkp"
}
```

---

#### `POST /prove`

Generate a Groth16 zk-SNARK proof for a model layer's weight tensor.

**What it proves**: The weight vector, when quantized and hashed with MiMC, has a sum-of-squares (squared Euclidean norm) below `bound_sq`. Specifically:

$$\sum_{i} w_i^2 \leq \text{bound\_sq}$$

where $w_i$ are integer-quantized weights (original float × scale, truncated to int64).

**Request:**
```json
{
  "layer_name": "fc1.weight",
  "weights_b64": "<base64 of little-endian int64 serialized weights>",
  "shape": [128, 64],
  "scale": "1000000",
  "bound_sq": "10000000000000000"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `layer_name` | string | Identifier for circuit caching (same shape = cached circuit) |
| `weights_b64` | string | Base64-encoded `int64[]` (8 bytes per weight, little-endian) |
| `shape` | int[] | Tensor dimensions (used for circuit size, not for proof) |
| `scale` | string | Quantization scale (float × scale → int64); use string to avoid precision loss |
| `bound_sq` | string | Max allowed sum-of-squares of quantized weights; string for large integers |

**Response:**
```json
{
  "proof_b64": "<base64 of gob-encoded Groth16 proof>",
  "hash_hex": "<hex of MiMC hash of quantized weights>",
  "verified": false
}
```

> `verified: false` in the `/prove` response is expected — the service does not self-verify by default. Use `/verify` to confirm.

**Typical latency**: 1.5–45s depending on layer size and hardware. The circuit is compiled once per (layer shape, bound) combination and cached in memory.

---

#### `POST /verify`

Verify a previously generated proof.

**Request:**
```json
{
  "layer_name": "fc1.weight",
  "proof_b64": "<same proof_b64 from /prove>",
  "hash_hex": "<same hash_hex from /prove>",
  "bound_sq": "10000000000000000"
}
```

**Response:**
```json
{
  "verified": true,
  "proof_b64": "",
  "hash_hex": ""
}
```

**Latency**: 2–15 ms (constant).

---

#### `POST /verify_light`

Lightweight verification using only the hash, without full zk-SNARK verification. Faster but provides weaker guarantees (hash pre-image resistance only, not full zero-knowledge soundness).

**Request:** Same as `/verify`  
**Response:** Same as `/verify`  
**Latency:** < 1 ms

> Use `/verify` (full) in production. `/verify_light` is for development/testing.

---

### 6. Python Client Integration

The Python client is in `core/zkp.py`. It provides two main functions:

#### `generate_gnark_proofs(state_dict, ...)`

Called by each FL **client** after local training, before sending updates to the server.

```python
from core.zkp import generate_gnark_proofs

## model_state is model.state_dict() or a dict of numpy arrays
proofs, total_bytes = generate_gnark_proofs(
    state_dict=model.state_dict(),
    layers=None,               # None = ALL layers; or list of layer names
    service_url="http://127.0.0.1:9000",
    scale=1_000_000,           # Quantization scale (float → int64)
    max_norm_sq=100.0 ** 2,    # Max gradient norm bound (squared)
    timeout=120                # Seconds to wait for proof service
)

## proofs: dict {layer_name: {"proof_b64": "...", "hash_hex": "..."}}
## total_bytes: int, total proof payload size for benchmarking
```

**Internal steps:**
1. For each selected layer:
   - Flatten the weight tensor to 1D
   - Multiply by `scale` and cast to `int64`
   - Serialize as little-endian bytes and base64-encode
   - `POST /prove` with quantized weights + `bound_sq = max_norm_sq * scale^2`
2. Collect `{proof_b64, hash_hex}` per layer
3. Return as JSON-serializable dict (passed via Flower's `metrics` field in `FitRes`)

---

#### `verify_gnark_proofs(params, layer_names, proofs, ...)`

Called by the FL **server** in `aggregate_fit` before including a client's update.

```python
from core.zkp import verify_gnark_proofs

ok, failed_layers = verify_gnark_proofs(
    params=parameters_to_ndarrays(fit_res.parameters),
    layer_names=["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"],
    proofs=fit_res.metrics.get("zkp_proofs"),   # dict from client
    service_url="http://127.0.0.1:9000",
    timeout=30
)

if not ok:
    print(f"Rejecting client — proof failed for layers: {failed_layers}")
    # Drop this client from aggregation
else:
    # Include in FedAvg
```

**Internal steps:**
1. For each layer in `layer_names`:
   - Re-quantize the received parameters (same scale as client)
   - Recompute `hash_hex` locally  
   - `POST /verify` with the client's `proof_b64` and the recomputed `hash_hex`
   - If `verified: false`, add to `failed_layers`
2. Return `(all_ok: bool, failed_layers: list[str])`

---

### 7. How ZKP Plugs into the FL Loop

```python
## client.py (simplified)
class ZKPClient(fl.client.NumPyClient):
    def fit(self, parameters, config):
        # 1. Standard local training
        set_parameters(self.model, parameters)
        train(self.model, self.train_loader, ...)
        updated_params = get_parameters(self.model)
        
        # 2. Generate ZKP proofs (NEW)
        with BenchmarkTimer("proof_generation"):
            proofs, proof_bytes = generate_gnark_proofs(
                self.model.state_dict()
            )
        
        # 3. Return parameters + proofs in metrics
        return updated_params, len(self.train_loader.dataset), {
            "zkp_proofs": json.dumps(proofs),
            "proof_bytes": proof_bytes,
        }

## server.py (simplified)
class ZKPStrategy(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        verified_results = []
        for client, fit_res in results:
            proofs = json.loads(fit_res.metrics.get("zkp_proofs", "{}"))
            
            # Verify before including
            with BenchmarkTimer("proof_verification"):
                ok, bad_layers = verify_gnark_proofs(
                    parameters_to_ndarrays(fit_res.parameters),
                    layer_names,
                    proofs
                )
            
            if ok:
                verified_results.append((client, fit_res))
            else:
                logger.warning(f"Client rejected: bad proof for {bad_layers}")
        
        # Aggregate only verified updates
        return super().aggregate_fit(server_round, verified_results, failures)
```

---

### 8. Circuit Design: What Is Being Proved

The gnark circuit (`main.go`) encodes two claims in a single Groth16 circuit over the BN254 prime field $\mathbb{F}_q$:

#### Claim 1: Hash Integrity (MiMC hash)

The prover holds a private witness $\mathbf{w} = (w_1, \ldots, w_n) \in \mathbb{Z}^n$ (quantized weights). The circuit computes:

$$h = \text{MiMC}(w_1, w_2, \ldots, w_n)$$

and asserts this equals the public input `hash_hex`. This is a **preimage proof**: the prover knows a preimage of the hash, demonstrating they hold the actual weights (not a fabricated proof). Since MiMC is a ZK-friendly hash (defined as polynomial operations over $\mathbb{F}_q$), it is efficient inside arithmetic circuits.

#### Claim 2: Norm Bound

The circuit also computes the **sum of squares** of all weights:

$$S = \sum_{i=1}^{n} w_i^2$$

and asserts $S \leq \text{bound\_sq}$ using gnark's range check gadgets. In finite field arithmetic, this uses the decomposition of the inequality into binary constraints.

**What this guarantees in FL context:**
- The gradient update has a bounded $\ell_2$ norm: $\|\Delta w\|_2 \leq \sqrt{\text{bound\_sq}} / \text{scale}$
- This prevents norm-inflated Byzantine attacks where adversarial gradients have dramatically larger magnitude than legitimate ones
- Combined with standard FedAvg averaging, norm-bounded updates limit any single client's impact on the global model

#### Circuit Sizes (approximate)

| Layer size | # Constraints | Proving time | Circuit cache size |
|-----------|--------------|-------------|-------------------|
| 10 params | ~500 | < 1s | Fast |
| 1k params | ~50k | ~1s | ~10 MB |
| 10k params | ~500k | ~15s | ~100 MB |
| 100k params | ~5M | ~2 min | ~1 GB |

> **`zkp_sampled` mode**: To keep proving tractable for large models, this mode samples a random subset of gradient coordinates (~100 values) and proves bounds on the sample. This provides probabilistic soundness with constant proving time.

---

### 9. Performance Characteristics

#### Latency (healthcare dataset, macOS Apple Silicon M-series)

| Operation | Time | Notes |
|-----------|------|-------|
| Proof generation (100 sampled params) | 15–45s | Includes circuit compilation on first call |
| Proof generation (cached circuit) | 5–20s | After first call for same layer shape |
| Proof verification (`/verify`) | 2–15 ms | Constant, independent of layer size |
| Service startup | < 1s | Just binary startup |
| Circuit compilation (first call) | 30–90s | One-time per (shape, bound_sq) pair |

#### Communication Overhead

| Per-layer proof overhead | Size |
|-------------------------|------|
| `proof_b64` | ~650 bytes |
| `hash_hex` | ~64 bytes (32-byte hash, hex-encoded) |
| JSON framing overhead | ~50 bytes per layer |
| **Total per layer** | **~764 bytes** |
| **Full model (6 layers)** | **~4.6 KB** |

Compare to HE upload overhead: ~244 MB (TenSEAL CKKS). ZKP communication overhead is negligible.

#### Memory Usage

| Component | Memory |
|-----------|--------|
| Service idle | ~50 MB |
| After first proof (circuit cached) | ~200–500 MB |
| Multiple cached circuits | ~200 MB × number of distinct layer shapes |

---

### 10. Using gnark Modes in the Framework

#### Via the Compare Runner (recommended)

```bash
## Single ZKP mode
python -m compare.runner --dataset healthcare --modes zkp_sampled

## All ZKP-containing modes
python -m compare.runner --dataset healthcare --modes zkp_sampled,he_tenseal_zkp,he_concrete_tfhe_zkp

## Full 10-mode comparison (gnark service auto-started)
python -m compare.runner --dataset healthcare --modes all
```

The runner (`compare/experiment.py`) automatically:
1. Detects if any selected mode requires gnark
2. Starts the gnark service process if not already running
3. Shuts it down after the run completes

#### Via Simulation Script

```bash
## Start gnark service manually first
cd zkp_gnark_service && ./gnark_service &

## Then run
export FL_ZKP_BACKEND=gnark
python simulation.py simulation \
  --zkp --zkp_backend gnark \
  --rounds 3 --number_clients 2 --benchmark \
  --save_results ./results/zkp_test/
```

#### Via Client-Server Mode

Terminal 1 (server):
```bash
export FL_ZKP_BACKEND=gnark
python main_server.py server --zkp --zkp_backend gnark \
  --rounds 3 --number_clients 2
```

Terminal 2 (gnark service):
```bash
cd zkp_gnark_service && ./gnark_service
```

Terminal 3+ (clients):
```bash
export FL_ZKP_BACKEND=gnark
python main_client.py client --zkp --zkp_backend gnark \
  --id_client 0
```

#### Environment Variables

| Variable | Default | Values | Description |
|----------|---------|--------|--------------|
| `FL_ZKP_BACKEND` | `gnark` | `gnark`, `pedersen` | Proof backend. `gnark` = Groth16 zk-SNARK (recommended); `pedersen` = legacy commitment (no soundness guarantee) |
| `FL_ZKP_SERVICE_URL` | `http://127.0.0.1:9000` | Any URL | gnark HTTP service endpoint |
| `FL_ZKP_SCALE` | `1000000` | Positive integer | Float→int64 quantization scale. Higher = more precision; too high = integer overflow |
| `FL_ZKP_MAX_NORM` | `100.0` | Positive float | Max allowed L2 norm of the gradient vector encoded in the proof circuit |
| `FL_ZKP_TIMEOUT` | `120` | Positive integer (s) | HTTP request timeout for each proof call. Increase for large models or slow hardware |
| `FL_ZKP_LAYERS` | `ALL` | `ALL` or CSV layer names | Which layers to prove when using the full (non-sampled) ZKP mode. `ALL` proves every layer |
| `FL_ZKP_SELECT_BY` | `size` | `size`, `random` | Layer selection strategy for `zkp_sampled`. `size` picks the layers with the most parameters (strongest coverage of the norm bound); `random` samples uniformly (varies across rounds) |
| `FL_ZKP_NUM_LAYERS` | `1` | Positive integer | Number of layers to sample per client per round. Lower = faster proofs, weaker per-round coverage. Mutually exclusive with `FL_ZKP_SAMPLE_PCT` (takes priority if both set) |
| `FL_ZKP_SAMPLE_PCT` | — | Float 0–1 | Fraction of layers to sample (ceiling). Alternative to `FL_ZKP_NUM_LAYERS`; useful when layer count varies across model architectures |
| `FL_ZKP_SAMPLE_SEED` | — | Integer | Fixes the random seed for layer sampling. Set for reproducible benchmarks; omit for random variation across rounds |
| `FL_ZKP_PARALLELISM` | `4` | Positive integer | Number of concurrent proof workers sent to the gnark service. Scaling above the gnark host's CPU count has diminishing returns |

---

### 11. Dataset-Specific Notes

#### Healthcare Dataset (918 samples, 13 features)
- Model: ~small fully-connected network
- Layer sizes: small (hundreds of parameters per layer)
- Proving time per round: ~15–45s (per client)
- Recommended: `zkp_sampled` mode for faster iteration

#### Credit Card Fraud Detection (284k samples, 30 features)
- Larger, so more gradient dimensions
- With subsampling (`--subsample 5000`): similar to healthcare
- Full dataset: proof generation ~12–18 minutes per round
- Recommended: `zkp_sampled` with a subset for development

#### CIFAR-10 (ResNet-style models)
- Large models with millions of parameters
- `zkp_sampled` is mandatory at scale (< 200 params sampled)
- Alternatively, prove only the final classification layer (`FL_ZKP_LAYERS=fc3.weight,fc3.bias`)

---

### 12. Security Model

#### What the Proof Guarantees

| Guarantee | Provided? | Notes |
|-----------|-----------|-------|
| Client holds the weight vector claimed | ✓ | MiMC hash preimage |
| Weight vector norm ≤ bound | ✓ | Sum-of-squares constraint |
| Client actually trained on their own data | ✗ | Not encoded in the circuit |
| Proof is unforgeable | ✓ | Groth16 soundness under BN254 discrete log |
| Proof reveals nothing about weights | ✓ | Groth16 zero-knowledge property |
| Server cannot forge a proof | ✓ | Knowledge soundness |

#### Trusted Setup

Groth16 requires a **Structured Reference String (SRS)** containing toxic waste $(\alpha, \beta, \gamma, \delta, \tau)$ that must be destroyed after setup. In this implementation:

- The SRS is generated fresh per circuit definition by gnark
- It is **not** generated via a public multi-party ceremony
- For research and experimental purposes, this is acceptable
- For production deployment, a proper Powers-of-Tau ceremony should be used

#### Threat Model Coverage

| Adversary Type | Mode | Protected? |
|---------------|------|-----------|
| Honest-but-curious server | `he_tenseal` / `he_concrete_tfhe` | ✓ |
| Malicious client (poisoned gradients) | `zkp_sampled` | ✓ (norm bound) |
| Malicious client + HBC server | `he_tenseal_zkp` / `he_concrete_tfhe_zkp` | ✓ both |
| Statistical reconstruction at test time | `dp` | ✓ |

#### Cryptographic Assumptions

The security of Groth16 rests on:
1. **Discrete logarithm hardness** on BN254 (254-bit security level)
2. **Bilinear assumptions** (BN254 is a type-3 pairing curve): d-power-of-tau and related assumptions
3. **Knowledge of exponent assumption**: for knowledge soundness

> ⚠️ BN254 provides ~128-bit classical security but is **not post-quantum secure** (unlike the RLWE-based HE schemes). An adversary with a large-scale quantum computer could forge proofs.

---

### 13. Configuration Reference

#### Quick Config Cheat Sheet

```bash
## Minimal setup — run from fl_ppml/
export FL_ZKP_BACKEND=gnark
export FL_ZKP_SERVICE_URL="http://127.0.0.1:9000"
export FL_ZKP_SCALE=1000000
export FL_ZKP_MAX_NORM=100.0
export FL_ZKP_TIMEOUT=120
export FL_ZKP_LAYERS=ALL
```

#### Tuning `FL_ZKP_SELECT_BY`

Controls which layers are selected when using `zkp_sampled` mode.

| Value | Behaviour | When to use |
|-------|-----------|-------------|
| `size` (default) | Selects the `FL_ZKP_NUM_LAYERS` layers with the most parameters | Maximum norm-bound coverage per proof; best for production security |
| `random` | Samples layers uniformly at random each round | Rotating coverage across rounds; useful for research into proof diversity |

Example — prove only the two largest layers:
```bash
export FL_ZKP_SELECT_BY=size
export FL_ZKP_NUM_LAYERS=2
```

#### Tuning `FL_ZKP_NUM_LAYERS` and `FL_ZKP_SAMPLE_PCT`

These two variables both govern how many layers get proven per round in `zkp_sampled` mode. Only one should be set; `FL_ZKP_NUM_LAYERS` takes priority if both are present.

| Setting | Proof time | Security |
|---------|-----------|----------|
| `FL_ZKP_NUM_LAYERS=1` (default) | Fastest (~22s for a 2-layer model) | One layer norm bound proven per round |
| `FL_ZKP_NUM_LAYERS=2` | ~2× slower | Full model norm bound per round |
| `FL_ZKP_SAMPLE_PCT=0.5` | Proves 50% of layers (ceiling) | Scales with model size automatically |

For a 2-layer model (`model.0.weight`, `model.0.bias`), `FL_ZKP_NUM_LAYERS=1` with `FL_ZKP_SELECT_BY=size` always proves `model.0.weight` (the weight matrix has far more parameters than the bias).

#### Tuning `FL_ZKP_PARALLELISM`

Sets the thread pool size used to dispatch proof requests to the gnark service concurrently.

```bash
## Default: 4 workers (good for Apple M-series, typical Linux dev boxes)
export FL_ZKP_PARALLELISM=4

## Maximize throughput on a high-core-count server
export FL_ZKP_PARALLELISM=16

## Serial proving (debug: isolate timing per layer)
export FL_ZKP_PARALLELISM=1
```

Setting this higher than the gnark host's CPU count yields minimal benefit and increases memory pressure on the service. The gnark service is the bottleneck, not the Python thread pool.

#### Tuning `FL_ZKP_TIMEOUT`

Increase for:
- Large models (many parameters → slower proving)
- Slower hardware (no M-series Apple Silicon / no GPU)
- First run (circuit compilation included in first request latency)

```bash
## For large models or slow hardware
export FL_ZKP_TIMEOUT=600

## For fast CI tests with tiny models
export FL_ZKP_TIMEOUT=30
```

#### Tuning `FL_ZKP_SCALE`

The scale converts floating-point weights to integers. Too low = precision loss. Too high = integer overflow.

| Weight magnitude | Recommended scale |
|-----------------|------------------|
| Very small (-0.01 to 0.01) | 1,000,000 (10^6) |
| Typical (-1.0 to 1.0) | 100,000 (10^5) |
| Large (-10 to 10) | 10,000 (10^4) |

The default `1,000,000` works for standard neural network weights.

#### Tuning `FL_ZKP_MAX_NORM`

This is the maximum allowed $\ell_2$ norm of the gradient update vector. Set equal to the DP clipping norm if using both DP and ZKP (they share the same bound):

```bash
## DP clipping norm = 1.0 (strong privacy)
export FL_ZKP_MAX_NORM=1.0

## Standard federated learning
export FL_ZKP_MAX_NORM=100.0
```

---

### 14. Troubleshooting

#### Service Won't Start

```bash
## Check port is free
lsof -i :9000

## Kill existing process
kill -9 $(lsof -ti :9000)

## Verify Go version (needs 1.21+)
go version

## Rebuild if binary is stale
cd zkp_gnark_service
go build -o gnark_service main.go

## macOS: fix quarantine if binary was downloaded (not compiled locally)
xattr -d com.apple.quarantine gnark_service
chmod +x gnark_service
```

#### Proof Generation Times Out

```bash
## Increase timeout
export FL_ZKP_TIMEOUT=600

## If still timing out, check service logs
tail -f /tmp/gnark_service.log

## Use sampled mode to reduce circuit size
python -m compare.runner --modes zkp_sampled   # not full zkp
```

#### `proof_verification = 0.0` in Benchmark Results

This means the server is not calling `/verify` — usually because the ZKP backend fell back to Pedersen (which has no server verifier in the gnark path). Fix:

```bash
## Ensure gnark backend is set
export FL_ZKP_BACKEND=gnark

## Or use the CLI flag
python main_server.py server --zkp --zkp_backend gnark ...
```

#### Proof Verification Fails (`verified: false`)

Most common cause: `FL_ZKP_SCALE` differs between client and server. Both must use the same scale value. Check:

```bash
## Both client and server processes must see the same value
echo $FL_ZKP_SCALE
```

Other causes:
- Network corruption of proof bytes (rare)
- Client sent gradients after local normalization at a different scale than expected

#### OOM (Out of Memory) on macOS

Reduce circuit memory by proving fewer layers at once:
```bash
## Only prove the final classification layer
export FL_ZKP_LAYERS="fc3.weight,fc3.bias"
```

Or use sampled mode (100 random parameters):
```bash
python -m compare.runner --modes zkp_sampled
```

---

### 15. gnark vs. Legacy Pedersen Backend

The framework includes a **legacy Pedersen commitment** backend (`FL_ZKP_BACKEND=pedersen`). It was the original ZKP implementation and is superseded by gnark.

| Property | gnark (Groth16) | Pedersen (Legacy) |
|----------|----------------|------------------|
| **Proof type** | zk-SNARK (Groth16 on BN254) | Commitment scheme |
| **Proof size** | ~650 bytes per layer | ~1–2 MB per layer |
| **Proof generation** | 1.5–45s (depends on circuit) | ~93s per layer |
| **Verification time** | 2–15 ms | ~85 ms |
| **Security model** | Zero-knowledge + soundness (crypto) | Computational hiding |
| **Infrastructure** | Go HTTP service required | Python-native, no service |
| **Setup** | Build Go binary | `python -m fl.keys generate zkp --output zkp_params.pkl` |
| **Norm bound proof** | ✓ Explicit circuit constraint | ✗ Not implemented |
| **Post-quantum** | ✗ BN254 is pre-quantum | ✗ |
| **Status** | ✅ Active default | ⚠️ Legacy, unsupported |

#### When to use Pedersen (Legacy)

Only if the gnark service cannot be compiled or run (e.g., no Go installation, restricted environment). Use `FL_ZKP_BACKEND=pedersen` and generate params first:
```bash
python -m fl.keys generate zkp --output zkp_params.pkl  # generates zkp_params.pkl
export FL_ZKP_BACKEND=pedersen
```

For all other cases, use gnark.

---

*See also: [ZKP.md](ZKP.md) for a one-page reference card*  
*See also: [ZKP.md](ZKP.md) for ZKP theory and FL integration concepts*  



---

## gnark ZKP Backend - Quick Reference

### ⚡ 30-Second Setup

```bash
## 1. Build service
cd fl_ppml/zkp_gnark_service
go mod download && go build -o gnark_service main.go

## 2. Start service (keep running)
./gnark_service

## 3. In another terminal, run FL
export FL_ZKP_BACKEND=gnark
cd ../
python simulation.py simulation --zkp --zkp_backend gnark --rounds 3 --benchmark
```

### 🔧 Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `FL_ZKP_BACKEND` | `gnark` | Backend: `gnark` or `pedersen` |
| `FL_ZKP_SERVICE_URL` | `http://127.0.0.1:9000` | Proof service endpoint |
| `FL_ZKP_SCALE` | `1000000` | Quantization scale for weights |
| `FL_ZKP_MAX_NORM` | `100.0` | Max gradient norm for proof |
| `FL_ZKP_TIMEOUT` | `120` | Service call timeout (seconds) |
| `FL_ZKP_LAYERS` | `ALL` | Layers to prove (CSV or `ALL`) |
| `FL_ZKP_SELECT_BY` | `size` | Sampling strategy: `size` or `random` |
| `FL_ZKP_NUM_LAYERS` | `1` | Number of layers to sample per round |
| `FL_ZKP_SAMPLE_PCT` | — | Fraction of layers to sample (alt. to `NUM_LAYERS`) |
| `FL_ZKP_SAMPLE_SEED` | — | Fixed seed for deterministic layer selection |
| `FL_ZKP_PARALLELISM` | `4` | Parallel proof workers (threads) |

### 📊 Service Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/prove` | POST | Generate proof for layer weights |
| `/verify` | POST | Verify proof against public values |
| `/health` | GET | Service health check |

### ✅ Verify Setup

```bash
## Check service is running
curl http://127.0.0.1:9000/health

## Test proof generation
python3 << 'EOF'
from core.zkp_gnark import generate_gnark_proofs
import numpy as np

params = {'layer0': np.random.randn(100).astype(np.float32)}
proofs, size = generate_gnark_proofs(params)
print(f"✓ Service working! Generated {len(proofs)} proof(s), {size} bytes")
EOF
```

### 🐛 Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| `Connection refused` | Start service: `./gnark_service` |
| `Timeout during proof generation` | Increase `FL_ZKP_TIMEOUT=300` |
| `Proof verification failed` | Check `FL_ZKP_SCALE` is same on client & server |
| `Port 9000 already in use` | `lsof -i :9000` then `kill -9 <PID>` |
| `Out of memory` | Reduce batch size or increase system RAM |

### 📈 Expected Performance

| Operation | Time | Size |
|-----------|------|------|
| Proof generation (10k params) | 1.2s | 650 bytes |
| Proof verification | 15ms | - |
| Per-layer overhead | ~0.5s | ~130-650 bytes |
| Network latency | Negligible | Fit in single TCP packet |

### 🏗️ Architecture

```
Client                          Service                    Server
┌─────────────┐                ┌─────────┐               ┌──────────┐
│   Train     │─────→ weights  │ /prove  │─────→ proof   │ /verify  │
│   Generate  │  POST          │         │                │ Aggregate│
│   Proof     │                │ Circuit │                │          │
└─────────────┘                └─────────┘               └──────────┘
```

### 📚 Documentation

- **Full Guide**: See [ZKP.md](ZKP.md)
- **Setup Instructions**: See [ZKP.md](ZKP.md)
- **Technical Details**: Circuit design, performance tuning in [ZKP.md](ZKP.md)

### 💡 Pro Tips

1. **Batch Proofs**: Prove multiple layers in one call for better throughput
2. **Layer Selection**: Use `FL_ZKP_LAYERS=fc1.weight,fc2.weight` to skip small layers
3. **Caching**: Service caches compiled circuits per shape—reuse layer sizes
4. **Monitoring**: Check service logs with `tail -f /tmp/gnark_service.log`
5. **Fallback**: If service fails, set `FL_ZKP_BACKEND=pedersen` to revert

### 🚀 Deployment Options

| Mode | Command | Best For |
|------|---------|----------|
| **Local** | `./gnark_service` | Development |
| **Docker** | `docker run -p 9000:9000 fhe-zkp-service` | Testing |
| **Docker Compose** | `docker compose up` | Multi-service |
| **Production** | Kubernetes + monitoring | Scaling |

---

**Last Updated**: Jan 2024 | **Service Version**: gnark v0.10.0
