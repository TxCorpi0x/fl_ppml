# Blockchain Integration: ZKP, FHE, Federated Learning, and Distributed Data

> **Navigation**: [README.md](README.md) | [FL.md](FL.md) | [FHE.md](FHE.md) | [ZKP.md](ZKP.md) | [DP.md](DP.md)

## Table of Contents

1. [What is a Blockchain?](#1-what-is-a-blockchain)
2. [Core Blockchain Primitives](#2-core-blockchain-primitives)
3. [Smart Contracts and Programmable Trust](#3-smart-contracts-and-programmable-trust)
4. [Consensus Mechanisms](#4-consensus-mechanisms)
5. [Decentralized Data Storage](#5-decentralized-data-storage)
6. [ZKP and Blockchain: Verifiable Computation On-Chain](#6-zkp-and-blockchain-verifiable-computation-on-chain)
7. [FHE and Blockchain: Encrypted Computation Delegation](#7-fhe-and-blockchain-encrypted-computation-delegation)
8. [Federated Learning and Blockchain: Replacing the Central Server](#8-federated-learning-and-blockchain-replacing-the-central-server)
9. [Full-Stack Integration: ZKP + FHE + FL + Blockchain](#9-full-stack-integration-zkp--fhe--fl--blockchain)
10. [Security Model and Threat Analysis](#10-security-model-and-threat-analysis)
11. [Current Systems and Implementations](#11-current-systems-and-implementations)
12. [Challenges and Open Problems](#12-challenges-and-open-problems)
13. [Further Reading](#13-further-reading)

---

## 1. What is a Blockchain?

### The Core Insight

A blockchain is a distributed ledger — a data structure shared and maintained identically across many machines — in which the integrity of the record is guaranteed not by any central authority but by cryptographic links between entries and the economic or computational incentives of the participants who maintain the chain.

The intellectual novelty of blockchain is not in any individual component. Cryptographic hash functions, digital signatures, peer-to-peer networking, and Byzantine fault-tolerant consensus algorithms all predate Bitcoin. What blockchain synthesized was a mechanism for achieving *consensus without coordination*: a set of mutually distrusting participants can agree on a shared state — the ledger — without any single participant controlling, or needing to be trusted to control, the process of agreement.

This property — trustless consensus — is the substrate into which all other technologies discussed in this document are embedded. When we say that ZKP proofs are "verified on-chain," or that FL model updates are "committed to the blockchain," we mean specifically that the verification or commitment is performed by a decentralized consensus mechanism rather than a trusted intermediary, inheriting the ledger's properties of tamper-evidence, transparency, and censorship resistance.

### Positioning Relative to Traditional Systems

A traditional trusted server is a single point of failure, a single point of censorship, and a single point of compromise. Its guarantees are *operational*: the server behaves correctly as long as it functions correctly and the organization running it is honest. If the server is hacked, data is corrupted; if the organization is coerced, records are altered; if the server fails, service stops.

Blockchain replaces this operational trust with *cryptographic* and *economic* trust. The ledger cannot be altered without re-doing the work that anchored it (in proof-of-work systems) or without controlling a supermajority of stake (in proof-of-stake systems). No single party controls the canonical history. Availability is distributed across hundreds or thousands of nodes. These properties come at a cost — throughput, latency, and computational overhead that centralized systems avoid — but for applications where the trust assumptions of a central server are unacceptable, blockchain offers an alternative trust model.

In the context of federated learning, the central server plays the role analogous to a traditional server: it coordinates training, aggregates model updates, and maintains the canonical global model. All of the security concerns about a central FL server — it might be malicious, it might be compromised, it might apply biased aggregation — are exactly the problems blockchain is engineered to remove.

---

## 2. Core Blockchain Primitives

### 2.1 Cryptographic Hash Functions

The foundational primitive of all blockchain systems is the cryptographic hash function $H: \{0,1\}^* \to \{0,1\}^\lambda$, mapping arbitrary-length inputs to fixed-length digests. Hash functions used in production blockchains (SHA-256 in Bitcoin, Keccak-256 in Ethereum) satisfy three properties:

**Pre-image resistance**: Given a digest $d$, it is computationally infeasible to find any input $x$ such that $H(x) = d$. This makes stored digests a commitment to data whose value remains unknown to parties who do not have the data itself.

**Second pre-image resistance**: Given input $x$, it is infeasible to find $x' \neq x$ with $H(x') = H(x)$. Documents cannot be swapped for other documents with the same hash.

**Collision resistance**: It is infeasible to find any pair $(x, x')$ with $x \neq x'$ and $H(x) = H(x')$. This is a stronger property than second pre-image resistance and ensures the uniqueness of digests as identifiers.

These properties enable a hash to serve as a *commitment* to a value: one can publish $H(v)$ as a binding promise to $v$ that reveals nothing about $v$ itself, then later reveal $v$ and allow anyone to verify $H(v)$ matches. This pattern — commit-then-reveal — appears throughout blockchain protocols and in the ZKP-based FL frameworks described in Sections 6 and 9.

### 2.2 Merkle Trees

A Merkle tree is a binary tree of hash values in which every leaf node contains the hash of a data block, and every internal node contains the hash of the concatenation of its two children:

$$\text{node}[i] = H(\text{node}[2i] \,\|\, \text{node}[2i+1])$$

The root of the tree — the **Merkle root** — is a single hash that commits to the entire dataset. Its critical property is enabling **Merkle proofs**: to prove that a particular data block $d_i$ is in the dataset committed to by root $r$, one provides a path of $O(\log n)$ sibling hashes from $d_i$ to the root. The verifier recomputes the path and checks it matches $r$, requiring $O(\log n)$ hash evaluations rather than access to all $n$ data items.

In blockchain contexts, Merkle trees commit to the set of transactions in each block. In FL contexts, Merkle trees commit to the set of client model updates in each training round, enabling efficient proof that a specific client's update was included in the aggregated model without revealing all updates. **Zkfhed** (Zhang et al., 2025) uses Merkle trees to store model commitments on-chain, allowing lightweight inclusion proofs.

### 2.3 Digital Signatures

Each participant in a blockchain network possesses an asymmetric key pair: a private signing key $sk$ and a public verification key $pk$. Transactions are signed using $sk$ and can be verified by any party using $pk$:

$$\sigma = \text{Sign}(sk, m), \quad \text{Verify}(pk, m, \sigma) \in \{\text{accept}, \text{reject}\}$$

The security requirement — existential unforgeability under chosen-message attack (EUF-CMA) — ensures that no adversary without $sk$ can produce a valid signature on any message, even after observing many signatures on messages of their choice. In blockchain, signatures authenticate transactions: "I, the holder of private key $sk$ corresponding to the publicly known address $pk$, authorize transfer of $x$ tokens."

In blockchain-based FL, signatures authenticate client contributions: each client signs its local model update with their on-chain identity, creating a non-repudiation trail that the aggregating smart contract can verify. Clients cannot later deny having submitted a particular update.

### 2.4 The Block Structure

A blockchain is a sequence of blocks $B_1, B_2, \ldots$, where each block $B_t$ contains:

- A **header** including the hash of the previous block $H(B_{t-1})$, a timestamp, and consensus-specific metadata (nonce for PoW, validator signature for PoS)
- A **Merkle root** committing to the block's transactions
- A **body** containing the actual transaction data

The chain of hashes creates the tamper-evident property. If an adversary modifies any transaction in block $B_i$, the Merkle root of $B_i$ changes, which changes $H(B_i)$, which invalidates the pointer from $B_{i+1}$ to $B_i$, cascading to invalidate all subsequent blocks. Re-computing valid blocks for the altered history requires redoing all the work (PoW) or acquiring all the stake (PoS) for every block from $B_i$ forward, which is computationally or economically prohibitive if the chain is long and the adversary does not control a majority of the network's resources.

---

## 3. Smart Contracts and Programmable Trust

### The Concept of Programmable Trust

A **smart contract** is a program deployed to a blockchain whose execution is enforced by the consensus mechanism of the network. Once deployed, a smart contract's code is immutable and its execution is deterministic: given the same inputs, every node in the network computes the same result, and the network's consensus mechanism ensures that the result that gets recorded on-chain is the correct one.

The philosophical significance is that smart contracts transform blockchain from a passive ledger into an *active enforcer*. A traditional contract between two parties requires a legal system — courts, lawyers, enforcement mechanisms — to ensure compliance. A smart contract self-executes automatically when conditions are met, with no trusted intermediary required. The code *is* the law, in the sense that the outcome is determined by the code's logic rather than by any human judgment or institutional process.

### Ethereum and the EVM

The most significant smart contract platform for research is Ethereum, whose Ethereum Virtual Machine (EVM) is a Turing-complete stack machine that executes bytecode uploaded to the blockchain. Programs in Solidity (or Vyper, or Huff) compile to EVM bytecode. Each operation consumes **gas** — a unit of computational cost — and the sender of a transaction pays gas fees proportional to the computation performed.

The gas model has critical implications for privacy-preserving FL. Verifying a Groth16 zk-SNARK proof on the Ethereum mainnet costs approximately 600,000–800,000 gas units. At 2026 gas prices, this translates to substantial cost per verification. Systems like VerifBFL (Bellachia et al., 2025) achieve on-chain verification in under 0.6 seconds precisely by minimizing the on-chain verification work — proofs verified on-chain should be small and verification should be computationally cheap, even if proof generation off-chain is expensive. This is the fundamental design principle of succinct proof systems: **prover-heavy, verifier-light**.

### Smart Contracts in FL Systems

In blockchain-based FL systems, smart contracts serve several distinct roles:

**Coordination and orchestration**: The smart contract defines the FL protocol — announcing training rounds, specifying the global model architecture, selecting participating clients, and enforcing aggregation timing. No human administrator is needed; the protocol is self-executing.

**Payment and incentive enforcement**: Smart contracts can programmatically distribute token rewards to clients whose contributions are accepted, creating Sybil-resistant incentive mechanisms. Clients who submit valid, high-quality updates receive payment automatically; clients who submit invalid updates or attempt to free-ride receive nothing.

**Proof verification**: The smart contract verifies ZKP proofs submitted by clients and aggregators. Only updates accompanied by valid proofs — proving correct training on authorized data following the specified protocol — are accepted into aggregation.

**Aggregation logic**: Lightweight aggregation (weighted averaging of gradient commitments) can be performed on-chain, with the aggregated model stored as a state variable of the smart contract accessible to all clients.

**Dispute resolution**: In adversarial scenarios, smart contracts can adjudicate disputes cryptographically. A client who claims the aggregator cheated can submit the aggregator's alleged input and output along with a proof of the discrepancy; the smart contract evaluates the claim and penalizes the dishonest party automatically.

### Account Abstraction and Role-Based Access

Modern smart contract platforms support **role-based access control** (RBAC) patterns: the contract designates certain addresses as data owners, certain addresses as aggregators, and certain addresses as auditors, with different permissions for each role. In FL contexts:

- **Data owners** (clients): may submit encrypted gradient updates and ZKP proofs for their assigned training rounds
- **Aggregators**: may trigger the aggregation function and post global model updates
- **Auditors**: may read all public commitments and proofs; cannot modify state
- **Contract admin**: may pause the contract in emergency; ideally this role is controlled by a DAO (decentralized autonomous organization) rather than a single key

---

## 4. Consensus Mechanisms

### The Byzantine Generals Problem

The core theoretical challenge blockchain solves is **Byzantine fault tolerance** (BFT): achieving agreement among $n$ nodes when up to $f$ nodes may behave arbitrarily (i.e., Byzantine — lying, selectively communicating, or otherwise deviating from protocol). The classical result (Lamport, Shostak, Pease, 1982) establishes that classical deterministic BFT is achievable if and only if $n > 3f$ (more than two-thirds of nodes are honest).

This is directly relevant to both blockchain consensus and FL aggregation. In an FL system where blockchain enforces the training protocol, the security of FL inherits the security of the consensus mechanism: a majority of validators must be honest for the protocol to maintain integrity.

### Proof of Work (PoW)

In PoW (Bitcoin), validators (*miners*) compete to find a nonce $r$ such that:

$$H(\text{block\_header} \,\|\, r) < T$$

where $T$ is a target threshold set dynamically to achieve a target block time. The first miner to find a valid nonce broadcasts the block; all others discard their work and begin mining the next block on top of it.

**Security model**: Altering the history requires outpacing the honest network in hash computation, infeasible if the attacker controls less than 50% of the network hash rate. The energy consumption of PoW is intentional — it makes history alteration expensive.

**Relevance to FL**: PoW is poorly suited as the consensus mechanism for FL coordination. The energy cost is orthogonal to FL contribution quality, block times (10 minutes for Bitcoin) are too slow for FL training rounds, and the competitive mining process does not naturally integrate model quality verification. Specialized FL consensus mechanisms have been proposed instead.

### Proof of Stake (PoS)

In PoS (Ethereum 2.0, Cardano), validators are selected to propose blocks proportionally to the number of tokens they stake as collateral. Validators who behave dishonestly lose their stake through *slashing* — automated confiscation enforced by the protocol itself.

**Security model**: Altering history requires controlling more than one-third of staked tokens (for finality attacks), or more than one-half (for reversibility attacks). Unlike PoW, attacking is economically expensive because the attacker's stake can be slashed.

**Relevance to FL**: PoS is more energy-efficient and has faster finality than PoW (seconds to minutes vs. hours), making it more practical for FL coordination. The slashing mechanism also provides a template for penalizing misbehaving FL participants who stake tokens.

### Delegated Proof of Stake and Permissioned Chains

**Delegated Proof of Stake (DPoS)**: Token holders elect a fixed set of delegates who produce blocks in rotation. Faster finality and higher throughput than standard PoS, at the cost of some decentralization.

**Permissioned blockchains** (Hyperledger Fabric, Quorum): A known set of pre-authorized validators runs a classical BFT protocol (PBFT, HotStuff, Tendermint). No expensive puzzles or staking required. Throughput is orders of magnitude higher than public chains, at the cost of assuming the validator set is vetted and partially trusted. Appropriate for enterprise FL deployments where the participant organizations are known.

### FL-Specific Consensus Mechanisms

The surveys on blockchain-enabled FL (Rangwala et al., 2025) catalog several FL-specific consensus mechanisms designed to replace PoW/PoS with mechanisms that directly measure FL contribution quality:

**Proof of Federated Learning (PoFL)**: The "mining" puzzle is replaced by performing a training round. The validator who submits the best local model update — measured by loss improvement on a shared validation set — earns the right to propose the next block and receives the block reward. This aligns economic incentives with model quality.

**Proof of Quality (PoQ)**: Validators evaluate submitted model updates using cryptographic quality metrics (loss reduction, gradient norm, cosine similarity to the historical average). Updates that pass the quality threshold are included; others are rejected on-chain. This integrates Byzantine-robust aggregation directly into the consensus layer.

**Proof of Learning (PoL)**: Participants log model state checkpoints during training. The proof that training occurred correctly is the verifiable sequence of parameter updates: if the model evolved along a path consistent with gradient descent on the claimed data, the sequence is valid. Related to the gradient auditing work in Zkfhed (Zhang et al., 2025).

---

## 5. Decentralized Data Storage

### The On-Chain vs. Off-Chain Divide

Storing data directly on a blockchain is expensive. On Ethereum, each byte of storage costs approximately 625 gas (as of 2026), making large-scale on-chain storage prohibitively expensive for datasets or model weights. A 100 MB model stored on Ethereum mainnet would cost thousands of USD at typical gas prices. This creates a fundamental architectural constraint: **blockchain is suitable for commitments and proofs; bulk data must live elsewhere**.

The standard pattern is:
1. Store large data (model weights, training datasets, gradient archives) off-chain in a decentralized storage network
2. Store the *hash* or *commitment* to that data on-chain
3. Use smart contracts to enforce rules about how the data hash transitions — which hash was the model at round $t$, which hashes are valid client submissions

This separation preserves the tamper-evidence and verifiability of blockchain while avoiding its storage costs.

### IPFS: The InterPlanetary File System

The **InterPlanetary File System (IPFS)** is a peer-to-peer content-addressed storage network. Each file is identified by its **Content Identifier (CID)**, which is the cryptographic hash of the file's content. This has a profound property: two parties who store the same file under different names inevitably arrive at the same CID, and anyone can verify the file's integrity by recomputing the CID.

In blockchain-enabled FL:
- Clients publish their encrypted model updates to IPFS, obtaining a CID
- Clients submit the CID (and an optional ZKP proof) to the smart contract on-chain
- The smart contract records the CID commitment and verifies the proof
- The aggregator retrieves all submitted updates from IPFS by CID, aggregates them, and publishes the new global model to IPFS
- The aggregator submits the new global model's CID to the smart contract, which verifies it and records it as the canonical model for the next round

This pattern — **IPFS for bulk storage, blockchain for commitments and coordination** — is the dominant architecture in production BCFL systems. It achieves the tamper-evidence of blockchain without the storage cost, while remaining decentralized: IPFS has no single point of failure, and files persist as long as any node is pinning them.

### Filecoin and Storage Incentives

IPFS alone does not guarantee persistence. Files are only available as long as nodes choose to store them. **Filecoin** adds an economic layer: storage providers commit to storing files for specified durations and provide cryptographic *proofs of storage* (PoRep — Proof of Replication, PoSt — Proof of Spacetime) that they are actually holding the data. These proofs are verified on the Filecoin blockchain, and providers are paid in FIL tokens for storage service.

In FL contexts, Filecoin enables persistent, verifiable storage of training artifacts — model snapshots, training logs, gradient archives — with economic guarantees that the data will remain available for the specified retention period.

### Distributed Hash Tables and Kademlia

Both IPFS and many blockchain P2P networks use the **Kademlia** distributed hash table (DHT) for peer discovery and content routing. Kademlia organizes peers in a 160-bit ID space and routes queries in $O(\log n)$ hops to the peer whose ID is closest to a target key. Nodes maintain $k$-buckets of peers at various distances, providing resilience to churn (nodes leaving and rejoining the network).

Understanding DHT routing is important for security analysis of decentralized FL: an adversary controlling a significant fraction of peers can execute *eclipse attacks* — surrounding a target client with adversary-controlled peers and controlling all information the target receives. Eclipse attacks enable various FL poisoning attacks; cryptographic measures (signed model CIDs, on-chain commitment verification) are the primary defenses.

### Trusted Execution Environments as Off-Chain Data Processors

**Trusted Execution Environments (TEEs)** — Intel SGX, AMD SEV, ARM TrustZone — are hardware-isolated computation environments that provide confidentiality and integrity guarantees for in-enclave computation. TEEs can serve as off-chain processors for FL aggregation, handling computations too expensive for the EVM while providing hardware attestation that the computation was performed correctly.

In a TEE-assisted BCFL architecture:
- The TEE generates an **attestation report** — a hardware-signed statement asserting that a specific program is running in an authenticated enclave
- The blockchain smart contract verifies the attestation and trusts the TEE's output
- The TEE performs aggregation on encrypted gradients, decrypts within the enclave, and re-encrypts the aggregated model

TEEs are an alternative to ZKP-based verification: rather than proving computation was correct via cryptography, they prove it via hardware attestation. They are faster than ZKP verification but depend on the hardware vendor's security guarantees, which have suffered high-profile vulnerabilities (Spectre, Meltdown, Plundervolt).

---

## 6. ZKP and Blockchain: Verifiable Computation On-Chain

### The Fundamental Complementarity

ZKP and blockchain are naturally complementary technologies that address each other's weaknesses. Blockchain provides *persistent, public, tamper-evident ledger* and *decentralized consensus* but cannot perform private computation. ZKPs provide *verifiable private computation* — proofs that a computation was performed correctly on private data without revealing the data — but require a trusted verifier. When the ZKP verifier is a smart contract on a blockchain, the trust requirement is removed: the verification is performed by decentralized consensus, trusting no individual party.

The combination achieves something neither can alone: **publicly verifiable private computation**. Anyone can verify that a computation on private data produced a specific result, without learning anything about the private data, without trusting any individual party to have performed the verification honestly.

This is the cryptographic foundation for blockchain-based FL with ZKP: clients can prove their training was correct, on authorized data, following the specified protocol, and this proof can be verified by a smart contract that all parties trust by construction.

### zk-SNARK Proof Verification On-Chain

A zk-SNARK (Zero-Knowledge Succinct Non-Interactive Argument of Knowledge) proof has the crucial property of being **succinct**: the proof size and verification time are constant (or logarithmic) in the size of the computation, regardless of how complex the computation itself is. This makes on-chain verification practical.

For a Groth16 proof (the most widely deployed zk-SNARK system), the proof consists of three elliptic curve points $(\pi_A, \pi_B, \pi_C)$. Verification requires one pairing computation:

$$e(\pi_A, \pi_B) = e(\alpha, \beta) \cdot e(\sum_i a_i \gamma_i, \gamma) \cdot e(\pi_C, \delta)$$

where $e: \mathbb{G}_1 \times \mathbb{G}_2 \to \mathbb{G}_T$ is a bilinear pairing, $a_i$ are the public inputs, $\alpha, \beta, \gamma, \delta$ are elements of the structured reference string (SRS), and $\gamma_i$ are precomputed verification key elements. This computation is deterministic and takes a fixed amount of gas regardless of what circuit the proof is for.

The Ethereum EVM includes precompiled contracts (EIP-197) for BN254 elliptic curve operations and pairings, enabling efficient Groth16 verification. A complete Groth16 verifier in Solidity costs approximately 600,000–800,000 gas. At 2025–2026 gas prices, this ranges from a few cents to a few dollars per proof, making it economically feasible at the scale of FL training rounds (one proof per client per round).

### Proof of Computation: The Circuit Analogy

To generate a ZKP, the computation being proved must be expressed as an **arithmetic circuit** — a directed acyclic graph of addition and multiplication gates over a prime field $\mathbb{F}_p$. Every operation in the computation (gradient computation, loss evaluation, matrix multiplication) must be encoded as field operations.

For ML computations, this presents a fundamental tension:
- **Floating-point arithmetic** used in standard neural network training does not map cleanly to finite field arithmetic
- **Non-linear activations** (ReLU, sigmoid, softmax) require polynomial approximations or lookup tables in the circuit
- **Division and comparison** operations require auxiliary witnesses (bit decompositions, range proofs)

Current research approaches this tension in several ways:
- **Quantization**: Represent all model parameters as fixed-point integers over the prime field, eliminating floating-point
- **Polynomial approximations**: Replace $\text{ReLU}(x) = \max(0, x)$ with a polynomial approximation valid in the expected input range
- **Lookup arguments**: PLOOKUP, Lasso, and similar techniques enable efficient proofs of lookups in precomputed tables, applicable to activation functions
- **Selective proving**: Only prove the parts of computation most critical for security (e.g., gradient norm bounds, loss below threshold) rather than the full training procedure

### Incrementally Verifiable Computation (IVC)

Training a neural network involves thousands of gradient descent steps. Proving all of them with a monolithic zk-SNARK would require a circuit of impractical size. **Incrementally Verifiable Computation (IVC)** addresses this by enabling a proof that an iterated computation was performed correctly, where each step's proof builds on the previous step's proof.

Formally, IVC enables proofs of the statement "I started from state $z_0$ and applied function $F$ exactly $n$ times, arriving at state $z_n$" — where $z_i = F(z_{i-1})$ for each step. The proof has constant size regardless of $n$.

The key IVC constructions used in 2025 systems are:
- **NOVA** (Kothapalli et al., 2022): Uses *relaxed R1CS* and folding schemes to achieve highly efficient IVC with minimal per-step overhead
- **SuperNova/HyperNova**: Extends NOVA to handle varying per-step computations, important for training where the computation may differ across iterations
- **Recursive SNARKs**: Use a SNARK to prove the verification of a previous SNARK, enabling proof composition

**VerifBFL** (Bellachia et al., 2025) employs IVC to produce proofs for multiple local training steps without circuit size explosion, achieving practical proof generation times under 81 seconds for full local training.

### ZKP for FL-Specific Computations

Several FL-specific proofs have been developed and deployed in recent systems:

**Data Provenance Proofs**: Prove that training data was drawn from an authorized dataset (one whose Merkle root is registered on-chain) without revealing which specific samples were used. The prover commits to a random subset of their data, provides Merkle inclusion proofs for each sample, and proves that the gradient computation used samples consistent with the commitments.

**Training Correctness Proofs**: Prove that local gradient updates are the result of running a specific training procedure (gradient descent with specified learning rate and batch size) on the committed data. This prevents clients from submitting fabricated updates.

**Gradient Bound Proofs**: Prove that the $\ell_2$-norm of a gradient update satisfies $\|\nabla\|_2 \leq C$ for a clipping bound $C$ — the same operation performed in DP-SGD — without revealing the gradient. This is a range proof over field elements.

**Loss Threshold Proofs**: Prove that local validation loss is below a threshold $\tau$ without revealing the actual loss value. **ZKP-FedEval** (Commey et al., 2025) implements this: clients prove $\mathcal{L}_{\text{local}} \leq \tau$ using Circom circuits, enabling verifiable evaluation without metric leakage.

**Client Selection Proofs**: Prove eligibility for participation (data freshness, hardware capability, stake deposit, accuracy certificate) without revealing private attributes. **Veri-CS-FL** (Wang et al., 2025) uses this to implement verifiable client selection.

---

## 7. FHE and Blockchain: Encrypted Computation Delegation

### The Delegation Problem

A fundamental limitation of smart contracts is that they execute publicly: all inputs and all intermediate state are visible to all validators (and all observers) on a transparent blockchain. For FL, this means that model updates submitted to a smart contract would be visible to everyone, defeating the purpose of keeping local data private.

One approach is to maintain gradient privacy by having clients encrypt updates before submitting them, with the smart contract performing aggregation on ciphertexts. This requires the aggregation operation to be computable on encrypted values — exactly what FHE provides.

**Fully Homomorphic Encryption (FHE)** allows arbitrary computation on ciphertexts without decryption. The central property:

$$\text{Eval}(f, \text{Enc}(x_1), \ldots, \text{Enc}(x_k)) = \text{Enc}(f(x_1, \ldots, x_k))$$

For FL aggregation, $f$ is the weighted averaging function, and $x_1, \ldots, x_k$ are the local gradient updates from $k$ clients. The smart contract (or an off-chain aggregator whose output is committed on-chain) computes $\text{Enc}(\bar{\nabla}) = \text{Enc}(\frac{1}{k}\sum_i \nabla_i)$ from the submitted ciphertexts, without learning any individual $\nabla_i$.

### CKKS for Gradient Aggregation

The **CKKS** (Cheon-Kim-Kim-Song) scheme, implemented in TenSEAL and Microsoft SEAL, is the most practical FHE scheme for FL. CKKS natively supports approximate arithmetic over real or complex numbers, encoding vectors of floating-point values as polynomial plaintext. This matches the structure of gradient vectors directly.

The key FHE operations for FL aggregation are:
- **Homomorphic addition**: $\text{Enc}(\nabla_i) + \text{Enc}(\nabla_j) = \text{Enc}(\nabla_i + \nabla_j)$ — used to sum gradient ciphertexts
- **Scalar multiplication**: $s \cdot \text{Enc}(\nabla_i) = \text{Enc}(s \cdot \nabla_i)$ — used to apply per-client weights
- **Rotation**: Shift elements within a ciphertext polynomial, used to sum across the ciphertext slots for global pooling

The aggregation protocol for $k$ clients with CKKS:
1. Each client encrypts their gradient vector under the server's public key: $c_i = \text{Enc}_{pk}(\nabla_i)$
2. The aggregator computes $c_{\text{agg}} = \frac{1}{k} \sum_{i=1}^k c_i$, a sequence of homomorphic additions and a scalar multiplication
3. The aggregator decrypts $\nabla_{\text{agg}} = \text{Dec}_{sk}(c_{\text{agg}})$
4. The decrypted aggregate is the FedAvg result, computed without any party learning individual gradients

On blockchain, the ciphertext $c_i$ and its hash $H(c_i)$ are committed on-chain. The commitment ensures that the ciphertext submitted for aggregation cannot be swapped after the fact.

### The Key Management Problem

The FHE decryption key $sk$ must be held by someone or something. This creates a trust re-centralization problem: if a single aggregator holds $sk$, they can decrypt individual gradients by simply decrypting each ciphertext before adding. The blockchain commitment prevents ciphertext substitution but not decryption.

Several architectures address this:

**Threshold FHE**: The decryption key is secret-shared among $n$ parties such that decryption requires cooperation of at least $t$ of them ($t$-of-$n$ threshold). No single party can decrypt alone. Clients can even serve as decryption share holders, requiring a quorum of them to decrypt the aggregated result while individual gradients remain protected.

**On-chain aggregation with TEE**: The aggregator runs inside a TEE that provably destroys $sk$ after aggregation. The TEE's attestation vouches for this. The blockchain verifies the attestation and records only the decrypted aggregate.

**Homomorphic Encryption Delegation Learning (HEDL)**: Proposed in **Zkfhed** (Zhang et al., 2025), HEDL allows clients to outsource FHE computation to external computing providers. Clients encrypt their data under their own key, generate a switching key that allows the provider to re-encrypt under the server's aggregation key, and the provider performs the HE computation. The provider never receives the plaintext; the decryption key never leaves the client. This separates computation from key ownership at the cost of an additional re-encryption operation.

**Multi-Key FHE**: Each client encrypts under their own public key; the aggregator performs computation on ciphertexts encrypted under different keys using multi-key homomorphic operations. Decryption requires all parties' partial decryptions, so no subset of parties can decrypt individual contributions.

### FHE on Blockchain: Performance Constraints

FHE ciphertexts are large. A CKKS ciphertext encoding a 1,024-element vector at 128-bit security has a size of approximately 50–200 KB depending on polynomial degree and coefficient modulus. Uploading ciphertexts to Ethereum directly is prohibitively expensive; IPFS is used for ciphertext bulk storage, with on-chain commitments to the CIDs.

EVM execution of FHE operations is currently impractical: a single CKKS multiplication on-chain would require hundreds of millions of gas. The emerging solution is FHE co-processors — L2 rollups or application-specific blockchains that natively support FHE operations at the opcode level. Projects like **Zama's fhEVM** deploy a modified EVM where encrypted values are first-class types and FHE operations are precompiled, reducing their cost by orders of magnitude. In this architecture, the FHE aggregation logic is expressed as a Solidity contract on the fhEVM, with all the programmability of smart contracts, but executing over encrypted state.

---

## 8. Federated Learning and Blockchain: Replacing the Central Server

### The Central Server as a Trust Bottleneck

Classical FL depends on a central aggregation server that:
1. Selects clients to participate in each round
2. Distributes the current global model
3. Receives client updates and aggregates them
4. Applies the aggregated update to maintain the global model
5. Decides when training is complete and distributes the final model

All five functions require trusting the server. A compromised or dishonest server can: select only complicit clients; distribute a backdoored model; ignore legitimate updates; apply a different aggregation function; pretend training has converged when it has not. The server is simultaneously a single point of failure and a single point of attack.

Blockchain decentralizes all five functions: client selection is handled by smart contract logic executed by consensus; model distribution is via IPFS with CID committed on-chain; updates are submitted to the smart contract; aggregation is either performed on-chain or by an attested off-chain aggregator whose output is committed on-chain; convergence criteria are encoded in the contract.

### BCFL Architecture Taxonomy (Rangwala et al., 2025)

The most systematic contemporary taxonomy of blockchain-enabled FL identifies four independent architectural dimensions:

**Coordination Structure**:
- *On-chain coordination*: Smart contract selects clients, manages rounds, and records all state. Fully decentralized but limited by EVM throughput.
- *Off-chain coordination with on-chain settlement*: A coordinator handles round management off-chain; final results and proofs are settled on-chain. Faster but reintroduces the coordinator as a partial trust assumption.
- *Hybrid*: Critical decisions (final model checkpoint, dispute resolution) are settled on-chain; routine coordination is off-chain.

**Consensus Mechanism**: PoW, PoS, DPoS, or FL-specific mechanisms (PoFL, PoQ) as described in Section 4.

**Storage Architecture**:
- *Full on-chain*: Only feasible for very small models or when using layer-2 solutions with cheap storage.
- *IPFS / Filecoin*: Dominant pattern for production systems.
- *Hybrid*: Commitments on-chain, bulk data on IPFS.

**Trust Model**:
- *Permissionless*: Any participant may join; identity is a blockchain address; Sybil resistance via staking.
- *Permissioned*: Participants are pre-authorized; identity is verified off-chain; classical BFT consensus.
- *Consortium*: Hybrid where a consortium of known organizations runs the validators, and any participant from within the consortium may act as a client.

### Model Aggregation on Blockchain

**On-chain aggregation** using FedAvg:

$$\theta^{t+1} = \frac{\sum_{k \in S^t} n_k \theta_k^{t+1}}{\sum_{k \in S^t} n_k}$$

For this computation to occur on-chain, all $\theta_k^{t+1}$ must either be uploaded to chain (expensive) or committed via CID while the aggregation is performed off-chain by an attestable aggregator.

**Aggregation smart contract** (conceptual Solidity pseudocode):

```solidity
contract FedLearningAggregator {
    mapping(uint => mapping(address => bytes32)) public updateCIDs;  // round => client => IPFS CID
    mapping(uint => bytes32) public globalModelCIDs;  // round => global model CID
    mapping(address => bool) public registeredClients;

    function submitUpdate(uint round, bytes32 cid, bytes calldata zkProof) external {
        require(registeredClients[msg.sender], "Not registered");
        require(verifyProof(zkProof, cid, round), "Invalid ZKP");
        updateCIDs[round][msg.sender] = cid;
        emit UpdateSubmitted(round, msg.sender, cid);
    }

    function finalizeRound(uint round, bytes32 aggregatedModelCID, bytes calldata aggProof) external {
        require(isAggregator(msg.sender), "Not aggregator");
        require(verifyAggregationProof(aggProof, round, aggregatedModelCID), "Invalid aggregation proof");
        globalModelCIDs[round + 1] = aggregatedModelCID;
        emit RoundFinalized(round, aggregatedModelCID);
    }
}
```

Each `submitUpdate` call verifies a ZKP that the submitted update was computed correctly on authorized data; `finalizeRound` verifies a proof that the aggregation followed the FedAvg protocol correctly.

### Byzantine Robustness via On-Chain Verification

Classical Byzantine-robust aggregators (Krum, Trimmed Mean, FLTrust) require computing pairwise distances or norms across submitted gradients. For several recent systems, these operations are performed off-chain by the aggregator and proved correct via ZKP:

1. The aggregator computes pairwise cosine similarities between submitted gradients
2. The aggregator runs Krum selection: eliminates the update most different from its neighbors
3. The aggregator generates a ZKP proving that the elimination was performed correctly per protocol
4. The smart contract verifies the proof and accepts only the post-filtering aggregate

This approach — **provably Byzantine-robust aggregation** — closes a critical gap in blockchain FL. Without proofs, an on-chain system must trust the aggregator's claim that it applied a robust aggregation rule. With proofs, the smart contract enforces it.

---

## 9. Full-Stack Integration: ZKP + FHE + FL + Blockchain

### The Architecture Space

The four technologies compose along multiple axes, and different systems make different architectural choices depending on their threat model and performance requirements. The general composition can be described systematically:

**Layer 1 — Data Privacy (FHE)**: Client gradient updates are encrypted using FHE (CKKS for approximate arithmetic, TFHE for binary computations). Encrypted updates are uploaded to IPFS. Only ciphertext hashes are posted on-chain.

**Layer 2 — Computational Integrity (ZKP)**: Clients generate ZKP proofs asserting that (a) their ciphertext encodes a gradient legitimately computed on authorized data, (b) the gradient norm satisfies bounds preventing data leakage, and (c) training followed the specified protocol. Proofs are verified on-chain by the smart contract.

**Layer 3 — Aggregation and Coordination (Blockchain)**: A smart contract coordinates training rounds, verifies proofs, records model commitments, and arbitrates disputes. The aggregated model is committed as an IPFS CID on-chain, forming an immutable, auditable training history.

**Layer 4 — Verifiable Aggregation**: The aggregator (which may be a TEE, a threshold committee, or computed via multi-party computation) proves correct aggregation and posts the proof on-chain. The contract verifies that the aggregated model is the correct output of the FedAvg (or other) algorithm applied to the accepted client updates.

**Layer 5 — Post-Aggregation Privacy (DP)**: After aggregation, calibrated Gaussian or Laplace noise is added to the decrypted aggregate, providing differential privacy guarantees even to parties who observe the final model. The DP noise application can itself be proved correct via ZKP.

### The Full-Stack Protocol

A complete training round in a ZKP + FHE + FL + Blockchain system proceeds as follows:

**Round Announcement (Blockchain → Clients)**:
- Smart contract announces round $t$, specifying global model CID $c_{\text{model}}^t$, client selection criteria, ZKP circuit hash (commits to what training procedure will be accepted), and round deadline

**Local Training (Client)**:
1. Client retrieves global model from IPFS by CID $c_{\text{model}}^t$ and verifies its hash
2. Client trains locally on its private dataset $\mathcal{D}_k$ for $E$ epochs, computing local parameters $\theta_k^{t+1}$
3. Client computes gradient update $\Delta_k^t = \theta_k^{t+1} - \theta^t$ and clips: $\hat{\Delta}_k^t = \Delta_k^t / \max(1, \|\Delta_k^t\|_2 / C)$
4. Client encrypts update under aggregation public key: $c_k = \text{CKKS.Enc}_{pk}(\hat{\Delta}_k^t)$
5. Client generates ZKP proof $\pi_k$ asserting: data membership (Merkle proof), gradient computation correctness (IVC proof over training steps), and gradient norm bound ($\|\hat{\Delta}_k^t\|_2 \leq C$)
6. Client uploads $c_k$ to IPFS, obtaining CID $\text{cid}_k$
7. Client submits $(\text{cid}_k, H(c_k), \pi_k)$ to smart contract

**On-Chain Verification (Smart Contract)**:
- For each client submission $(k, \text{cid}_k, H(c_k), \pi_k)$:
  - Verify $\pi_k$ against the public inputs (round $t$, client identity $k$, gradient CID hash $H(c_k)$, global model CID $c_{\text{model}}^t$)
  - Record $c_k$'s CID in the accepted-updates mapping for round $t$
  - Mark client $k$ as having submitted for round $t$

**FHE Aggregation (Off-Chain Aggregator with On-Chain Commitment)**:
1. Aggregator retrieves all accepted ciphertexts from IPFS by CID
2. Aggregator computes $c_{\text{agg}}^t = \frac{1}{|S^t|} \sum_{k \in S^t} c_k$ using homomorphic addition and scalar multiplication
3. Aggregator decrypts: $\hat{\Delta}_{\text{agg}}^t = \text{CKKS.Dec}_{sk}(c_{\text{agg}}^t)$
4. Aggregator applies DP noise: $\tilde{\Delta}_{\text{agg}}^t = \hat{\Delta}_{\text{agg}}^t + \mathcal{N}(0, \sigma^2 I)$
5. Aggregator updates global model: $\theta^{t+1} = \theta^t + \tilde{\Delta}_{\text{agg}}^t$
6. Aggregator uploads $\theta^{t+1}$ to IPFS, obtaining $c_{\text{model}}^{t+1}$
7. Aggregator generates ZKP $\pi_{\text{agg}}$ proving aggregation was computed correctly over the set of accepted CIDs
8. Aggregator submits $(c_{\text{model}}^{t+1}, \pi_{\text{agg}})$ to smart contract

**Round Finalization (Smart Contract)**:
- Verifies $\pi_{\text{agg}}$ (ZKP that aggregation over accepted CIDs produced model $c_{\text{model}}^{t+1}$)
- Records $c_{\text{model}}^{t+1}$ as the new global model
- Distributes rewards to clients whose updates were accepted
- Announces round $t+1$

This protocol provides:
- **Gradient privacy**: FHE ensures no party learns individual gradients
- **Training integrity**: ZKP proofs bind each ciphertext to correct local training
- **Aggregation integrity**: ZKP proof from aggregator binds global model to accepted updates
- **Tamper-evidence**: Blockchain commits to the entire audit trail
- **Decentralization**: No trusted coordinator; all enforcement is by smart contract

### Privacy Budget Accounting On-Chain

Differential privacy requires tracking the cumulative privacy budget $\varepsilon$ across training rounds. In a decentralized setting, this accounting must be transparent and tamper-proof. Blockchain provides this naturally: the smart contract maintains the running privacy budget state:

$$\varepsilon_{\text{total}}^t = \varepsilon_{\text{total}}^{t-1} + \Delta\varepsilon^t$$

where $\Delta\varepsilon^t$ is the per-round privacy cost computed via Rényi Differential Privacy accounting (Mironov, 2017). The smart contract enforces a maximum budget $\varepsilon_{\max}$; once exhausted, training terminates automatically and provably.

This on-chain DP budget tracking solves a significant governance problem in deployed FL systems: if the organization running FL claims to apply DP with $\varepsilon = 1$, there is no way to independently verify this claim. With on-chain accounting, the DP parameters are public, the accounting is transparent, and the enforcement is automatic.

---

## 10. Security Model and Threat Analysis

### Threat Model Classification

Security analysis of ZKP + FHE + FL + Blockchain systems must specify the adversary's capabilities precisely. The standard threat model in the literature considers:

**Adversary type**:
- *Semi-honest (honest-but-curious)*: Follows the protocol but attempts to infer private information from observed messages
- *Malicious (Byzantine)*: May deviate arbitrarily from the protocol, sending fabricated messages, selectively participating, or coordinating with other malicious parties
- *Adaptive*: May choose which parties to corrupt dynamically, based on observed messages

**Corruption bound**: What fraction of clients and which protocol roles (clients, aggregator, validators) the adversary may control. Standard assumption: adversary controls at most $f < n/3$ Byzantine nodes in the consensus layer, and at most $f < n/2$ malicious clients.

**Computational model**: Standard cryptographic hardness assumptions — LWE (Learning With Errors), discrete logarithm, random oracle model for hash functions.

### Attack Surface Analysis

**Sybil attacks**: An adversary creates many fake identities (blockchain addresses) to gain disproportionate influence. Defenses: staking deposits (economic Sybil resistance), on-chain identity verification, permissioned participant lists.

**Model poisoning**: Malicious clients submit crafted updates designed to degrade model performance or embed backdoors. Defenses: ZKP training correctness proofs (cannot submit fabricated gradients); Byzantine-robust aggregation proved correct on-chain; anomaly detection with ZKP-verified filtering scores.

**Data poisoning**: Malicious clients train on curated datasets designed to influence the model. Defenses: data provenance proofs (Merkle inclusion for authorized datasets); data quality ZKPs; DP noise masking data-specific signals.

**Gradient inversion**: Adversary reconstructs training data from gradient updates. Defenses: FHE (gradients are never transmitted in plaintext); DP noise; gradient compression; mixing (batched encryption across multiple rounds).

**Aggregator malice**: Aggregator applies incorrect aggregation, drops legitimate updates, or steals the decryption key. Defenses: aggregation ZKP proving correct FedAvg over committed inputs; threshold decryption (aggregator cannot decrypt alone); aggregator slashing if incorrect proof is submitted.

**Eclipse attacks** on the P2P layer: Adversary surrounds a target client with controlled peers, feeding them a stale or corrupted global model. Defenses: Model hash verification (client checks against on-chain CID); content-addressed retrieval (IPFS CID cannot be forged); multi-source model retrieval.

**Smart contract vulnerabilities**: Reentrancy, integer overflow, access control bugs. Defenses: formal verification of contract code (Certora, Echidna); upgradeable contracts with timelocked governance; multi-sig admin keys.

**Quantum adversaries**: zk-SNARKs based on elliptic curve pairings are broken by Shor's algorithm on a sufficiently powerful quantum computer. LWE-based FHE is believed quantum-secure. Post-quantum ZKP systems (lattice-based SNARGs, STARKs) exist but are less mature. The hybrid nature of ZKP + FHE systems creates asymmetric quantum vulnerability — FHE survives quantum attacks but ZKP verification does not, unless STARKs are used.

### Formal Security Definitions

The security of a full-stack ZKP + FHE + FL + Blockchain system is typically formalized via **simulation-based security**: the real protocol is secure if no adversary in the real world can distinguish its view from a simulated view produced by an ideal functionality that has access only to the final model output.

The ideal FL functionality $\mathcal{F}_{\text{FL}}$:
- Receives private datasets $\mathcal{D}_1, \ldots, \mathcal{D}_k$ from clients
- Computes $\theta^* = \arg\min_\theta \sum_k \frac{n_k}{n} F_k(\theta)$
- Outputs $\theta^*$ to all parties

A protocol $\Pi$ realizes $\mathcal{F}_{\text{FL}}$ if for every PPT adversary $\mathcal{A}$ attacking $\Pi$, there exists a PPT simulator $\mathcal{S}$ such that the adversary's view in a real execution of $\Pi$ is computationally indistinguishable from a simulated execution against $\mathcal{F}_{\text{FL}}$.

**VerifBFL** (Bellachia et al., 2025) provides a formal security proof in this model under the ZK and DP assumptions. **Zkfhed** (Zhang et al., 2025) provides security analysis under active adversaries controlling a minority of clients.

---

## 11. Current Systems and Implementations

### VerifBFL (2025) — Trustless ZK-Blockchain FL

**Reference**: Bellachia, Bouchiha, Ghamri-Doudane, Rabah — arXiv:2501.04319 — NOMS'25

The first FL system to use **Incrementally Verifiable Computation (IVC)** for local training proofs combined with on-chain verification. Architecture:

- *Proof of training*: IVC proof (NOVA-based) over all local gradient descent steps; proof size is constant regardless of training duration
- *Proof of aggregation*: zk-SNARK over the aggregation computation
- *Blockchain*: Smart contracts verify both proofs; aggregation proofs cost ~0.6s on-chain
- *Privacy*: Differential privacy protects training data from inference
- *Benchmark*: Local training proof generation < 81s; aggregation proof < 2s; on-chain verification < 0.6s

This is the most complete demonstration of the IVC-based FL paradigm, with practical benchmarks showing the approach is within range of production feasibility.

### Zkfhed (2025) — ZKP + FHE + Blockchain, TKDE

**Reference**: Zhang, Lu, Wu, Ren (NUS), Zhu — IEEE TKDE, Vol. 37, No. 6, 2025. DOI: 10.1109/TKDE.2025.3550546

The system most directly analogous to this project's `he_tenseal_zkp` and `he_concrete_tfhe_zkp` modes. Architecture:

- *Two-stage ZKP audit*: Stage 1 proves data provenance (training data from trusted organizations); Stage 2 proves training protocol compliance (computations follow the specified procedure)
- *HEDL (HE Delegation Learning)*: FHE-based computation outsourcing; clients encrypt under their key, switching key allows re-encryption under aggregation key; provider computes without accessing plaintext
- *Blockchain coordination*: Smart contracts coordinate rounds; Merkle trees store model commitments; slashing for proven misbehavior
- *DP post-processing*: Gaussian noise applied to decrypted aggregate
- *Validation*: Malicious client detection demonstrated on real-world benchmarks

### SoK: Verifiable FL (2025) — Systematization

**Reference**: Bruschi, Esposito, Gagliardoni, Rizzini (Horizen Labs / Politecnico di Milano) — IACR ePrint 2025/2296

The comprehensive academic survey of the field. Key findings:

- Current ZKP systems can prove individual linear layers efficiently but struggle with non-linear activations due to circuit complexity
- IVC and NOVA-based approaches are the most promising for proving multi-step training
- Blockchain integration introduces latency that dominates FL round time; off-chain proof generation with on-chain verification is necessary
- No existing system provides full composable security proofs for ZKP + FHE + FL + Blockchain; this remains an open problem

### ZK-FL Framework (2025) — IEEE Communications Magazine

**Reference**: Wang et al. — arXiv:2503.15550 — IEEE Communications Magazine (accepted Jan 2026)

Provides a taxonomy of ZKP roles across FL stages. The **Veri-CS-FL** (Verifiable Client Selection FL) contribution is significant: clients generate ZKP proofs of their local model performance, enabling the server/smart contract to perform verifiable client selection without accessing raw model updates. This extends the ZKP-in-FL paradigm to the pre-training phase (client selection) as well as training and aggregation.

### ZKP-FedEval (2025) — Verifiable Evaluation

**Reference**: Commey, Appiah, Klogo, Crosby — arXiv:2507.11649

The only published system addressing ZKP-based FL evaluation. Implementation uses **Circom** for circuit design and Groth16 for proof generation. Key contribution: evaluation phase metrics (loss values, accuracy) can themselves be privacy-sensitive and require protection. Threshold-based proofs ("my loss is below $\tau$") prevent adversarial inference from aggregated evaluation reports.

### Blockchain-Enabled FL Survey (2025) — BCFL Taxonomy

**Reference**: Rangwala, Venugopal, Buyya — arXiv:2508.06406

Introduces the **TrustMesh** case study: an IoT-focused BCFL framework that integrates smart-contract-based reputation scoring with FL training. Clients build reputation over time based on proof-verified training quality; high-reputation clients are selected more frequently. Reputation is stored on-chain, providing a transparent, tamper-evident record of each client's historical contribution quality.

### Zama fhEVM — FHE-Native Blockchain

**Reference**: Zama AI (2024–2025) — https://github.com/zama-ai/fhevm

Not a FL system but a critical infrastructure component. Zama's **fhEVM** modifies the EVM to treat encrypted values as first-class types. Solidity contracts can operate on `euint32` (encrypted uint32) and `ebool` (encrypted boolean) values, with all operations implemented via TFHE under the hood. This enables FL aggregation logic to be written directly in Solidity while operating over encrypted gradients, removing the need for an off-chain FHE aggregator. As of 2025–2026, the fhEVM has deployed on several testnets and represents the next architectural direction for on-chain private FL.

---

## 12. Challenges and Open Problems

### Circuit Complexity of Neural Networks

The most significant practical barrier to ZK-FL is the size of arithmetic circuits representing neural network computations. A single forward pass of a ResNet-50 requires approximately $4 \times 10^9$ floating-point operations. Even with quantization, representing this as an R1CS circuit produces constraints on the order of $10^{10}$, making proof generation time prohibitive with current tools. This limits verified FL to smaller models (CNN on MNIST, small LLMs).

Current research directions:
- **Lookup tables and logup arguments**: Represent non-linear activations as table lookups rather than polynomial evaluations; recent PlonKish/HyperPlonk backends achieve this efficiently
- **Proof recursion**: Prove execution of one layer at a time, recursively composing proofs, rather than proving the entire network in a monolithic circuit
- **Approximate proofs**: Relax soundness from cryptographic to statistical, accepting small error in exchange for dramatically smaller circuits

### Latency vs. Decentralization Trade-Off

A fundamental tension exists between decentralization and performance. A fully on-chain FL system inherits blockchain's throughput limits: Ethereum processes approximately 15–100 transactions per second (higher on L2 rollups). FL training rounds typically require one transaction per client; with 1,000 clients, a single round requires at least 1,000 transactions, taking minutes at mainnet throughput and potentially longer at peak congestion.

Layer-2 solutions (Optimistic rollups, ZK rollups) improve throughput but add latency for finality. Application-specific blockchains (app-chains) with custom consensus offer further improvement but sacrifice security from the main chain's validator set.

No single architecture dominates across all FL settings. The appropriate choice depends on the number of clients (cross-device vs. cross-silo), acceptable latency, required security level, and economic constraints.

### Composable Security Proofs

The individual components — ZKP, FHE, blockchain consensus, DP — each have well-understood security definitions. However, formally proving the security of their composition remains an open problem. The **Universal Composability (UC) framework** (Canetti, 2001) provides a theoretical foundation for proving that protocol components combine securely when composed, but applying it to the full ZKP + FHE + FL + Blockchain stack requires careful specification of each component's ideal functionality and their interaction interfaces.

The **SoK: Verifiable FL** (Bruschi et al., 2025) explicitly identifies the absence of composable security proofs as a critical open problem and calls for standardized ideal functionalities for each FL phase to enable modular security analysis.

### Gas Cost of Proof Verification

On Ethereum mainnet, verifying a Groth16 proof costs 600,000–800,000 gas per proof. With 100 clients submitting proofs per round, a single FL round costs 60–80 million gas — approximately 0.1 ETH at typical 2025 prices, or roughly $300–$400 per training round. For 100 training rounds, this is $30,000–$40,000 in pure gas costs, plus storage costs for IPFS pinning.

For healthcare or financial applications where compliance value justifies the cost, this may be acceptable. For general-purpose FL, it remains prohibitive. Ongoing research on more efficient proof systems (PLONK, STARK, Lasso) and L2 rollups continue to improve this.

### Post-Quantum Security

The dominant ZKP systems in 2025 (Groth16, PLONK) rely on elliptic curve pairings, which are vulnerable to Shor's algorithm on quantum computers. FHE schemes based on LWE (CKKS, TFHE) are believed to be quantum-secure. A quantum-capable adversary could break the ZKP layer of a ZKP + FHE + FL system while the FHE layer remains secure, undermining proof verifiability.

**STARKs** (Scalable Transparent Arguments of Knowledge) are plausibly post-quantum (based on collision-resistant hash functions rather than elliptic curves) and are already deployed in production (StarkNet, Polygon zkEVM). The trade-off is larger proof sizes (tens of kilobytes vs. hundreds of bytes for Groth16), higher verification gas costs, and less mature tooling for ML circuits. The transition to post-quantum ZKP is an active area, with systems like **Orion** and **Brakedown** exploring hash-based polynomial commitments.

---

## 13. Regulatory, Economic, and Scaling Considerations

### 13.1 GDPR, Blockchain Immutability, and the Right to Erasure

The EU General Data Protection Regulation (GDPR, Art. 17) grants individuals a *right to erasure* ("right to be forgotten"): personal data must be deleted on request when it is no longer necessary, consent is withdrawn, or the subject objects to processing. Blockchain's defining property — immutability — creates a fundamental tension with this right. Once a transaction is committed to a public chain, neither the data nor its hash can be modified or removed without reorganizing the chain (requiring majority computational power).

For a federated learning blockchain that records client identifiers, gradient commitments, or model CIDs per-round, these on-chain records may constitute personal data if they are *linkable to an identifiable natural person* — the definition under GDPR Art. 4(1). A hospital client's blockchain address, if linkable to the institution (which is likely), may create a pseudonymous linkage between the institution and each round's gradient commitment.

**Technical approaches to GDPR-compatible blockchain FL logging**:

*Hash-only on-chain data with erasable off-chain content*: Commit only $H(\text{gradient CID})$ and $H(\text{client ID})$ to the blockchain — neither the raw data nor the actual CID. The preimage (content) is stored off-chain in a system supporting deletion. Deleting the off-chain content "erases" the data in the GDPR sense; the remaining on-chain hash is computationally unrelatable to deleted content if the hash preserves preimage resistance. This is the implicit approach in this framework's audit log.

*Chameleon hashes*: A chameleon hash (Krawczyk and Rabin, 1998) admits a *trapdoor collision* — the holder of a trapdoor key can find a second preimage for any commitment. This allows replacing committed content on-chain with an erasure token (e.g., $H_{\text{chameleon}}(\text{"GDPR erased"}, r')$) that produces the same on-chain value as the original commitment $H_{\text{chameleon}}(\text{data}, r)$. To external observers, the chain appears unchanged; the controller uses the trapdoor to substitute. The trapdoor must be held exclusively by the data controller — introducing a trust assumption that the trapdoor cannot be misused to forge non-erasure entries.

*Zero-knowledge erasure proofs*: The committer proves, via ZKP, that a specific piece of data has been removed from a committed dataset without revealing what the data was or any other dataset contents. The chain records the erasure proof rather than the erased content. This is a research-stage approach formalized in the Lethe framework (2022) but not yet deployed at production scale.

*Storage layering (recommended for production)*: Maintain strictly zero personal-data-adjacent information on-chain: only functional commitments (IPFS CIDs, DP budget state, training round number, aggregated model hashes), with all participant-identifying records in off-chain systems subject to standard deletion obligations. The blockchain's audit role is then restricted to verifying *that* training occurred correctly, not *who* participated. This design is GDPR-compliant provided the on-chain CIDs are not linkable to individuals.

**Regulatory guidance**: The European Data Protection Board (EDPB)'s 2021 Guidelines 05/2021 clarify that: (a) pseudonymization alone does not remove GDPR obligations (merely reduces risk); (b) hash functions are not sufficient for anonymization if preimage attacks or re-identification via auxiliary data are possible; (c) GDPR's "right to erasure" applies to data processed by automated means — which on-chain data qualifies as. Controllers deploying blockchain FL must conduct a DPIA (Data Protection Impact Assessment) specifically addressing the immutability tension and document the chosen mitigation.

---

### 13.2 Layer 2 Scaling and ZK-Rollups for FL Proof Aggregation

On-chain proof verification by individual client at FL scale faces a throughput mismatch: Ethereum L1 processes 15–100 transactions per second, while a 100-client FL system submitting proofs per training round generates 100 sequential transactions, taking minutes at mainnet throughput and costing $30,000–$40,000 in gas per 100 training rounds at typical 2025 prices.

**Layer 2 solutions** batch many off-chain transactions into a single L1 transaction, with different security models:

*Optimistic rollups* (Optimism, Arbitrum): Transactions are processed off-chain by a sequencer and assumed valid unless challenged. State roots are posted to L1 with a 7-day fraud challenge window. FL proof verifications run off-chain; the sequencer posts the resulting "all proofs passed / $k$ proofs failed" state root to L1. Cost: ~10–100× reduction versus L1. Latency for finality: 7 days (acceptable for FL training, which occurs over weeks). Security: inherits L1 security if at least one honest node monitors for fraud proofs.

*ZK-rollups* (zkSync Era, Polygon zkEVM, StarkNet): The L2 posts a *validity proof* (SNARK or STARK) to L1 alongside every batch state root, proving all L2 computations were correct. No challenge window required — cryptographic certainty is immediate. For FL:

1. All $n$ client Groth16 proofs are verified off-chain by the L2 prover.
2. The L2 prover generates a *recursive* SNARK proving: "I verified $n$ Groth16 proofs and all passed / $k$ failed." This recursive verification is the key technical operation.
3. The L2 posts one batch proof to L1 per FL training round: one pairing verification on L1 regardless of how many client proofs were batched.

**Gas cost reduction**: 100 client proofs @ ~700K gas each = 70M gas per round on L1. With ZK-rollup batching: one proof @ ~200K gas = 200K gas per round — a 350× cost reduction making blockchain-based FL economically feasible at scale.

**Recursive proof verification overhead**: PLONK-based recursion requires embedding the Groth16 verification circuit (3 pairings) inside a PLONK circuit. On the BN254 curve, a single Groth16 verifier requires approximately 2,000–5,000 PLONK constraints. Batching 100 Groth16 verifications requires 200,000–500,000 PLONK constraints — well within the proving capacity of modern PLONK provers (Halo2, gnark's PLONK backend). Batch proof generation time: approximately 5–30 seconds for 100 proofs.

This architecture — per-client Groth16 proving on local hardware (as in the current gnark service), followed by ZK-rollup aggregation for on-chain settlement — represents the mature production architecture for scalable blockchain FL.

---

### 13.3 Token Economics and Incentive Design for Decentralized FL

Sustaining participation in decentralized FL requires economic incentives that fairly compensate clients for legitimate training contributions while penalizing adversarial behavior. Token-based systems native to the blockchain coordination layer are the natural mechanism.

**Three-Token Architecture** (common across BCFL systems):

*Participation tokens*: Issued to clients who complete a training round with a valid ZKP proof. Proportional to declared local data size, adjusted by a credibility factor from historical proof verification rates. These tokens compensate baseline participation costs (computation, bandwidth).

*Performance tokens*: Issued based on the improvement attributable to the client's gradient update on a held-out server validation set. Computed via leave-one-out evaluation or gradient cosine similarity with the global update direction. Clients whose updates consistently improve aggregate accuracy receive more performance tokens.

*Stake tokens*: Must be deposited (locked) before joining a training round. Stake is slashed (forfeit) if: the client's ZKP proof fails verification, the client is detected submitting Byzantine gradients via post-hoc audit, or the client drops out after receiving the global model but before submitting an update. Stake provides Sybil resistance (creating $k$ Sybil identities costs $k \times \text{stake}$) and bonds honest behavior economically.

**Shapley value integration**: Computing exact Shapley values is exponential in client count. For a consortium of $n \leq 10$ hospitals (as in this framework's cross-silo setting), Monte Carlo Shapley approximation (Ghorbani and Zou, 2019) over $O(n \log n)$ coalition samplings is computationally feasible. The resulting Shapley-approximate values are committed on-chain by the verified aggregator and used to determine token distribution — providing a theoretically principled and publicly auditable basis for payment that satisfies efficiency, symmetry, linearity, and null-player axioms.

**Anti-free-rider enforcement**: Clients submitting near-zero gradient updates (minimal contribution) can pass norm-bound ZKP while contributing nothing. A *minimum contribution threshold* $B_{\min}$ can be added as a second ZKP circuit public input: the gradient must satisfy $B_{\min}^2 \leq \|\hat{g}_k\|_2^2 \leq B_{\max}^2$ (both lower and upper bounded). Clients whose gradients are below $B_{\min}$ fail proof generation, preventing participation without effort. The appropriate $B_{\min}$ is set proportionally to the dataset size each client declares — a client claiming 1,000 records must submit a gradient consistent with meaningful training on 1,000 records.

**On-chain DP budget governance**: A critical governance application of the blockchain coordination layer is transparent privacy budget tracking. The smart contract maintains the cumulative privacy budget $\varepsilon_{\text{total}}^t = \sum_{r=1}^t \Delta\varepsilon^r$, where $\Delta\varepsilon^r$ is the per-round budget consumption (computed via RDP accounting and posted by the verified aggregator). When $\varepsilon_{\text{total}}^t \geq \varepsilon_{\max}$ (a pre-agreed budget ceiling configured at contract deployment), the contract automatically refuses further rounds. This on-chain budget enforcement provides the first *verifiable DP compliance* mechanism: rather than trusting the coordinator's claim of $\varepsilon = 1.0$, any party can query the contract and verify the cumulative consumption independently.

---

## 14. Further Reading

### Foundational Blockchain

- **Nakamoto (2008)**: "Bitcoin: A Peer-to-Peer Electronic Cash System" — the original blockchain paper; defines the PoW consensus and UTXO model
- **Buterin (2013)**: "Ethereum: A Next-Generation Smart Contract and Decentralized Application Platform" — introduces programmable blockchain and the EVM
- **Lamport, Shostak, Pease (1982)**: "The Byzantine Generals Problem" — the theoretical foundation for consensus under arbitrary faults
- **Wood (2014)**: "Ethereum: A Secure Decentralised Generalised Transaction Ledger" — formal Ethereum yellow paper

### Blockchain-Enabled Federated Learning

- **Rangwala, Venugopal, Buyya (2025)**: "Blockchain-Enabled Federated Learning" — arXiv:2508.06406 — comprehensive BCFL taxonomy with four-dimensional classification and TrustMesh case study
- **Xing et al. (2023)**: "Blockchain-Federated Learning with ZKP" — one of the first systems to combine all three technologies, included in this project's reference library
- **Ruckel, Schlör, Heindl (2022)**: "Fairness, Integrity, and Privacy in a Scalable Blockchain-based FL system" — formal treatment of ZKP-fairness in FL rewards

### ZKP on Blockchain

- **Groth (2016)**: "On the Size of Pairing-Based Non-interactive Arguments" — the Groth16 proof system used in most on-chain ZKP verifiers
- **Ben-Sasson et al. (2013)**: "SNARKs for C" — foundational paper on zkSNARKs for general computation; precursor to the programming model used in Circom and gnark
- **Kothapalli, Setty, Tzialla (2022)**: "NOVA: Recursive Zero-Knowledge Arguments from Folding Schemes" — the IVC construction used in VerifBFL

### Combined ZKP + FHE + FL + Blockchain Systems

- **Bellachia et al. (2025)**: "VerifBFL: Leveraging zk-SNARKs for A Verifiable Blockchained Federated Learning" — arXiv:2501.04319 — NOMS'25 — IVC-based full-stack system with benchmarks
- **Zhang et al. (2025)**: "Zkfhed: A Verifiable and Scalable Blockchain-Enhanced Federated Learning System" — IEEE TKDE DOI:10.1109/TKDE.2025.3550546 — ZKP + FHE + smart contracts, closest to this project's architecture
- **Bruschi, Esposito, Gagliardoni, Rizzini (2025)**: "SoK: Verifiable Federated Learning" — IACR ePrint 2025/2296 — systematization of the entire field; required reading
- **Wang et al. (2025)**: "Zero-Knowledge Federated Learning" — arXiv:2503.15550 — ZK-FL taxonomy and Veri-CS-FL algorithm

### FHE for Blockchain

- **Cheon, Kim, Kim, Song (2017)**: "Homomorphic Encryption for Arithmetic of Approximate Numbers" — the CKKS scheme implemented in TenSEAL; basis for FHE-FL gradient aggregation
- **Chillotti et al. (2020)**: "TFHE: Fast Bootstrapping over the Torus" — the TFHE scheme underlying Concrete ML; enables fast bootstrapping for non-linear circuit evaluation
- **Yang (2025)**: "Improving Efficiency in Federated Learning with Optimized Homomorphic Encryption" — arXiv:2504.03002 — selective encryption and sensitivity maps for 3× efficiency

### Distributed Storage

- **Benet (2014)**: "IPFS — Content Addressed, Versioned, P2P File System" — the whitepaper for IPFS; explains content addressing and the DHT routing layer
- **Protocol Labs (2017)**: "Filecoin: A Decentralized Storage Network" — adds economic incentives and cryptographic proofs of storage to IPFS

### Evaluation and Verification of FL

- **Commey, Appiah, Klogo, Crosby (2025)**: "ZKP-FedEval: Verifiable and Privacy-Preserving Federated Evaluation using Zero-Knowledge Proofs" — arXiv:2507.11649 — evaluation-phase privacy via ZKP
- **Huang et al. (2025)**: "Advancing Practical Homomorphic Encryption for Federated Learning" — arXiv:2509.20476 — spectral analysis and encryption ratio study for selective HE
- **Alaa et al. (2025)**: "Verifiable Split Learning via zk-SNARKs" — arXiv:2511.01356 — extends verifiable computation to split learning with key finding that ZKP is strictly stronger than blockchain-only approaches

---

## 15. Implementation in This Framework

This section describes what is actually built in `fl_ppml/fl/` — moving from theory to running code.

### Module Overview

| File | Role |
|------|------|
| `fl/chain.py` | Chain backend factory + `MockChain` + `Web3Chain` |
| `fl/chain_contract/FLLedger.sol` | On-chain audit ledger smart contract |
| `fl/config.py` | `FLConfig` — four new chain fields |
| `fl/server.py` | Writes chain events after every aggregation round |
| `fl/privacy/zkp.py` | Populates `_last_anchor_data` for ZKP-only modes |
| `fl/privacy/he_zkp.py` | Populates `_last_anchor_data` for HE+ZKP composite modes |
| `compare.py` | `--chain-backend` / `--chain-ledger-dir` CLI flags |
| `fl/compare/runner.py` | Per-mode ledger paths; merges ledgers; prints audit table |

---

### `fl/chain.py` — Backend Factory

```python
from fl.chain import get_chain

chain = get_chain(config)   # reads config.chain_backend
```

Three backends:

| `chain_backend` | Class | Behaviour |
|----------------|-------|-----------|
| `"mock"` (default) | `MockChain` | Appends events to an in-process list; writes `{"ledger": [...]}` JSON to disk |
| `"web3"` | `Web3Chain` | Submits transactions to an EVM node at `config.chain_rpc_url` |
| `"none"` | `NullChain` | No-op; disables the audit ledger |

`MockChain` saves after every event, so ledger files survive crashes mid-run.

---

### `fl/chain_contract/FLLedger.sol` — Smart Contract

The Solidity contract exposes two entry points:

```solidity
// Record a global model update for a round
function commitModel(
    uint256 round,
    bytes32 modelHash,
    bytes32[] calldata clientHashes
) external;                        // emits ModelCommitted

// Anchor the ZKP proofs submitted that round
function anchorProofs(
    uint256 round,
    bytes32[] calldata proofHashes,
    uint256[] calldata clientIds
) external;                        // emits ProofsAnchored
```

Emitted events are indexed by `round` for efficient on-chain querying and provide the audit evidence for regulatory compliance.

---

### `fl/config.py` — New Chain Fields

```python
@dataclass
class FLConfig:
    # ... existing fields ...
    chain_backend: str = "mock"                    # "mock" | "web3" | "none"
    chain_rpc_url: str = "http://127.0.0.1:8545"   # Ganache / Hardhat / real node
    chain_contract_addr: str = ""                  # deployed FLLedger.sol address
    chain_ledger_path: str = "./results/ledger.json"
```

---

### Per-Round Flow

Every aggregation round the server executes:

```
aggregate_fit()
 └─ _chain_commit(server_round, aggregated_weights, client_results)
     ├─ chain.commit_model(round, model_hash, client_hashes)   # ALL modes
     └─ chain.anchor_proofs(round, proof_hashes, client_ids)   # ZKP modes only
         (only when privacy_mode._last_anchor_data is populated)
```

ZKP-containing modes (`zkp`, `zkp_sampled`, `he_tenseal_zkp`, `he_concrete_tfhe_zkp`) set `_last_anchor_data` inside their `aggregate_fit_override()`. Non-ZKP modes never set it, so `anchor_proofs` is never called — this is the source of the 3 vs 6 event counts seen in the audit table.

---

### CLI Usage

```bash
## Default: mock backend, ledgers written to results/<dataset>/<timestamp>/ledgers/
python compare.py --dataset healthcare

## Disable ledger entirely
python compare.py --dataset healthcare --chain-backend none

## Custom ledger directory
python compare.py --dataset healthcare --chain-ledger-dir /tmp/my_ledgers

## Simulation mode also supports chain flags
python simulation.py simulation --rounds 3 --benchmark \
  --chain_backend mock --chain_ledger_path /tmp/sim_ledger.json

## Server subprocess (used internally by compare runner)
python main_server.py --chain_backend mock --chain_ledger_path ./results/ledger.json
```

---

### Output Files

Each run produces per-mode ledger files plus a combined audit file:

```
results/<dataset>/<timestamp>/
├── ledger_comparison.json          ← combined audit trail (all modes merged)
└── ledgers/
    ├── ledger_baseline.json        ← {"ledger": [{type, round, ...}, ...]}
    ├── ledger_dp.json
    ├── ledger_he_tenseal.json
    ├── ledger_he_concrete_tfhe.json
    ├── ledger_zkp.json
    ├── ledger_zkp_sampled.json
    ├── ledger_he_tenseal_zkp.json
    └── ledger_he_concrete_tfhe_zkp.json
```

Each entry in a ledger file has this shape:

```json
{"ledger": [
  {
    "type": "ModelCommit",
    "round": 1,
    "model_hash": "a3f9...",
    "client_hashes": ["b1c2...", "d4e5..."],
    "block": 1
  },
  {
    "type": "ProofAnchor",
    "round": 1,
    "proof_hashes": ["f6a7...", "09b8..."],
    "client_ids": [0, 1],
    "block": 2
  }
]}
```

---

### Blockchain Audit Summary (console output)

After every compare run the runner prints:

```
=== Blockchain Audit Summary ===
Mode                     Events  ModelCommit  ProofAnchor  Last Block
baseline                      3            3            0           3
dp                            3            3            0           3
he_tenseal                    3            3            0           3
he_concrete_tfhe              3            3            0           3
zkp                           6            3            3           6
zkp_sampled                   6            3            3           6
he_tenseal_zkp                6            3            3           6
he_concrete_tfhe_zkp          6            3            3           6
TOTAL                        36           24           12
```

Verified production result (3 rounds, 2 clients, 8 modes): **36 total events — all checks passed ✓**

---

### Switching to a Real EVM Node

To use an actual blockchain instead of the mock JSON ledger:

1. Deploy `FLLedger.sol` to a network (Ganache, Hardhat, Sepolia, etc.)
2. Note the deployed contract address
3. Update `fl/config.py` (or pass via CLI):
   ```python
   chain_backend = "web3"
   chain_rpc_url = "http://127.0.0.1:8545"   # or Infura / Alchemy endpoint
   chain_contract_addr = "0xYourDeployedAddress"
   ```
4. Run normally — `Web3Chain` will replace `MockChain` automatically

The `Web3Chain` class uses `web3.py` and the same `commit_model` / `anchor_proofs` interface, so no other code changes are required.

---

*This document is part of the federated learning framework documentation. For the ZKP implementation details (gnark, Groth16, Pedersen commitments), see [ZKP.md](ZKP.md). For the FHE implementation details (TenSEAL/CKKS and Concrete ML/TFHE), see [FHE.md](FHE.md). For differential privacy and the Opacus integration, see [DP.md](DP.md). For the overall framework architecture and mode comparison, see [README.md](README.md).*
