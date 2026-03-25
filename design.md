Written with the help of Gemini and Claude.

# SocratesNSLM (Neuro-Symbolic Language Model)

## 1. System Overview

The Socrates NSLM is a hybrid architecture composed of two coupled systems:
1.  **The Semantic Topology ($\mathcal{G}$):** A weighted graph derived from corpus statistics that defines the semantic manifold, with an incremental update mechanism for integrating newly encountered vocabulary.
2.  **The Generative Model ($\mathcal{M}$):** A Transformer-based neural network whose output probability space and training curriculum are constrained by $\mathcal{G}$, but capable of open-vocabulary generation via a fallback mechanism.

Let the corpus be a sequence of discrete symbols $C = (w_1, w_2, ..., w_T)$, where each $w_t \in \mathcal{V}$ (the vocabulary of whole words).

---

## 2. The Semantic Topology ($\mathcal{G}$)

We define an undirected weighted graph $\mathcal{G} = (V, E, W)$.

### 2.1. Vertex & Edge Definition
* **Vertices ($V$):** The set of unique concepts (whole words) in the corpus, $V = \{v_1, ..., v_N\}$.
* **Co-occurrence Window ($\omega$):** A configurable integer parameter $\omega \geq 1$ controlling the maximum token distance at which two words are considered co-occurring. When $\omega = 1$, only direct bigrams are counted (recovering the original design). Higher values capture broader semantic relationships at the cost of increased graph density and noise.
* **Edges ($E$):** An edge $(v_i, v_j)$ exists if and only if $v_i$ and $v_j$ appear within $\omega$ tokens of each other in $C$ (i.e., $\exists\, t$ such that $w_t = v_i$ and $w_{t+d} = v_j$ for some $1 \leq |d| \leq \omega$), and the edge satisfies the significance threshold defined in 2.3.

The choice of $\omega$ governs the structural properties of the resulting graph:
- **$\omega = 1$:** Yields phrasal and collocational structure (bigrams), producing clusters of words with strict adjacency relationships.
- **$\omega \in [2, 5]$:** Captures local syntactic and semantic neighborhoods, including modifier–head, verb–argument, and short-range topical associations.
- **$\omega \in [5, 10]$:** Reflects broader topical co-occurrence at the discourse level, at the cost of increased edge noise.

The parameter $\omega$ should be tuned on a held-out set by measuring downstream cluster coherence (e.g., via word intrusion detection).

### 2.2. Edge Weighting (Normalized PMI)
Let $P(v_i)$ be the probability of word $v_i$ occurring in $C$, and $P_\omega(v_i, v_j)$ be the joint probability of words $v_i$ and $v_j$ appearing within $\omega$ tokens of each other.
The edge weight $W_{ij}$ is defined as the **Normalized Pointwise Mutual Information (NPMI)**:

$$
W_{ij} = \text{NPMI}_\omega(v_i, v_j) = \frac{\log_2 \frac{P_\omega(v_i, v_j)}{P(v_i)P(v_j)}}{-\log_2 P_\omega(v_i, v_j)}
$$

where $W_{ij} \in [-1, 1]$.

Note that as $\omega$ increases, $P_\omega(v_i, v_j)$ rises for most pairs, which shifts the NPMI distribution. The significance threshold in 2.3 is computed relative to this distribution, so it self-adjusts, but the z-score parameter $\alpha$ may need to be increased for larger $\omega$ to maintain comparable sparsity.

### 2.3. Topology Filtering (Entropy & Significance)
We define a filter function $\Phi: E \rightarrow \{0, 1\}$ to sparsify the graph.

1.  **Hub Removal (Entropy Filter):**
    Let $H(v_i)$ be the Shannon entropy of the neighbor distribution of $v_i$. If $H(v_i) > \mu_H + k\sigma_H$, node $v_i$ is classified as a "Stopword Hub" and is removed from the set of valid cluster centers (though it remains in the graph as a connector).

2.  **Significance Constraint (Z-Score Pruning):**
    An edge $(v_i, v_j)$ is retained only if:
    $$W_{ij} > \mu_W + \alpha \sigma_W$$
    where $\mu_W$ and $\sigma_W$ are the mean and standard deviation of all positive NPMI scores, and $\alpha$ is a tunable Z-score parameter. Note that $\alpha$ may need to be increased for larger $\omega$ to maintain comparable graph sparsity.

### 2.4. Overlapping Community Detection (Soft Clustering)
We define a **Membership Matrix** $M \in \{0, 1\}^{N \times K}$ where $M_{ik} = 1$ if word $v_i$ belongs to Cluster $k$.

1.  **Initial Partitioning (Modularity Maximization):**
    We maximize the Modularity $Q$ to find the initial disjoint communities:
    $$Q = \frac{1}{2m} \sum_{i,j} \left( W_{ij} - \frac{k_i k_j}{2m} \right) \delta(c_i, c_j)$$
    where $m$ is the sum of all weights, $k_i$ is the weighted degree of node $i$, and $\delta$ is the Kronecker delta.

2.  **Fuzzy Membership Extension:**
    To account for polysemy (Bridge Nodes), we extend membership. A word $v_i$ is assigned to cluster $C_k$ if its connectivity to that cluster exceeds a ratio $\beta$:
    $$M_{ik} = \mathbb{I}\left( \frac{\sum_{j \in C_k} W_{ij}}{\sum_{j \in V} W_{ij}} > \beta \right)$$
    * **Core Nodes:** $\sum_k M_{ik} = 1$.
    * **Bridge Nodes:** $\sum_k M_{ik} > 1$.

### 2.5. Incremental Graph Evolution

The topology $\mathcal{G}$ is not permanently frozen after initial construction. An **Incremental Update Protocol** allows the graph to incorporate newly encountered vocabulary without full reconstruction, avoiding the instability of wholesale cluster reorganization.

#### 2.5.1. Candidate Accumulation

During continued training or inference, the model maintains a **staging buffer** $\mathcal{B}$ of words that triggered the Character Passthrough ($C_\emptyset$). Each candidate $w_c$ accumulates:
- A raw frequency count $n(w_c)$ across encounters.
- A running co-occurrence vector $\mathbf{q}(w_c) \in \mathbb{R}^{|V|}$, where $q_j(w_c)$ counts how many times $w_c$ appeared within $\omega$ tokens of existing vertex $v_j$.

A candidate is promoted when $n(w_c) \geq \tau_{\text{freq}}$, where $\tau_{\text{freq}}$ is a frequency threshold (e.g., the minimum frequency of any vertex in the current graph, or a fixed value).

#### 2.5.2. Node Insertion

When a candidate $w_c$ is promoted:

1. **Vertex addition:** $V \leftarrow V \cup \{w_c\}$.
2. **Edge construction:** For each existing vertex $v_j$, compute $\text{NPMI}_\omega(w_c, v_j)$ from the accumulated statistics. Retain edges passing the significance threshold from §2.3.
3. **Cluster assignment:** Assign $w_c$ to the cluster $C_k$ that maximizes its normalized connectivity:
$$k^\ast = \arg\max_k \frac{\sum_{j \in C_k} W_{c,j}}{\sum_{j \in V} W_{c,j}}$$
If secondary cluster connectivity exceeds $\beta$ (from §2.4), assign additional memberships (the new word may immediately become a Bridge Node).
4. **Output head extension:** The weight vectors for the assigned cluster head(s) are extended by one dimension, initialized to the mean of existing word vectors in that cluster scaled by a small factor $\epsilon$ to avoid disrupting existing predictions:
$$W_{2,k}[w_c] \leftarrow \epsilon \cdot \frac{1}{|C_k|} \sum_{w \in C_k} W_{2,k}[w]$$

#### 2.5.3. Stability Guarantees

The protocol is insertion-only for graph topology — existing edges and cluster assignments are never modified by the incremental update. This ensures:
- **No cluster reorganization:** Existing words keep their cluster memberships. The model's learned routing behavior for known vocabulary is unaffected.
- **Monotonic vocabulary growth:** $|V|$ is non-decreasing. The output space can only expand.
- **Bounded disruption:** The only parameters that change are the newly added rows/columns in the affected cluster heads. All other weights remain untouched until the next gradient update from normal training.

Full graph reconstruction (re-running §2.1–2.4 from scratch with updated corpus statistics) can be performed periodically at major checkpoints as an optional consolidation step, followed by a short fine-tuning phase to re-align the model with any cluster boundary changes.

---

## 3. The Generative Model ($\mathcal{M}$)

The model is a Transformer parameterized by $\theta$.

### 3.1. Input Representation (Asymmetric Tokenization)
The input sequence uses **Type-Frequency Optimized Tokens (Cicero)**, a subword tokenization, to handle open-vocabulary input. The output predicts whole words from the graph vocabulary $V$. This asymmetry requires an explicit alignment step.

#### 3.1.1. Subword-to-Word Boundary Pooling

Let the subword input sequence be $x = (t_1, ..., t_L)$, producing Transformer hidden states $z_1, ..., z_L$. Word boundaries are deterministic given the tokenizer: each whole word $w_i$ corresponds to a contiguous span of subword tokens $t_{a_i}, ..., t_{b_i}$.

We aggregate subword states to word-level representations using **attention-weighted boundary pooling**:

$$
\tilde{z}_i = \sum_{j=a_i}^{b_i} \alpha_j \cdot z_j, \quad \alpha_j = \frac{\exp(u^\top z_j)}{\sum_{l=a_i}^{b_i} \exp(u^\top z_l)}
$$

where $u \in \mathbb{R}^d$ is a learned context vector shared across all positions. This aggregation introduces minimal additional parameters while allowing the model to learn which subword positions carry the most predictive information for the subsequent whole-word decision.

The pooled states $\tilde{z}_i$ are the inputs to the hierarchical output structure (§3.2) and the character passthrough (§3.3). All output-side operations are defined over $\tilde{z}$, not the raw subword states $z$.

### 3.2. Hierarchical Output Structure
The output probability is computed via a **Multi-Path Hierarchical Softmax** over the graph clusters, extended with a **Passthrough Mechanism** for OOV entities.

We augment the set of $K$ semantic clusters with a special **Null Cluster** $C_\emptyset$. The probability of generating a token is a mixture of the Graph Mode and Passthrough Mode.

#### 3.2.1. Cluster Head Parameterization

Rather than maintaining $K$ fully independent projection matrices, we use a **shared-trunk with low-rank cluster adapters** to control parameter growth:

$$
W_{2,k} = W_{\text{shared}} + A_k B_k^\top
$$

where $W_{\text{shared}} \in \mathbb{R}^{|V_{\max}| \times d}$ is a single shared output embedding matrix, and $A_k \in \mathbb{R}^{|C_k| \times r}$, $B_k \in \mathbb{R}^{d \times r}$ are low-rank adapter matrices with rank $r \ll d$. This provides each cluster with a specialized output subspace while sharing the majority of the parameters.

Total output parameters scale as $O(|V| \cdot d + K \cdot r \cdot (d + \bar{c}))$ where $\bar{c}$ is the average cluster size, compared to $O(K \cdot \bar{c} \cdot d)$ for independent heads.

#### 3.2.2. Generation Stages

1.  **Stage 1: Mode Steering.**
    A projection layer $f_1$ predicts the target cluster over $K+1$ possibilities:
    $$P(C_k | \tilde{z}) = \text{Softmax}(W_1 \tilde{z} + b_1)_k$$

2.  **Stage 2: Conditional Generation.**
    * **Case A (Graph Mode):** If $k \neq \emptyset$, predict word $w \in C_k$:
        $$P(w | C_k, \tilde{z}) = \text{Softmax}(W_{2,k}\, \tilde{z} + b_{2,k})_w$$
    * **Case B (Passthrough Mode):** If $k = \emptyset$, initiate the **Character Passthrough** (see §3.3).

### 3.3. Character Passthrough ($\pi_{\text{char}}$)
To handle novel identifiers (variable names, hashes, strict syntax) not present in $\mathcal{G}$, the model enters an autoregressive character generation loop.

Given the pooled hidden state $\tilde{z}$, a lightweight Transformer head generates a sequence of characters $c_{1:T}$ until a specialized End-of-Word token ($\text{EOW}$) is produced:

$$
P(c_t | c_{\lt t}, \tilde{z}) = \text{Softmax}(W_{\text{char}}\, h_t + b_{\text{char}})
$$

where $h_t = \text{CharTransformer}(c_{1:t-1};\, \tilde{z})$, with $\tilde{z}$ injected via cross-attention at each layer rather than as a frozen initial state. This allows the character decoder to re-read the word-level context at every step, avoiding the information bottleneck of single-shot conditioning.

The total probability for a novel word $w_{\text{new}}$ composed of characters $c_{1:T}$ is:

$$
P(w_{\text{new}} | \tilde{z}) = P(C_{\emptyset} | \tilde{z}) \cdot \prod_{t=1}^{T} P(c_t | c_{\lt t}, \tilde{z})
$$

Words generated via Passthrough that later meet the promotion threshold (§2.5.1) are candidates for integration into $\mathcal{G}$.

---

## 4. The Training Curriculum

We define a **Phase-based Loss Function** $\mathcal{L}$ to stabilize training.

### Phase 1: Core Concept Acquisition
Training is restricted to **Core Nodes** (where $|\mathcal{K}(w)| = 1$). The loss minimizes the negative log-likelihood of the unique valid path:

$$
\mathcal{L}_{\text{core}} = -\log P(C_{k^\ast} | \tilde{z}) - \log P(w | C_{k^\ast}, \tilde{z})
$$

where $k^\ast$ is the unique cluster of word $w$.

### Phase 2: Bridge Node & Passthrough Integration
Training includes **Bridge Nodes** and **OOV Literals**.

**Bridge Nodes** are optimized via a **Marginal Likelihood** over all valid cluster assignments. For a target word $w$ with membership set $\mathcal{K}(w) = \{k : M_{wk} = 1\}$:

$$
\mathcal{L}_{\text{bridge}} = -\log \sum_{k \in \mathcal{K}(w)} P(C_k | \tilde{z}) \cdot P(w | C_k, \tilde{z})
$$

This allows the model to learn contextual word sense disambiguation: the gradient naturally upweights the cluster that best fits the current context without requiring an explicit hard assignment. The marginalization is exact and differentiable since $|\mathcal{K}(w)|$ is small for polysemous words.

**Passthrough Trigger:** For any target word $w \notin V$ (not in Graph), the model is forced to predict $C_\emptyset$ and minimize the character sequence loss.

The combined loss for Phase 2 is:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{bridge}} + \lambda_{\text{char}} \mathbb{I}(w \notin V) \left[ -\log P(C_\emptyset | \tilde{z}) - \sum_{t} \log P(c_t | c_{\lt t}, \tilde{z}) \right]
$$

where $\lambda_{\text{char}}$ regulates the cost of using the passthrough to prevent the model from bypassing the semantic graph for known words.

---

## 5. Inference: Hierarchical Beam Decoding

At inference time, the two-stage output structure and the character passthrough require a modified beam search that operates across both modes.

### 5.1. Beam Expansion

At each generation step, given beam budget $B$:

1. **Cluster pre-selection.** Compute $P(C_k | \tilde{z})$ for all $K+1$ clusters. Retain the top-$B_c$ clusters (where $B_c \leq B$ is a cluster beam width).

2. **Intra-cluster expansion.** For each retained Graph Mode cluster $C_k$, compute $P(w | C_k, \tilde{z})$ and take the top-$B_w$ words. Each produces a candidate with joint score:
$$s(w) = \log P(C_k | \tilde{z}) + \log P(w | C_k, \tilde{z})$$

3. **Passthrough expansion.** If $C_\emptyset$ is among the top-$B_c$ clusters, allocate $B_p$ beam slots to the character decoder. Run the character decoder autoregressively, maintaining $B_p$ character-level beams until $\text{EOW}$ is produced. Each completed passthrough word $w_{\text{new}}$ receives score:
$$s(w_{\text{new}}) = \log P(C_\emptyset | \tilde{z}) + \sum_{t=1}^{T} \log P(c_t | c_{\lt t}, \tilde{z})$$

4. **Pool and prune.** Merge all candidates from steps 2 and 3 into a single pool. Apply length-normalized scoring (§5.2) and retain the top-$B$ hypotheses for the next step.

### 5.2. Length-Normalized Scoring

Raw log-probabilities are not directly comparable between Graph Mode (single-step) and Passthrough Mode (multi-step character product). We normalize by effective decision count:

$$
\hat{s}(w) = \frac{s(w)}{D(w)^\gamma}
$$

where $D(w) = 1$ for Graph Mode words (one cluster decision + one word decision are already folded into $s$) and $D(w) = T + 1$ for Passthrough words ($T$ character decisions + the null cluster decision), and $\gamma \in [0.5, 1.0]$ is a length penalty exponent tuned on a validation set. Setting $\gamma = 0$ recovers unnormalized scoring; $\gamma = 1$ gives full per-decision normalization.

The effective candidate pool per step is at most $B_c \cdot B_w + B_p$, pruned to $B$ before the next step. The values of $B$, $B_c$, $B_w$, $B_p$, and $\gamma$ should be determined empirically.
