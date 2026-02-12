Written with the help of Gemini.

# SocratesNSLM (Neuro-Symbolic Language Model)

## 1. System Overview

The Socrates NSLM is a hybrid architecture composed of two coupled systems:
1.  **The Semantic Topology ($\mathcal{G}$):** A static, weighted graph derived from corpus statistics that defines the semantic manifold.
2.  **The Generative Model ($\mathcal{M}$):** A Transformer-based neural network whose output probability space and training curriculum are constrained by $\mathcal{G}$, but capable of open-vocabulary generation via a fallback mechanism.

Let the corpus be a sequence of discrete symbols $C = (w_1, w_2, ..., w_T)$, where each $w_t \in \mathcal{V}$ (the vocabulary of whole words).

---

## 2. The Semantic Topology ($\mathcal{G}$)

We define an undirected weighted graph $\mathcal{G} = (V, E, W)$.

### 2.1. Vertex & Edge Definition
* **Vertices ($V$):** The set of unique concepts (whole words) in the corpus, $V = \{v_1, ..., v_N\}$.
* **Edges ($E$):** An edge $(v_i, v_j)$ exists if and only if the words appear as **Direct Neighbors** (bigrams) in $C$ and satisfy the significance threshold defined in 2.3.

### 2.2. Edge Weighting (Normalized PMI)
Let $P(v_i)$ be the probability of word $v_i$ occurring in $C$, and $P(v_i, v_j)$ be the joint probability of the bigram $(v_i, v_j)$.
The edge weight $W_{ij}$ is defined as the **Normalized Pointwise Mutual Information (NPMI)**:

$$
W_{ij} = \text{NPMI}(v_i, v_j) = \frac{\log_2 \frac{P(v_i, v_j)}{P(v_i)P(v_j)}}{-\log_2 P(v_i, v_j)}
$$

where $W_{ij} \in [-1, 1]$.

### 2.3. Topology Filtering (Entropy & Significance)
We define a filter function $\Phi: E \rightarrow \{0, 1\}$ to sparsify the graph.

1.  **Hub Removal (Entropy Filter):**
    Let $H(v_i)$ be the Shannon entropy of the neighbor distribution of $v_i$. If $H(v_i) > \mu_H + k\sigma_H$, node $v_i$ is classified as a "Stopword Hub" and is removed from the set of valid cluster centers (though it remains in the graph as a connector).

2.  **Significance Constraint (Z-Score Pruning):**
    An edge $(v_i, v_j)$ is retained only if:
    $$W_{ij} > \mu_W + \alpha \sigma_W$$
    where $\mu_W$ and $\sigma_W$ are the mean and standard deviation of all positive NPMI scores, and $\alpha$ is a tunable Z-score parameter (typically $\alpha \approx 1.0$).

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

---

## 3. The Generative Model ($\mathcal{M}$)

The model is a Transformer parameterized by $\theta$.

### 3.1. Input Representation (Asymmetric Tokenization)
The input sequence $x = (t_1, ..., t_L)$ consists of **Type-Frequency Optimized Tokens (Cicero)** to handle open-vocabulary input.

$$
h_0 = \text{Embedding}(x) + \text{PositionalEncoding}(x)
$$
$$
z = \text{TransformerBlock}(h_{L-1})
$$

### 3.2. Hierarchical Output Structure
The output probability is computed via a **Multi-Path Hierarchical Softmax** over the graph clusters, extended with a **Passthrough Mechanism** for OOV entities.

We augment the set of $K$ semantic clusters with a special **Null Cluster** $C_\emptyset$. The probability of generating a token is a mixture of the Graph Mode and Passthrough Mode:

1.  **Stage 1: Mode Steering:**
    A projection layer $f_1$ predicts the target cluster over $K+1$ possibilities:
    $$P(C_k | z) = \text{Softmax}(W_1 z + b_1)_k$$

2.  **Stage 2: Conditional Generation:**
    * **Case A (Graph Mode):** If $k \neq \emptyset$, predict word $w \in C_k$:
        $$P(w | C_k, z) = \text{Softmax}(W_{2,k} z + b_{2,k})_w$$
    * **Case B (Passthrough Mode):** If $k = \emptyset$, initiate the **Character Passthrough** (see 3.3).

### 3.3. Character Passthrough ($\pi_{\text{char}}$)
To handle novel identifiers (variable names, hashes, strict syntax) not present in $\mathcal{G}$, the model enters an autoregressive character generation loop.

Given the frozen hidden state $z$, a lightweight recurrent head (or small Transformer block) generates a sequence of characters $c_{1:T}$ until a specialized End-of-Word token ($\text{EOW}$) is produced:

$$
P(c_t | c_{\lt}, z) = \text{Softmax}(W_{\text{char}} h_t + b_{\text{char}})
$$

where $h_t = \text{RNN}(c_{t-1}, h_{t-1}; z)$. The total probability for a novel word $w_{new}$ composed of characters $c_{1:T}$ is:

$$
P(w_{new} | z) = P(C_{\emptyset} | z) \cdot \prod_{t=1}^{T} P(c_t | c_{\lt}, z)
$$

---

## 4. The Training Curriculum

We define a **Phase-based Loss Function** $\mathcal{L}$ to stabilize training.

### Phase 1: Core Concept Acquisition
Training is restricted to **Core Nodes** (where $|\mathcal{K}(w)| = 1$). The loss minimizes the negative log-likelihood of the unique valid path:

$$
\mathcal{L}_{\text{core}} = -\log P(C_{k^\ast} | z) - \log P(w | C_{k^\ast}, z)
$$

where $k^\ast$ is the unique cluster of word $w$.

### Phase 2: Bridge Node & Passthrough Integration
Training includes **Bridge Nodes** and **OOV Literals**.
* **Bridge Nodes:** Optimized via Multi-Label cluster prediction (as defined previously).
* **Passthrough Trigger:** For any target word $w \notin V$ (not in Graph), the model is forced to predict $C_\emptyset$ and minimize the character sequence loss.

The combined loss for Phase 2 is:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{bridge}} + \lambda_{\text{char}} \mathbb{I}(w \notin V) \left[ -\log P(C_\emptyset | z) - \sum_{t} \log P(c_t | c_{<t}, z) \right]
$$

*(Note: $\lambda_{\text{char}}$ regulates the cost of using the passthrough to prevent the model from bypassing the semantic graph for known words.)*
