Written with the help of Gemini.

# SocratesNSLM (Neuro-Symbolic Language Model)

## 1. System Overview

The Socrates NSLM is a hybrid architecture composed of two coupled systems:
1.  **The Semantic Topology ($\mathcal{G}$):** A static, weighted graph derived from corpus statistics that defines the semantic manifold.
2.  **The Generative Model ($\mathcal{M}$):** A Transformer-based neural network whose output probability space and training curriculum are constrained by $\mathcal{G}$.

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
    Let $H(v_i)$ be the Shannon entropy of the neighbor distribution of $v_i$. If $H(v_i) > \mu_H + k\sigma_H$, node $v_i$ is classified as a "Stopword Hub" and is removed from the set of valid cluster centers.

2.  **Significance Constraint (Z-Score Pruning):**
    An edge $(v_i, v_j)$ is retained only if:
    $$
    W_{ij} > \mu_W + \alpha \sigma_W
    $$
    where $\mu_W$ and $\sigma_W$ are the mean and standard deviation of all positive NPMI scores, and $\alpha$ is a tunable Z-score parameter (typically $\alpha \approx 1.0$).

### 2.4. Overlapping Community Detection (Soft Clustering)
We define a **Membership Matrix** $M \in \{0, 1\}^{N \times K}$ where $M_{ik} = 1$ if word $v_i$ belongs to Cluster $k$.

1.  **Initial Partitioning (Modularity Maximization):**
    We maximize the Modularity $Q$ to find the initial disjoint communities:
    $$
    Q = \frac{1}{2m} \sum_{i,j} \left( W_{ij} - \frac{k_i k_j}{2m} \right) \delta(c_i, c_j)
    $$
    where $m$ is the sum of all weights, $k_i$ is the weighted degree of node $i$, and $\delta$ is the Kronecker delta.

2.  **Fuzzy Membership Extension:**
    To account for polysemy (Bridge Nodes), we extend membership. A word $v_i$ is assigned to cluster $C_k$ if its connectivity to that cluster exceeds a ratio $\beta$:
    $$
    M_{ik} = \mathbb{I}\left( \frac{\sum_{j \in C_k} W_{ij}}{\sum_{j \in V} W_{ij}} > \beta \right)
    $$
    * **Core Nodes:** $\sum_k M_{ik} = 1$.
    * **Bridge Nodes:** $\sum_k M_{ik} > 1$.

---

## 3. The Generative Model ($\mathcal{M}$)

The model is a Transformer parameterized by $\theta$.

### 3.1. Input Representation (Asymmetric Tokenization)
The input sequence $x = (t_1, ..., t_L)$ consists of **Sub-word Tokens (BPE)** to handle open-vocabulary input.

$$
h_0 = \text{Embedding}(x) + \text{PositionalEncoding}(x)
$$
$$
z = \text{TransformerBlock}(h_{L-1})
$$

### 3.2. Hierarchical Output Structure
The output probability $P(w | z)$ is computed via a **Multi-Path Hierarchical Softmax** over the whole-word graph clusters.

Let $\mathcal{K}(w) = \{k \mid M_{wk} = 1\}$ be the set of clusters containing word $w$. The probability of generating word $w$ is the marginal probability summed over all valid semantic paths:

$$
P(w | z) = \sum_{k \in \mathcal{K}(w)} P(w | C_k, z) \cdot P(C_k | z)
$$

1.  **Stage 1: Cluster Prediction (Steering):**
    A projection layer $f_1$ maps the final hidden state $z$ to the probability distribution over the $K$ clusters:
    $$
    P(C_k | z) = \text{Softmax}(W_1 z + b_1)_k
    $$

2.  **Stage 2: Concept Prediction (Specifics):**
    Given an active cluster $k$, a cluster-specific projection $f_{2,k}$ maps $z$ to the words strictly within $C_k$:
    $$
    P(w | C_k, z) = \text{Softmax}(W_{2,k} z + b_{2,k})_w
    $$

---

## 4. The Training Curriculum

We define a **Phase-based Loss Function** $\mathcal{L}$ to stabilize training.

### Phase 1: Core Concept Acquisition
Training is restricted to **Core Nodes** (where $|\mathcal{K}(w)| = 1$). The loss minimizes the negative log-likelihood of the unique valid path:

$$
\mathcal{L}_{\text{core}} = -\log P(C_{k^*} | z) - \log P(w | C_{k^*}, z)
$$
where $k^*$ is the unique cluster of word $w$.

### Phase 2: Bridge Node Integration
Training includes **Bridge Nodes** (where $|\mathcal{K}(w)| > 1$). The loss treats the cluster prediction as a Multi-Label problem, permitting any valid path:

$$
\mathcal{L}_{\text{bridge}} = -\log \left( \sum_{k \in \mathcal{K}(w)} P(C_k | z) \cdot P(w | C_k, z) \right) + \lambda \sum_{k} |P(C_k | z)|
$$

*(Note: An L1 regularization term $\lambda$ is added to enforce sparsity in cluster selection.)*
