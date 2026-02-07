Written with the help of Gemini.

# SocratesNSLM (Neuro-Symbolic Language Model)

This architecture consists of two distinct coupled systems:

1. **The Semantic Topology  (The Teacher):** A weighted graph constructed from corpus statistics that maps the manifold of semantic concepts.
2. **The Generative Model  (The Student):** A neural network whose training trajectory and output probability space are constrained by .

---

### I. The Semantic Topology ()

Let the corpus be a sequence of discrete symbols , where each  (the vocabulary of whole words).

#### 1. Graph Definition

We define an undirected weighted graph :

* **Vertices ():** The set of unique concepts (words) in the corpus, .
* **Edges ():** An edge  exists if and only if the words appear as direct neighbors (bigrams) in  and satisfy the significance threshold .

#### 2. Edge Weighting (Normalized PMI)

Let  be the probability of word  occurring in , and  be the joint probability of bigram . The Pointwise Mutual Information (PMI) is:


To bound the weights and penalize high-frequency stop-words, we use Normalized PMI (NPMI):



where .

#### 3. Topology Filtering (Entropy & Significance)

We define the **Filter Function**  to sparsify the graph.

* **Entropy Constraint:** Let  be the Shannon entropy of the neighbor distribution of . If , then  is a "Hub" (stopword) and is removed from the set of valid cluster centers.
* **Significance Constraint:** An edge  is retained only if:



where  and  are the mean and standard deviation of all positive NPMI scores, and  is the Z-score parameter (e.g., ).

#### 4. Semantic Partitioning (Clustering)

We partition  into  disjoint communities (semantic topics)  by maximizing the Modularity :



where  is the sum of all weights,  is the weighted degree of node ,  is the community of node , and  is the Kronecker delta.

---

### II. The Generative Model ()

The model is a Transformer parameterized by .

#### 1. Input Representation (Hybrid)

The input is a sequence of sub-word tokens  to handle open-vocabulary tasks.


#### 2. Hierarchical Output Structure

Unlike standard LLMs which compute a flat softmax over , this architecture utilizes a **Two-Stage Hierarchical Softmax** steered by the graph communities .

Let  be the final hidden state of the Transformer. The probability of predicting next word  is:


* **Stage 1: Cluster Prediction (The "Steering")**
A projection layer  maps  to the probability distribution over the  clusters:


* **Stage 2: Word Prediction (The "Specifics")**
Given a selected cluster , a cluster-specific projection  maps  to the words strictly within :



#### 3. Computational Complexity Analysis

Let  be the vocabulary size (e.g., 150,000).
Let  be the number of clusters (e.g., 2,000).
Let  be the average cluster size (e.g., 75).

* **Standard Softmax Cost:** 
* **Hierarchical Cost:** 

Since  and , the computational cost for the output layer is reduced by factor .

---

### III. The Training Curriculum (Steered Sampling)

Training is not performed on random batches. We define a **Steered Loss Function** .

Let a training batch  be constructed exclusively from sentences containing the centroid and neighbors of Cluster .

The loss function for this batch is:


1. **Cluster Loss (Coarse Steering):** Cross-entropy loss on predicting the correct topic ID .


2. **Word Loss (Fine-Grained):** Cross-entropy loss on predicting the correct word .



This ensures gradients flow through the "Topic" pathway first, stabilizing the "Concept" pathway.
