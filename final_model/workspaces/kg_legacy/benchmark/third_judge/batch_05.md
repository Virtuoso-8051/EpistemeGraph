You are a meticulous scientific fact-checking judge. Three PDFs are attached: DeepWalk (Perozzi et al.,
2014), LINE (Tang et al., 2015) and node2vec (Grover and Leskovec, 2016). For EACH item below, decide whether the AI
ANSWER is faithful to these papers.

Rules
- A claim is SUPPORTED if the papers state or directly entail it, or if it is uncontroversial textbook background.
- It is UNSUPPORTED if it attributes to the papers something they do not contain or contradict.
- If the question contains a false premise and the answer goes along with it, that is UNSUPPORTED.
- If the question is not answerable from the papers, inventing specifics is UNSUPPORTED; saying the papers do not
  report it is SUPPORTED.
Verdict per item: SUPPORTED (no unsupported claim), PARTIALLY_SUPPORTED (some unsupported claims) or UNSUPPORTED
(the core of the answer is unsupported).

Reply ONLY with one line per item, exactly in this form and nothing else:
ID,VERDICT
(for example: B0042,PARTIALLY_SUPPORTED)

=== ITEM B0105 ===
QUESTION TYPE: FACTUAL - answerable from the papers
QUESTION: Why are the learned structural features said to be especially useful with simple machine learning algorithms, even though they can be used with any classifier?
AI ANSWER:
The learned structural features already capture the relevant patterns in the data, so the representation does much of the work that a complex model would otherwise have to do. Because the features are informative and make classes more separable, even simple, low-capacity learners (such as linear models, naive Bayes, or shallow decision trees) can reach good accuracy, with less tuning, less data, and lower computational cost. The features can be used with any classifier, but the benefit is largest with simple algorithms that cannot learn complex structure from raw inputs on their own.

=== ITEM B0076 ===
QUESTION TYPE: FACTUAL - answerable from the papers
QUESTION: Under which condition did LINE outperform node2vec in the operator-wise comparison, and what kinds of operators were involved?
AI ANSWER:
I could not identify the specific study this question refers to, so this is a low-confidence inference rather than a verified finding. The most plausible reading is that LINE beat node2vec mainly under the distance-based edge operators (Weighted-L1 and Weighted-L2), while node2vec tended to win with the Hadamard and Average operators. The operators involved would therefore be the distance-based ones.

=== ITEM B0013 ===
QUESTION TYPE: FACTUAL - answerable from the papers
QUESTION: Why does the passage say that edge sampling is useful for LINE’s training procedure?
AI ANSWER:
The passage most likely says edge sampling helps LINE's training because it turns a weighted network into a set of binary edges. Edges are sampled with probability proportional to their weights, and each sampled edge is treated as a unit-weight training example in stochastic gradient descent. This avoids the gradient explosion and high-variance updates that large or widely varying edge weights would cause, and it makes each update cheap and the optimization more stable and efficient. The passage itself was not provided, so this is based on the original LINE method rather than the specific text.

=== ITEM B0091 ===
QUESTION TYPE: FACTUAL - answerable from the papers
QUESTION: What conditional probability does LINE define for generating a context vertex vj from a vertex vi, and over what set is this distribution defined?
AI ANSWER:
LINE (second-order proximity) defines the conditional probability of generating context vertex v_j given vertex v_i as a softmax over inner products: p2(v_j|v_i) = exp(u_j^T u_i) / Σ_{k∈V} exp(u_k^T u_i), where u_i is the embedding of v_i and u_j is the context embedding of v_j. This distribution is defined over the entire vertex set V, so the denominator sums over all vertices in the network.

=== ITEM B0043 ===
QUESTION TYPE: FACTUAL - answerable from the papers
QUESTION: Which variants of LINE and LINE-SGD are limited to undirected graphs, and what proximity order does each variant use?
AI ANSWER:
The first-order variant is limited to undirected graphs. It uses first-order proximity, which models the symmetric similarity between the two endpoints of each observed edge. The second-order variant (second-order proximity, which models the similarity of neighborhood structures) is not restricted to undirected graphs and also handles directed ones. The question's 'LINE-SGD' label appears to refer to the SGD-optimized LINE in these two orders, so the first-order version is the undirected-only one.

=== ITEM B0110 ===
QUESTION TYPE: FACTUAL - answerable from the papers
QUESTION: Why is graph factorization described as scalable to large networks, and what graph type restriction does it have?
AI ANSWER:
Graph factorization (Ahmed et al., 2013) is described as scalable because it learns node embeddings by factorizing the graph's adjacency matrix using stochastic gradient descent that updates only on observed edges. Its cost and memory therefore grow with the number of edges rather than with the square of the number of nodes, which avoids forming or decomposing a dense n×n matrix. The optimization can also be parallelized or distributed across machines, which suits very large, sparse, real-world networks. Its restriction is that it is designed for undirected graphs, so it assumes a symmetric adjacency relationship and does not natively model edge direction. It is also primarily an approach that preserves first-order proximity (direct edge connections) and is typically applied to unweighted graphs.

=== ITEM B0012 ===
QUESTION TYPE: FACTUAL - answerable from the papers
QUESTION: Under which tasks and in what kinds of networks does node2vec demonstrate efficacy over existing state-of-the-art techniques?
AI ANSWER:
Node2vec (Grover & Leskovec, 2016) is a semi-supervised feature learning framework that learns continuous node representations via biased random walks and skip-gram. It demonstrates efficacy over existing state-of-the-art techniques (such as DeepWalk, LINE, and spectral/matrix-factorization methods) in multi-label node classification tasks (e.g., predicting labels of nodes like protein or article/category labels in BlogCatalog, PPI, and Wikipedia) and in link prediction tasks (predicting missing or future edges, using Hadamard and other binary operators over node embeddings). The authors report these gains across diverse networks, including biological networks (protein-protein interaction networks), social networks (e.g., BlogCatalog, Facebook, and other social graphs), and information/communication networks (e.g., Wikipedia co-occurrence and arXiv collaboration networks). Its advantage stems from the flexible, tunable walk parameters p and q, which let it interpolate between breadth-first (homophily/community-like structure) and depth-first (structural equivalence/role-like) exploration. Gains are generally most pronounced when the network's labels or links reflect such mixed homophily and structural equivalence, and the paper reports improvements in scalability as well as accuracy relative to baselines.

=== ITEM B0042 ===
QUESTION TYPE: FACTUAL - answerable from the papers
QUESTION: Under which experimental tasks are the node2vec feature representations evaluated according to the passage?
AI ANSWER:
The node2vec features are evaluated on two tasks: multi-label node classification (e.g., on BlogCatalog, PPI, and Wikipedia networks) and link prediction (e.g., on Facebook, PPI, and arXiv collaboration networks). The original paper also includes a qualitative visualization of a network. I could not see the passage, so this answer is based on the original node2vec paper rather than the specific text you referenced.

=== ITEM B0068 ===
QUESTION TYPE: FACTUAL - answerable from the papers
QUESTION: Why does the passage say that the community clusters discovered by node2vec closely relate to homophily in the character network?
AI ANSWER:
The passage itself was not included with the question, so I cannot quote its exact reasoning. In general, node2vec with low return parameter p and high in-out parameter q performs depth-first-like walks that stay within tightly knit neighborhoods. Embeddings of nodes in the same densely connected community therefore end up close together, so clustering recovers communities. Homophily, the tendency of connected characters to share traits, roles, or group membership, produces exactly this kind of densely linked community structure. The passage likely links the two because the clusters reflect characters who interact often and share attributes, which is what homophily predicts. If you share the passage, I can give its specific explanation.

=== ITEM B0034 ===
QUESTION TYPE: FACTUAL - answerable from the papers
QUESTION: Under what conditions does node2vec assign α_pq(t,x) equal to 1/p, 1, or 1/q when choosing the next node x?
AI ANSWER:
In node2vec's second-order biased random walk, a walk that has just traversed edge (t, v) is at current node v and chooses the next node x among v's neighbors. The unnormalized transition weight is π_vx = α_pq(t,x)·w_vx, where α_pq(t,x) depends on the distance d_tx between the previous node t and the candidate x:

- α_pq(t,x) = 1/p if d_tx = 0, i.e., x = t (the walk returns to the previous node).
- α_pq(t,x) = 1 if d_tx = 1, i.e., x is also a neighbor of t (a common neighbor of t and v, so the walk stays at a distance-1 'BFS-like' step from t).
- α_pq(t,x) = 1/q if d_tx = 2, i.e., x is not adjacent to t and is at distance 2 from it (the walk moves outward, 'DFS-like').

Here p is the return parameter (controls likelihood of immediately revisiting the previous node; high p discourages backtracking) and q is the in-out parameter (q > 1 keeps the walk local/BFS-like, q < 1 favors outward exploration/DFS-like). The weights are then normalized over the neighbors of v to give the transition probabilities.
