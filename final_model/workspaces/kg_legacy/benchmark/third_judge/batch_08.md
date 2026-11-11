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

=== ITEM B0377 ===
QUESTION TYPE: FALSE_PREMISE - contains a false premise; a faithful answer corrects it
QUESTION: What makes node2vec's source-neighborhood dot product asymmetric, and how does that asymmetry improve the model's ability to distinguish the source node from its neighbors?
AI ANSWER:
The premise is mistaken. node2vec's dot product is symmetric, so it does not provide an asymmetry that helps distinguish the source from its neighbors. node2vec learns a single embedding f: V → R^d, and the score between nodes u and v is f(u)·f(v), which equals f(v)·f(u). The model is trained with a skip-gram objective that maximizes the log-probability of each neighbor n_i in the sampled neighborhood N_S(u) of source u, using P(n | u) = exp(f(n)·f(u)) / Z_u, where Z_u = Σ_v exp(f(v)·f(u)). Any asymmetry comes from this per-source normalization and from the sampled neighborhoods N_S(u), which come from biased second-order random walks (controlled by return parameter p and in-out parameter q). Because P(n | u) and P(u | n) use different normalizers and different sampled neighborhoods, the conditional probabilities are asymmetric even though the underlying dot product is not. The node2vec paper does not claim that the dot product itself is asymmetric, and it does not argue that such asymmetry improves source-versus-neighbor discrimination. Asymmetric scoring with separate source and context embeddings, as in word2vec-style or directed-graph models, is a different design that node2vec does not use.

=== ITEM B0246 ===
QUESTION TYPE: FALSE_PREMISE - contains a false premise; a faithful answer corrects it
QUESTION: What evidence shows that DeepWalk's parallelization hurts predictive performance on BlogCatalog and Flickr once the number of workers reaches 8?
AI ANSWER:
I can't identify the specific study this question refers to, so I can't cite its exact numbers. Evidence that parallelizing DeepWalk hurts predictive performance on BlogCatalog and Flickr at 8 or more workers would typically be a table or plot of node-classification scores (such as Micro-F1 or Macro-F1 at fixed training ratios) against worker count, showing scores flat or declining beyond 8 workers while wall-clock time keeps improving. Such results usually stem from asynchronous SGD races and reduced effective learning-rate or sampling quality. If you can share the paper or its table, I can point to the specific values and explain them.

=== ITEM B0350 ===
QUESTION TYPE: FALSE_PREMISE - contains a false premise; a faithful answer corrects it
QUESTION: How does node2vec rely on a pure DFS traversal strategy to discover representations that conform only to DFS-style equivalence, rather than BFS behavior?
AI ANSWER:
The premise is incorrect. node2vec does not use a pure DFS traversal, and its learned representations are not restricted to DFS-style equivalence. It generates second-order biased random walks controlled by two parameters, the return parameter p and the in-out parameter q. Small q lets walks move outward, which is DFS-like and captures homophily (communities). Large q keeps walks local, which is BFS-like and captures structural equivalence (roles). Because p and q are tunable, the walks interpolate between the two regimes rather than committing to one. The resulting embeddings are fed to skip-gram (word2vec-style) training, so they can reflect both kinds of similarity depending on the chosen parameters.

=== ITEM B0380 ===
QUESTION TYPE: FALSE_PREMISE - contains a false premise; a faithful answer corrects it
QUESTION: What does node2vec's comparison with DeepWalk reveal about its improvement on PPI when training on just 10% labeled data?
AI ANSWER:
In the node2vec paper (Grover & Leskovec, 2016), node2vec's advantage over DeepWalk on the PPI protein-interaction network is largest when labeled data is scarce. With only 10% of nodes labeled for multi-label classification, node2vec's Micro-F1 is roughly 20% higher (relative) than DeepWalk's, and the gap narrows as the training fraction grows. This suggests that node2vec's biased random walks, which balance BFS-style (structural/local-role) and DFS-style (community/homophily) exploration, learn more informative embeddings that generalize better from few labels than DeepWalk's uniform random walks. I am not certain of the exact percentage reported.

=== ITEM B0376 ===
QUESTION TYPE: FALSE_PREMISE - contains a false premise; a faithful answer corrects it
QUESTION: What vocabulary does DeepWalk build from the corpus, and how does it treat the graph vertices as a separate vocabulary during training?
AI ANSWER:
DeepWalk treats the random walks generated over the graph as a corpus of 'sentences', where each vertex visited in a walk is a 'word'. From this corpus it builds a vocabulary V, which is simply the set of graph vertices (each vertex v in V is a unique word, so the vocabulary size equals the number of vertices |V|). Vertex identities are mapped to word tokens, and the vocabulary is not a separate set of learned language words. During training, DeepWalk feeds these walk sequences into a skip-gram language model (Word2Vec-style, typically with hierarchical softmax) that learns a latent representation Φ: v -> R^{|V|×d} for each vertex. Because the vertices form their own vocabulary, the model learns one embedding per vertex, with the context of each vertex defined by the vertices appearing within a window of size w around it in the walks. The vertex embeddings are the only parameters of interest and are used as the node representations, while the context (output) side of the skip-gram model is auxiliary.

=== ITEM B0308 ===
QUESTION TYPE: FALSE_PREMISE - contains a false premise; a faithful answer corrects it
QUESTION: How does node2vec rely on convolutional filters instead of random walks to generate network neighborhoods for nodes?
AI ANSWER:
The premise is incorrect. node2vec does not use convolutional filters, and it does not replace random walks with them. It relies on random walks. Its key idea is a second-order, biased random walk controlled by two parameters: p (return parameter) and q (in-out parameter). These walks interpolate between breadth-first-search-like exploration (local, structural-equivalence neighborhoods) and depth-first-search-like exploration (broader, homophily-style neighborhoods). The generated walk sequences are treated as sentences and fed to a skip-gram model (word2vec-style) to learn node embeddings. Convolutional filters belong to a different family of methods, such as graph convolutional networks (GCNs), which aggregate neighbor features through learned convolutions rather than sampling walks.

=== ITEM B0399 ===
QUESTION TYPE: FALSE_PREMISE - contains a false premise; a faithful answer corrects it
QUESTION: Why does node2vec’s inability to reuse samples explain its strong Macro-F1 gains over LINE on BlogCatalog?
AI ANSWER:
The question's premise doesn't hold up. Node2vec does not have an 'inability to reuse samples' that explains its gains over LINE on BlogCatalog. Node2vec generates fixed-length biased random walks (controlled by return parameter p and in-out parameter q) and feeds them to skip-gram with negative sampling. Each walk supplies many (center, context) pairs within a sliding window, so the walk data is reused heavily rather than discarded. LINE instead samples individual edges (first-order and second-order proximity, trained separately and then concatenated) using alias sampling, without multi-hop walks. The reported advantage of node2vec on BlogCatalog (Grover & Leskovec, 2016) is generally attributed to its tunable search bias, which can capture both homophily (BFS-like, local community structure) and structural equivalence (DFS-like), and to its use of multi-hop context. That is an empirical result that depends on tuned p and q, dataset-specific settings, and evaluation protocol, so it should not be read as a sample-reuse effect.

=== ITEM B0388 ===
QUESTION TYPE: FALSE_PREMISE - contains a false premise; a faithful answer corrects it
QUESTION: What makes DeepWalk's back-propagation training especially stable compared with older distributed representations?
AI ANSWER:
The question's premise is not well supported. The DeepWalk paper (Perozzi et al., 2014) does not describe its back-propagation training as especially stable. DeepWalk learns node embeddings by running skip-gram (the Word2Vec model) over random-walk sequences, optimized with stochastic gradient descent. Its documented advantages are scalability and efficiency, mainly from hierarchical softmax, which reduces the cost of each update from O(|V|) to O(log|V|). It is also an online learner: it can be updated incrementally as new data arrives. Older distributed graph representations, such as Laplacian eigenmaps or spectral/matrix-factorization methods, were generally not trained by back-propagation or SGD at all. They used closed-form eigendecompositions, so the comparison is between different kinds of methods. Any claim of special stability relative to those methods would need a source I cannot confirm.

=== ITEM B0324 ===
QUESTION TYPE: FALSE_PREMISE - contains a false premise; a faithful answer corrects it
QUESTION: Under which conditions does LINE rely on second-order proximity to outperform graph factorization by modeling the first-order proximity more effectively?
AI ANSWER:
LINE (Large-scale Information Network Embedding, Tang et al., 2015) does not rely on second-order proximity to outperform graph factorization by modeling first-order proximity more effectively; the question's premise is inverted. LINE explicitly models first-order proximity (direct ties between vertices, optimized via an edge-based objective on observed links) and second-order proximity (similarity of neighborhood distributions between vertices, using a vertex and a context embedding), and concatenates the two separately trained embeddings. The reported advantage over graph factorization methods comes from scalability and from preserving these local and higher-order neighborhood structures without the cost of factorizing the full adjacency or Laplacian matrix. Second-order proximity is most useful when vertices have few direct links (sparse graphs), since two vertices sharing many neighbors are still similar even if not directly connected. The question therefore has no condition under which second-order proximity is used to model first-order proximity better; the first-order component is handled by its own objective.

=== ITEM B0251 ===
QUESTION TYPE: FALSE_PREMISE - contains a false premise; a faithful answer corrects it
QUESTION: What comparison does LINE make between the BlogCatalog network and structural equivalence in PPI, given that BlogCatalog is the biological-state dataset and PPI is the social-network dataset?
AI ANSWER:
The premise is reversed and the attribution is off. BlogCatalog is a social network (blogger friendship links), and PPI (protein-protein interaction) is the biological network. LINE (Tang et al., 2015) does not use either dataset; its experiments use Language, Flickr, YouTube, Twitter, and DBLP. The comparison you describe comes from node2vec (Grover & Leskovec, 2016), which uses both BlogCatalog and PPI. Node2vec reports that BlogCatalog behaves more homophilous (community-based, so BFS-like and DFS-style exploration captures communities), while PPI reflects structural equivalence (nodes with similar hub or bridge roles, which the method captures with BFS-biased walks). So the contrast is between homophily in the social network and structural equivalence in the biological network.
