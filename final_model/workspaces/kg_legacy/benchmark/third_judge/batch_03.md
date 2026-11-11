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

=== ITEM B0121 ===
QUESTION TYPE: CROSS_DOC - answerable by combining two of the papers
QUESTION: How do DeepWalk and LINE differ in the kind of network structure they use to learn node representations: what does DeepWalk treat as the equivalent of sentences, and what additional proximity does LINE explicitly optimize beyond observed links?
AI ANSWER:
DeepWalk treats truncated random walks over the graph as the equivalent of sentences: each walk is a sequence of node IDs, analogous to a sentence of words, and a skip-gram model with hierarchical softmax is trained so that nodes co-occurring within a window on these walks get similar embeddings. It thus captures higher-order neighborhood structure via walk co-occurrence, and it is designed for unweighted graphs. LINE instead works directly on the graph's edges (first-order proximity, the observed links between a pair of nodes, optimized by making the embedding similarity of linked nodes high) and additionally explicitly optimizes second-order proximity, which holds that two nodes are similar if they share many neighbors, i.e., their neighborhood distributions (context vectors) are similar. LINE trains these two objectives separately and concatenates the results, and it handles weighted edges via edge sampling rather than random walks.

=== ITEM B0127 ===
QUESTION TYPE: CROSS_DOC - answerable by combining two of the papers
QUESTION: Compare how node2vec and LINE relate their embedding objectives to network structure: what does node2vec emphasize about roles and communities, and what does LINE emphasize about first- versus second-order proximity in social networks?
AI ANSWER:
Node2vec (Grover & Leskovec, 2016) learns node embeddings by maximizing the likelihood of neighborhood co-occurrences in random walks, using the skip-gram objective with negative sampling. Its key contribution is a biased second-order random walk controlled by two parameters: return parameter p and in-out parameter q. Small q (BFS-like exploration) yields embeddings that capture homophily, i.e., nodes in the same community or tightly interconnected cluster end up close together. Large q (DFS-like exploration) yields embeddings that capture structural equivalence, so nodes playing similar structural roles (e.g., hubs, bridges, peripheral nodes) get similar representations even when they are far apart in the graph. Thus node2vec explicitly trades off community-style and role-style similarity, and it treats both as tunable aspects of network structure.

LINE (Tang et al., 2015, Large-scale Information Network Embedding) is designed for large, weighted, directed or undirected networks and explicitly defines two separate objectives. First-order proximity models the observed direct ties: for an edge (i,j), the joint probability of the two nodes being connected is defined from their embedding inner product (sigmoid), and the objective minimizes KL divergence to the empirical edge weight distribution, so only directly linked pairs are pulled together. Second-order proximity assumes that nodes sharing many neighbors are similar; each node has a vertex embedding and a separate context embedding, and the conditional probability of a context node given a node is trained to match the empirical neighborhood distribution. LINE concatenates separately trained first-order and second-order embeddings, emphasizing that first-order proximity captures direct social ties (friendship/follow links) while second-order proximity captures shared-neighborhood similarity (people with common friends are alike), and that both are needed for sparse, large social networks. LINE is edge-sampling based, which handles weighted graphs efficiently.

In short: node2vec emphasizes tunable community versus structural-role similarity via biased random walks, whereas LINE emphasizes an explicit split between direct (first-order) ties and shared-neighborhood (second-order) similarity, with LINE's objectives defined directly over edges and neighborhoods rather than over walk sequences.

=== ITEM B0154 ===
QUESTION TYPE: CROSS_DOC - answerable by combining two of the papers
QUESTION: What is the reported running time difference between LINE(2nd) and DeepWalk in the word analogy experiment, and how does that relate to node2vec’s claim about efficiency from random walks?
AI ANSWER:
I can't identify the specific paper or word analogy experiment this refers to, so I can't give the exact reported running-time difference between LINE(2nd) and DeepWalk. Generally, LINE(2nd) trains directly on sampled edges and does not generate random walks, so it avoids the walk-generation step that DeepWalk and node2vec use. node2vec's efficiency argument is that random walks let it sample local neighborhoods in a scalable way, so a reported gap between these methods would bear on that claim only if the numbers come from the same setup. To get the exact figures, check the paper's word analogy results table or running-time section.

=== ITEM B0123 ===
QUESTION TYPE: CROSS_DOC - answerable by combining two of the papers
QUESTION: Compared with LINE's handling of weighted edges, what is the key practical effect of increasing the random-walk parameter γ in the method from PASSAGE A, and when does that effect diminish?
AI ANSWER:
The question refers to "PASSAGE A," which was not provided, so I can't identify its method or its exact definition of γ. Going only on common conventions: LINE handles weighted graphs by sampling edges with probability proportional to their weight, so weights act directly as sampling frequencies. If the method in PASSAGE A uses a random walk whose transition probabilities scale edge weights by an exponent γ (proportional to w^γ), then increasing γ sharpens the bias toward heavier edges, so strong ties dominate the sampled neighborhoods and the embedding emphasizes them more. This effect diminishes when edge weights are already close to uniform, because raising equal weights to any power gives the same transition probabilities, so γ then has no effect. It also levels off at extreme values, where the walk becomes nearly deterministic (always choosing the heaviest edge). If PASSAGE A defines γ differently, such as a restart or return parameter, this answer would not apply.

=== ITEM B0156 ===
QUESTION TYPE: CROSS_DOC - answerable by combining two of the papers
QUESTION: What do node2vec and LINE each say about combining learned representations, and how do the combinations differ in what they are intended to represent?
AI ANSWER:
The two methods combine learned representations at different levels and for different purposes. node2vec combines the embeddings of two nodes into a single edge representation using simple binary operators: average, Hadamard (element-wise product), weighted L1 distance, and weighted L2 distance. These are meant to build edge features for link prediction, so the Hadamard product captures pairwise feature interactions, while the L1/L2 operators capture difference or distance between the endpoints. LINE instead combines representations at the node level. It trains first-order proximity embeddings (for direct, observed ties) and second-order proximity embeddings (for nodes sharing neighbourhood context) separately, then concatenates them into one vector per node. The concatenation is meant to preserve both local and higher-order structural proximity in a single node representation, without fusing the two signals into a shared space. In short, node2vec's combinations represent the relationship between a pair of nodes (edges), while LINE's combination represents a single node's multi-scale neighbourhood structure.

=== ITEM B0159 ===
QUESTION TYPE: CROSS_DOC - answerable by combining two of the papers
QUESTION: How do node2vec and the streaming variant differ in their treatment of a growing network, and what capability does the streaming variant add that node2vec alone does not explicitly mention?
AI ANSWER:
Node2vec is a static, batch method. It learns embeddings for a fixed graph by running biased random walks (controlled by the return parameter p and in-out parameter q) and then training a skip-gram model with negative sampling. It does not describe how to handle a network that changes over time. Incorporating new nodes or edges means rerunning the walks and retraining on the whole graph. The streaming variant treats the network as a sequence of arriving edges or nodes and updates the existing embeddings incrementally, without full retraining, so new nodes can be embedded and the embeddings kept current as the graph grows. The capability it adds that node2vec does not explicitly mention is this online or incremental updating of embeddings for a growing (streaming) graph, which allows new nodes and edges to be embedded on arrival at much lower cost. I am not certain which specific streaming node2vec paper is meant, so these details are general to that class of method.

=== ITEM B0133 ===
QUESTION TYPE: CROSS_DOC - answerable by combining two of the papers
QUESTION: Why would LINE be suitable for the BlogCatalog, PPI, and Wikipedia networks described in the datasets, and what property of LINE makes it applicable to those three networks regardless of whether they are social, biological, or word cooccurrence graphs?
AI ANSWER:
LINE (Large-scale Information Network Embedding; Tang et al., 2015) learns node embeddings from network structure alone, without requiring node attributes, labels, or any domain-specific semantics. It is designed for large-scale graphs and preserves both first-order proximity (directly connected nodes have similar embeddings, via an edge-based objective over observed edges) and second-order proximity (nodes sharing many neighbors have similar embeddings, via a context-vector objective). It optimizes these with edge sampling and negative sampling, which makes it efficient on large, weighted, and directed or undirected graphs. The BlogCatalog (social follower network), PPI (protein-protein interaction network), and Wikipedia (word co-occurrence network) datasets are all graphs whose informative signal lies in their edge structure and neighborhood overlap. Because LINE is model-agnostic about what the nodes and edges represent, it can be applied to all three regardless of whether the graph is social, biological, or linguistic. Its key property is therefore that it is a domain-agnostic, purely topology-driven, scalable embedding method that handles arbitrary (including weighted) networks, which is why it is used for multi-label node classification on these benchmarks.

=== ITEM B0129 ===
QUESTION TYPE: CROSS_DOC - answerable by combining two of the papers
QUESTION: Under which conditions does PASSAGE A report stable results for its random-walk sampling strategy, and how does that stability contrast with LINE's motivation for edge sampling?
AI ANSWER:
I can't answer this as asked because PASSAGE A is not included in your message, so I can't say under which conditions it reports stable results. Generally, LINE (Tang et al., 2015) motivated edge sampling by the problem that plain stochastic gradient descent on weighted edges has high variance when edge weights differ widely. It samples edges in proportion to their weights and treats each sampled edge as binary, which keeps the learning rate stable without weight-dependent gradient scaling. A random-walk strategy such as DeepWalk instead builds training pairs from co-occurrence within walks. If you paste PASSAGE A, I can give the specific conditions it reports and compare them directly with LINE's motivation.

=== ITEM B0151 ===
QUESTION TYPE: CROSS_DOC - answerable by combining two of the papers
QUESTION: Why do PASSAGE A and PASSAGE B both claim that their original neighborhood or context definitions are insufficient, and what replacement does each method use?
AI ANSWER:
The two passages were not included with your question, so I cannot tell what each one claims about its original neighborhood or context definition or which replacement each method uses. If you paste the text of PASSAGE A and PASSAGE B, I can compare their stated limitations and the replacements they propose.

=== ITEM B0142 ===
QUESTION TYPE: CROSS_DOC - answerable by combining two of the papers
QUESTION: How do DeepWalk and the model in paper1 differ in what structural information they explicitly preserve or exploit?
AI ANSWER:
I can't compare the two models because the question refers to "paper1," and no paper was provided. What I can say about DeepWalk: it learns node embeddings by generating truncated random walks over the graph and feeding them to a skip-gram model with hierarchical softmax, so nodes that co-occur within a short window of a walk get similar embeddings. This implicitly captures local neighborhood structure and, through multi-step walks, some higher-order proximity, but it does not explicitly model global structure, node attributes, edge weights, or community labels. To give a precise comparison, please provide paper1's title or text. Then I can state what structural information that model explicitly encodes (for example, first-order adjacency, higher-order proximity, structural roles, attributes, or global objectives) and contrast it with DeepWalk's walk-based co-occurrence.
