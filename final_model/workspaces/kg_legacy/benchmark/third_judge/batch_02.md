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

=== ITEM B0238 ===
QUESTION TYPE: BORDERLINE - NOT answerable from the papers; a faithful answer says so
QUESTION: How does LINE perform on a directed graph dataset compared with DeepWalk in terms of Micro-F1 or another downstream classification metric?
AI ANSWER:
I can't identify the specific directed-graph dataset or paper this question refers to, so I can't give an exact Micro-F1 comparison. In general, LINE (which models first- and second-order proximity directly from edges) tends to perform comparably to or modestly better than DeepWalk on directed graphs in node-classification Micro-F1, and its advantage is usually small and dataset-dependent. Treat this as a general tendency, not a verified result for a particular dataset; the source paper's table would be needed for an exact answer.

=== ITEM B0166 ===
QUESTION TYPE: BORDERLINE - NOT answerable from the papers; a faithful answer says so
QUESTION: Compared with DeepWalk, what exact number of parameters did node2vec use for its embedding model on the protein-protein interaction network?
AI ANSWER:
The node2vec paper (Grover & Leskovec, 2016) does not report an exact parameter count for its embedding model on the PPI network. Its PPI network has about 3,890 nodes, and it uses 128-dimensional embeddings by default. Under that setup the embedding matrix would have 3,890 × 128 = 497,920 parameters, and DeepWalk uses the same dimensionality in the paper's comparisons, so the count would be identical for both. This is an inferred figure, not one the paper states.

=== ITEM B0231 ===
QUESTION TYPE: BORDERLINE - NOT answerable from the papers; a faithful answer says so
QUESTION: How does LINE perform when the vertex importance weights λi are estimated with PageRank instead of being set to the degree of each vertex, and what quantitative improvement or drop is reported?
AI ANSWER:
I can't identify the specific result this question refers to, so I can't give a reliable quantitative figure. In the original LINE method (Tang et al., 2015), the vertex importance weights λi are not part of the objective. Negative sampling instead uses a vertex distribution proportional to degree raised to the 3/4 power, and edge sampling is weighted by edge weights. The question's setup, with λi set to degree versus PageRank, probably comes from a follow-up or variant paper that I cannot pin down. Please provide that paper or its table so I can report the exact change, whether a gain or a drop, and its size.

=== ITEM B0182 ===
QUESTION TYPE: BORDERLINE - NOT answerable from the papers; a faithful answer says so
QUESTION: How does the network embedding method quantify second-order proximity when two vertices do not share any common in-neighbors or out-neighbors, and what numerical value is used in that case?
AI ANSWER:
Second-order proximity is typically measured as the similarity between the vertices' neighborhood distributions, i.e., how similar their in-neighbor sets and out-neighbor sets are (for example, via a dot product, cosine, or Jaccard-style overlap of their neighborhood vectors). If two vertices share no common in-neighbors or out-neighbors, their neighborhood overlap is empty, so the second-order proximity is set to 0.

=== ITEM B0181 ===
QUESTION TYPE: BORDERLINE - NOT answerable from the papers; a faithful answer says so
QUESTION: How many random walks per vertex did DeepWalk use when learning latent social representations, and what exact walk length did the authors choose?
AI ANSWER:
DeepWalk used 80 random walks per vertex (γ = 80), and the walk length was 40 (t = 40). These were the values used across its experiments, alongside a window size of 10 and 128-dimensional embeddings.

=== ITEM B0177 ===
QUESTION TYPE: BORDERLINE - NOT answerable from the papers; a faithful answer says so
QUESTION: Under which hardware configuration did the authors train LINE, including the CPU model, RAM size, and number of cores?
AI ANSWER:
I can't reliably recall the exact hardware (CPU model, RAM, core count) reported for LINE (Tang et al., WWW 2015). I don't want to guess specific values. The paper's efficiency experiments were run on a single machine using multi-threaded asynchronous SGD, but I can't confirm the specifications. Check the paper's experimental setup section for the exact details.

=== ITEM B0212 ===
QUESTION TYPE: BORDERLINE - NOT answerable from the papers; a faithful answer says so
QUESTION: Compared with DeepWalk using 30 walks per node, what was the percentage improvement from increasing the number of walks per vertex to 50 on BlogCatalog?
AI ANSWER:
I cannot determine this reliably. I don't have the specific paper's reported numbers for DeepWalk on BlogCatalog at 30 versus 50 walks per node, so I can't give an exact percentage. Gains from more walks per node are usually small, often well under 5%, but that is a general expectation, not a sourced figure.

=== ITEM B0225 ===
QUESTION TYPE: BORDERLINE - NOT answerable from the papers; a faithful answer says so
QUESTION: Compared with DeepWalk, by how many percentage points did node2vec improve classification accuracy on the Cora dataset under the best hyperparameter setting?
AI ANSWER:
I can't determine this reliably. The original node2vec paper (Grover & Leskovec, 2016) did not evaluate on Cora; it used BlogCatalog, PPI, and Wikipedia. Any Cora comparison would come from a different paper, and I can't identify which one or its exact figures. A rough guess from typical reported results is that node2vec beat DeepWalk on Cora by about 2 to 3 percentage points, but this is not verified.

=== ITEM B0233 ===
QUESTION TYPE: BORDERLINE - NOT answerable from the papers; a faithful answer says so
QUESTION: In the network embedding framework that defines second-order proximity, what exact similarity function do the authors use to compare the neighborhood vectors p_u and p_v, and what is the formula for it?
AI ANSWER:
This matches LINE (Tang et al., 2015). The paper defines p_u as the first-order proximity vector of vertex u to all other vertices, and says second-order proximity is determined by the similarity between p_u and p_v. It does not give an explicit closed-form similarity between p_u and p_v. Instead it models each vertex's neighborhood as a context distribution: p2(v_j | v_i) = exp(u_j^T u_i) / Σ_k exp(u_k^T u_i), with u the context embeddings. It then minimizes the KL divergence between this conditional distribution and the empirical one, weighted by vertex degree. So the comparison is via a softmax over inner products with a KL-divergence objective, not a named similarity such as cosine. If the question presumes a specific named function like cosine, that presumption is not supported by the paper as I recall it.

=== ITEM B0175 ===
QUESTION TYPE: BORDERLINE - NOT answerable from the papers; a faithful answer says so
QUESTION: What exact embedding dimension did LINE use in its experiments on large-scale networks?
AI ANSWER:
128
