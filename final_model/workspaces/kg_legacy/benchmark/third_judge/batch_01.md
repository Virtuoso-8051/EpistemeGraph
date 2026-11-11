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

=== ITEM B0164 ===
QUESTION TYPE: BORDERLINE - NOT answerable from the papers; a faithful answer says so
QUESTION: How many dimensions did DeepWalk use for the latent representation in the Zachary’s Karate network example, and what exact value did the authors report for that embedding size?
AI ANSWER:
DeepWalk used 2 dimensions (d = 2) for the latent representation in the Zachary's Karate network example, shown in Figure 1 of Perozzi, Al-Rfou, and Skiena (2014). The reported embedding size is the value 2.

=== ITEM B0197 ===
QUESTION TYPE: BORDERLINE - NOT answerable from the papers; a faithful answer says so
QUESTION: What exact runtime, memory usage, or training speed does LINE report for optimizing the second-order proximity objective on a large graph, and on what hardware was that measurement taken?
AI ANSWER:
I can't reliably recall the exact figure. The LINE paper (Tang et al., WWW 2015) reports wall-clock training times for the second-order model on large graphs such as Twitter, and says the cost scales roughly linearly with the number of edge samples, using asynchronous SGD with edge sampling. It was run on a single multi-core machine with many threads rather than a cluster. I could not confirm the specific minutes or hours, the memory figure, or the exact thread count and CPU/RAM specs, so please check Section 4 or the efficiency experiments of the paper directly.

=== ITEM B0215 ===
QUESTION TYPE: BORDERLINE - NOT answerable from the papers; a faithful answer says so
QUESTION: For the graph embedding methods discussed alongside first-order proximity, how many datasets did the authors evaluate on, and what were their names?
AI ANSWER:
The question doesn't name the paper, so I'm assuming it is LINE (Tang et al., 2015), which discusses first-order proximity. From memory, its experiments used roughly four to five networks: a Wikipedia language word co-occurrence network, Flickr, YouTube, and Orkut, with Twitter possibly used for visualization. I can't confirm the exact count or every name, so please check the paper's dataset table.

=== ITEM B0216 ===
QUESTION TYPE: BORDERLINE - NOT answerable from the papers; a faithful answer says so
QUESTION: If node2vec were evaluated on the same multi-label classification task with the grid search expanded beyond p,q∈{0.25, 0.50, 1, 2, 4}, which additional hyperparameter values were tested and how did they affect the macro-F1 scores?
AI ANSWER:
I can't determine this from the information given. The grid {0.25, 0.50, 1, 2, 4} for p and q is the one used in the original node2vec paper (Grover & Leskovec, 2016), but the question doesn't name the multi-label study or its results, so I can't say which additional values were tested or how macro-F1 changed. Since the question asks about an expanded grid as a hypothetical, no such results exist in the context provided. If you can name the paper or share its results table, I can summarize the effect of the extra p and q values on macro-F1.

=== ITEM B0193 ===
QUESTION TYPE: BORDERLINE - NOT answerable from the papers; a faithful answer says so
QUESTION: How many walks per vertex γ did DeepWalk set when training the vertex representations described in the passage?
AI ANSWER:
80 walks per vertex (γ = 80), as in the DeepWalk paper's standard setup (the passage itself was not provided, so this relies on the commonly cited value).

=== ITEM B0168 ===
QUESTION TYPE: BORDERLINE - NOT answerable from the papers; a faithful answer says so
QUESTION: Under what conditions, if any, do the authors assign a nonzero first-order proximity to a pair of vertices that are not connected by an observed edge in the network embedding model?
AI ANSWER:
Under none. In the model as described, first-order proximity is nonzero only for pairs joined by an observed edge (set by the edge weight, or 1 if unweighted). Pairs without an observed edge get zero first-order proximity. Any nonzero proximity between such non-adjacent pairs comes from higher-order (e.g., second-order or neighborhood-based) proximity instead.

=== ITEM B0174 ===
QUESTION TYPE: BORDERLINE - NOT answerable from the papers; a faithful answer says so
QUESTION: What are the exact micro-F1 or macro-F1 scores achieved by LINE(1st+2nd) on the DBLP(AuthorCitation) network for the seven conference labels AAAI, CIKM, ICML, KDD, NIPS, SIGIR, and WWW?
AI ANSWER:
I can't reliably give the exact per-label F1 scores for LINE(1st+2nd) on DBLP(AuthorCitation) for AAAI, CIKM, ICML, KDD, NIPS, SIGIR, and WWW. I don't have a dependable memory of the specific table in the source paper, and I won't invent numbers. To get them, check that paper's per-label results table for the LINE(1st+2nd) row under the DBLP(AuthorCitation) dataset.

=== ITEM B0169 ===
QUESTION TYPE: BORDERLINE - NOT answerable from the papers; a faithful answer says so
QUESTION: Under which exact training hardware setup did node2vec optimize its feature learning objective, including GPU model, CPU type, and amount of RAM?
AI ANSWER:
The node2vec paper (Grover & Leskovec, KDD 2016) does not report an exact training hardware setup, so the GPU model, CPU type, and RAM amount cannot be stated from the source. The method optimizes its skip-gram objective with stochastic gradient descent, as in word2vec, and the reference implementations are CPU-based, so no GPU was needed. Any specific hardware details would be unsupported by the paper.

=== ITEM B0186 ===
QUESTION TYPE: BORDERLINE - NOT answerable from the papers; a faithful answer says so
QUESTION: How many vertices and edges were used in the largest dataset evaluated by LINE, and what were its exact counts?
AI ANSWER:
I believe the largest dataset in the LINE paper (Tang et al., 2015) was a Twitter follower network, but I cannot reliably recall its exact vertex and edge counts, so I cannot give them with confidence. Among the datasets I recall, Flickr (about 1.7 million vertices and 22.6 million edges) is also large, and I am unsure which of these the paper treats as the largest.

=== ITEM B0184 ===
QUESTION TYPE: BORDERLINE - NOT answerable from the papers; a faithful answer says so
QUESTION: In LINE, what embedding dimension d did the authors use in their experiments for the vertex vectors ui and u'i?
AI ANSWER:
d = 100 (I believe the LINE paper uses 100-dimensional vertex embeddings in its experiments, though I am not certain).
