# Paper Summary: Density and Collapse in Fraud-Detection GNNs

**Title:** Graph Density and Representational Collapse in Fraud-Detection Graph Neural Networks: Why Normalisation Matters More Than Head Initialisation

## Overview
Graph neural networks (GNNs) are widely applied to financial and review fraud detection. However, standard training practices such as weight initialisation and activation normalisation often borrow assumptions from image and text domains—assumptions that graph-structured data explicitly violates.

This paper systematically evaluates three classification-head initialisation schemes (Default, Xavier, Kaiming) against five normalisation strategies (None, BatchNorm, LayerNorm, GraphNorm, PairNorm) for three prominent architectures (GCN, GAT, and GraphSAGE). The evaluation spans 918 seeded runs across three benchmark datasets: the Elliptic Bitcoin transaction graph, YelpChi, and Amazon opinion-fraud graphs.

## Key Findings

### 1. Normalisation Trumps Initialisation
Normalisation plays a highly significant role for every tested architecture and dataset, accounting for up to 42% of the within-architecture variance ($\eta^2 = 0.42$). In contrast, the choice of classification head initialisation has virtually no main effect, only becoming relevant when normalisation has failed and representations have already collapsed.

### 2. Graph Density Drives Representational Collapse
By manipulating the relation-specific adjacencies of the review-fraud benchmarks (spanning a mean degree of 2.2 to 736.5) and randomly thinning edges, the study isolates the causal impact of graph density. The spread between random node pairs declines log-linearly with mean degree. For example, on YelpChi, the distance decays as $\bar{k}^{-0.7}$ for unnormalised GCN and GAT, and $\bar{k}^{-0.18}$ for GraphSAGE. Density dictates how much poor normalisation will hurt performance.

### 3. GraphNorm Prevents Collapse
Of the normalisation strategies tested, operators that standardise each feature's variance across nodes (such as GraphNorm) are vastly superior. GraphNorm successfully holds representation distance constant across densities. The benefit of proper normalisation grows from a modest boost on sparse graphs to the difference between functional detection and baseline failure on dense graphs. 

### 4. Density vs. Homophily
While graph density accurately predicts representational collapse (over-smoothing), it does not natively predict classification accuracy. Accuracy is instead independently governed by edge homophily—how informative the connections between nodes genuinely are.
