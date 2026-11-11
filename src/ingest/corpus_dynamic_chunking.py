"""Parsed data for 'Dynamic Chunking for End-to-End Hierarchical Sequence Modeling' (arXiv:2507.07955v2)."""

import json
import os

PAGES = [
    {
        "page_number": 1,
        "section": "Abstract & 1 Introduction",
        "text": """Dynamic Chunking for End-to-End Hierarchical Sequence Modeling
Sukjun Hwang, Brandon Wang, Albert Gu (Carnegie Mellon University, Cartesia AI)
Abstract
Major progress on language models (LMs) in recent years has largely resulted from moving away from specialized models designed for specific tasks, to general models based on powerful architectures (e.g. the Transformer) that learn everything from raw data. Despite this trend, pre-processing steps such as tokenization remain a barrier to true end-to-end foundation models. We introduce a collection of new techniques that enable a dynamic chunking mechanism which automatically learns content- and context-dependent segmentation strategies learned jointly with the rest of the model. Incorporating this into an explicit hierarchical network (H-Net) allows replacing the (implicitly hierarchical) tokenization-LM-detokenization pipeline with a single model learned fully end-to-end. When compute- and data-matched, an H-Net with one stage of hierarchy operating at the byte level outperforms a strong Transformer language model operating over BPE tokens. Iterating the hierarchy to multiple stages further increases its performance by modeling multiple levels of abstraction, demonstrating significantly better scaling with data and matching the token-based Transformer of twice its size. H-Nets pretrained on English show significantly increased character-level robustness, and qualitatively learn meaningful data-dependent chunking strategies without any heuristics or explicit supervision. Finally, the H-Net's improvement over tokenized pipelines is further increased in languages and modalities with weaker tokenization heuristics, such as Chinese and code, or DNA sequences (nearly 4x improvement in data efficiency over baselines).
1 Introduction
Fixed-vocabulary tokenization, such as byte-pair encoding (BPE), remains a pervasive handcrafted preprocessing step. Tokenization comes with drawbacks: poor character-level understanding, lack of meaning/interpretability, and degraded performance on complex languages and modalities. Replacing the tokenization-LM-detokenization pipeline with a single end-to-end model adheres better to the bitter lesson."""
    },
    {
        "page_number": 2,
        "section": "Introduction Continued: Motivation for Dynamic Chunking",
        "text": """Direct byte-level language modeling with isotropic architectures (MambaByte) incurs prohibitive computational costs. Hierarchical architectures (Hourglass, MegaByte) use simple pooling strategies that are not data-dependent and perform poorly on variable information rates. SpaceByte and Byte Latent Transformer (BLT) use external boundary predictors or entropy heuristics, which are modality-specific and not fully end-to-end. Trainable boundary predictors (DPT) exhibit training instabilities.
Dynamic Chunking: End-to-end Sequence Modeling Without Tokenization
We introduce H-Net, an end-to-end hierarchical network that compresses raw data through a recursive, data-dependent dynamic chunking (DC) process.
Hierarchical Processing: (i) raw data is processed by a small encoder network, (ii) downsampled and passed through a main network operating on compressed chunks, (iii) and finally upsampled before being passed through a decoder network operating on original resolution. Crucially, encoder and decoder networks are strongly improved by using state space models (SSMs), which have an inductive bias for compression.
Dynamic Chunking: Novel DC mechanism interfaces between main network and encoder/decoder networks. It is composed of (i) a routing module predicting boundaries through similarity score and (ii) a smoothing module interpolating representations using router outputs to attenuate uncertain boundaries. Combined with an auxiliary ratio loss targeting desired downsampling ratios and discrete gradient techniques (Straight-Through Estimator)."""
    },
    {
        "page_number": 3,
        "section": "Figure 1 & Dynamic Chunking Architecture Overview",
        "text": """Figure 1: Architectural overview of H-Net with a two-stage hierarchical design (S = 2).
Components of a chunking layer: (a) routing module for dynamically drawing chunk boundaries, (b) downsampler that selectively retains vectors based on boundary indicators, reducing sequence length while preserving semantically significant positions.
Components of a dechunking layer: (c) smoothing module for converting discrete chunks into interpolated representations, (d) upsampler that restores compressed vectors to original resolution based on boundary indicators.
With a single stage of dynamic chunking, a byte-level H-Net matches perplexity and downstream performance of a strong BPE-tokenized Transformer at sizes exceeding 1B parameters. Dynamic chunking naturally compresses data to a similar resolution as BPE tokenizers (4.5-5 bytes/chunk)."""
    },
    {
        "page_number": 4,
        "section": "Hierarchical Chunking: From Data to Abstractions & 2 H-Net Architecture",
        "text": """Hierarchical Chunking: From Data to Abstractions
Because H-Net is fully end-to-end, it can be iterated recursively: the main network can itself be an H-Net. More stages of chunking represent higher order meanings: characters into words, words into clauses/sentences.
A byte-level 2-stage H-Net overtakes the perplexity of a strong tokenized Transformer after just 30B training bytes, with the gap widening throughout training, and matches downstream evaluations of a tokenized Transformer of twice its size.
Benefits: Robustness against textual perturbations (HellaSwag); Interpretability (discovers semantically coherent units); Other languages (Chinese 59.9 -> 66.3 on XWinograd-zh; DNA 3.6x data efficiency).
2 H-Net Architecture
2.1 Architectural Overview
2.1.1 Components of H-Net
H-Net comprises three primary components: encoder networks (E), main network (M), and decoder networks (D). In an S-stage model, encoder networks are E^s, decoder networks D^s for 0 <= s < S, and main network M resides at stage s = S.
Overall pipeline:
x_hat^s = E^s(x^s), z_hat^S = M(x^S), z_hat^s = D^s(z^s) (Equation 1)
(x^{s+1}, p^s) = Chunk(x_hat^s) (Equation 2)
z^s = Dechunk(z_hat^{s+1}, p^s) + Linear(x_hat^s) (Equation 3)
p^s in [0, 1]^{L^s} represents chunking router's confidence that token should be passed into main stage."""
    },
    {
        "page_number": 5,
        "section": "2.1.2 Design Principles",
        "text": """2.1.2 Design Principles
Encoder and Decoder Networks: We employ Mamba-2 layers as primary building blocks for encoder and decoder networks. This yields two benefits: effective handling of fine-grained inputs and substantially improved efficiency when processing long uncompressed sequences. SSMs have a stronger inductive bias for compression which helps build abstractions.
Main Network: Concentrates majority of model capacity, operating on progressively compressed sequences L^S << L^0. We default to Transformer layers for the main network as compressed representations align well with Transformer strengths in processing discrete tokens, enabling controlled comparison with BPE Transformers.
Architectural Guidelines:
1. Model width is monotone in the hierarchy: D^0 <= D^1 <= ... <= D^S.
2. Outer networks stick to four layers (or equivalent of four Mamba layers).
Dimension changes handled by appending shared trainable vector when expanding (D^s -> D^{s+1}), and taking first D^s dimensions when reducing."""
    },
    {
        "page_number": 6,
        "section": "2.2 Dynamic Chunking (DC): Routing Module & Downsampler",
        "text": """2.2 Dynamic Chunking (DC)
2.2.1 Chunking Layer
Routing Module: Boundaries emerge at points of contextual or semantic shift. We measure similarity between adjacent representations using cosine similarity between adjacent encoder outputs:
q_t = W_q * x_hat_t, k_t = W_k * x_hat_t
p_t = 0.5 * (1 - (q_t^T k_{t-1}) / (||q_t|| ||k_{t-1}||)) in [0, 1]
b_t = 1_{p_t >= 0.5} (Equation 4)
where p_1 = 1.0 by definition ensuring sequence begins with a boundary. When consecutive vectors span a semantic boundary, projections q_t and k_{t-1} diverge in latent space, yielding low cosine similarity and high boundary probability p_t.
Downsampler: Compresses encoder outputs x_hat^s into reduced vectors x^{s+1} using boundary indicators {b_t}. Vectors where b_t = 1 are retained in x^{s+1}, while b_t = 0 are discarded. Boundary probabilities p^s are similarly compressed into P^{s+1} for dechunking."""
    },
    {
        "page_number": 7,
        "section": "2.2.2 Dechunking: Smoothing Module & Upsampler",
        "text": """2.2.2 Dechunking
Smoothing Module: Discrete chunk boundaries impede gradient flow. The smoothing module transforms discrete chunking operations into differentiable computations using exponential moving average (EMA):
z_bar_t = P_t * z_hat_t + (1 - P_t) * z_bar_{t-1} (Equation 5)
Smoothing performs: (1) Differentiable boundary learning without stochastic exploration; (2) Adaptive error correction (high confidence P_t ~ 1.0 maintains discrete boundaries, low confidence P_t ~ 0.5 smoothed); (3) Training stability preventing overfitting to suboptimal chunking patterns early in training.
Upsampler: Decompresses z_bar^{s+1} to match original resolution:
c_t = p_t^{b_t} * (1 - p_t)^{1 - b_t} = p_t if b_t = 1 else (1 - p_t) (Equation 6)
STE(c_t) = c_t + stopgradient(1 - c_t) (Equation 7)
z_tilde_t = z_bar_{sum_{k=1}^t b_k} (Equation 8, Causal expansion)
Upsampler(z_bar, c)_t = STE(c_t) * z_tilde_t (Equation 9)
Confidence scoring rewards high probability at b_t=1 and penalizes false boundaries at b_t=0. STE rounds confidence scores to 1.0 in forward pass while maintaining continuous gradients during backpropagation."""
    },
    {
        "page_number": 8,
        "section": "2.2.3 Ratio Loss & 2.3 Improved Techniques",
        "text": """2.2.3 Ratio Loss
Without regularization, model may retain nearly all vectors or compress excessively. Inspired by MoE load balancing, we introduce ratio loss:
L_ratio = (N / (N - 1)) * ((N - 1)*F*G + (1 - F)*(1 - G)) (Equation 10)
where F = (1/L) sum_{t=1}^L b_t (fraction of vectors selected), G = (1/L) sum_{t=1}^L p_t (average boundary probability), and N controls target compression ratio.
When F = G = 1/N, L_ratio achieves minimum of 1.
Total training loss: L = L_AR + alpha * sum_{s=0}^{S-1} L_ratio^s, where alpha = 0.03 in all experiments.
2.3 Improved Techniques for Hierarchical Sequence Modeling
Norm Balance: RMSNorm placed at the end of each network component (E^s, D^s, M) prevents deeper components from dominating residual signals through imbalanced feature norms."""
    },
    {
        "page_number": 9,
        "section": "2.3 Continued: Stream Separation & LR Modulation & 2.4 Autoregressive Properties",
        "text": """Separation of Two Streams: Adding projection (Linear) only to residual connection: z^s = Dechunk(z_hat^{s+1}, p^s) + Linear(x_hat^s). This maintains unimpeded gradients through main computational path while allowing residual connection to adapt. Residual connection initialized close to 0.
Learning Rate Modulation: Outer stages handling longer sequences receive proportionally higher learning rates than inner stages operating on compressed representations.
2.4 Autoregressive Training and Inference
Training: Causal masking across all sequence mixing layers. DC maintains causality by computing boundary probabilities based only on current and previous representations (q_t, k_t).
Inference: State generation during prefill. DC step:
(a) Routing Module needs k_{t-1}, state consists of key value of most recent token.
(b) Dechunking Layer needs P_t and z_tilde_{t-1}, state consists of last z_tilde value.
(c) Token generation: pass token through encoder, step router to determine if processed by main network, step main network if necessary, step dechunker, step decoder network. H-Net decides dynamically how much compute to allocate per token (resembling speculative decoding)."""
    },
    {
        "page_number": 11,
        "section": "Table 1 Architectures & Training Protocols",
        "text": """Table 1: Architectures for main language models, all data-/FLOP-matched.
GPT-3 Large scale (760M reference):
- Transformer: Token input, GPT-2 tokenizer, L0=1792, BPIC=4.6, 760M params, T24, dmodel=1536
- SpaceByte++: Byte input, Spacelike chunking, L0=8192, BPIC=6.0, 850M params, M4 + T28 + M4, dmodel=1024, 1536
- H-Net (1-stage): Byte input, 6-DC chunking, L0=8192, BPIC=4.8, 680M params, M4 + T22 + M4, dmodel=1024, 1536
- H-Net (2-stage): Byte input, (3,3)-DC chunking, L0=8192, BPIC=7.0, 870M params, M4 + T1M4 + T26 + M4T1 + M4, dmodel=1024, 1024, 1536
GPT-3 XL scale (1.3B reference):
- Transformer: Token input, GPT-2 tokenizer, L0=1792, BPIC=4.6, 1.3B params, T24, dmodel=2048
- H-Net (1-stage): Byte input, 6-DC, L0=8192, BPIC=4.7, 1.3B params, M4 + T24 + M4, dmodel=1024, 2048
- H-Net (2-stage): Byte input, (3,3)-DC, L0=8192, BPIC=6.9, 1.6B params, M4 + T1M4 + T27 + M4T1 + M4, dmodel=1024, 1536, 2048
Training data: 100B token subset from FineWeb-Edu dataset. Sequence length 8192 bytes for byte models, 1792 tokens for Transformer. Batch size 256. Optimizer AdamW with WSD schedule."""
    },
    {
        "page_number": 13,
        "section": "Table 2 Downstream Evaluations & BPB Crossover",
        "text": """Table 2: Zero-shot performance comparison across multiple benchmarks (all data-/FLOP-matched):
At Large scale (760M):
- Transformer (Token): FineWeb-Edu validation BPB = 0.756, Average accuracy = 53.3%
- SpaceByte++ (Byte): BPB = 0.760, Average accuracy = 53.6%
- H-Net (1-stage, Byte): BPB = 0.755, Average accuracy = 53.6%
- H-Net (2-stage, Byte): BPB = 0.743, Average accuracy = 55.5% (outperforms Transformer by +2.2%)
At XL scale (1.3B):
- Transformer (Token): BPB = 0.730, Average accuracy = 55.5%
- SpaceByte++: BPB = 0.733, Average accuracy = 56.1%
- H-Net (1-stage): BPB = 0.728, Average accuracy = 56.2%
- H-Net (2-stage): BPB = 0.715, Average accuracy = 58.2% (+2.7% over Transformer)
Figure 3 crossover points: H-Net (2-stage) overtakes BPE Transformer after 30B training bytes."""
    },
    {
        "page_number": 17,
        "section": "Table 4 Chinese & Code, Table 5 DNA Modeling",
        "text": """Table 4: Chinese and Code benchmarks:
Chinese (FineWeb-Edu-Chinese-V2.1):
- Transformer: BPIC 3.62, Val BPB 0.7404, XWinograd-zh acc 0.599
- H-Net (space): BPIC 3.38, Val BPB 0.7478, XWinograd-zh acc 0.629
- H-Net (2-stage): BPIC 5.81, Val BPB 0.7032, XWinograd-zh acc 0.663
Code (Pile Github):
- Transformer: BPIC 3.58, Val BPB 0.3376
- H-Net (space): BPIC 7.97, Val BPB 0.3163
- H-Net (2-stage): BPIC 7.23, Val BPB 0.3161
Table 5 & Figure 6 DNA Modeling on HG38:
H-Net models achieve same pre-decay perplexity as isotropic models with approximately 3.6x less data (3.6x data efficiency)."""
    },
    {
        "page_number": 18,
        "section": "3.3 Ablation Studies on Smoothing, Cosine Routing, STE",
        "text": """Figure 7 & Section 3.3 Ablations:
1. Smoothing module: removing smoothing module (w/o smoothing) causes compression ratios to fluctuate severely throughout training, leading to substantial performance degradation and training instability.
2. Cosine routing: replacing cosine similarity with direct probability prediction (w/o cosine routing) degrades performance.
3. Straight-Through Estimator: skipping STE (w/o STE) degrades training stability.
Figure 8 & 9: Pure Transformer outer networks (T2-T2) perform worst. Mamba outer networks (M4-M4) achieve optimal results and highest compression efficiency."""
    }
]

def save():
    out_dir = "data/parsed"
    os.makedirs(out_dir, exist_ok=True)
    doc = {
        "arxiv_id": "2507.07955v2",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "version": "v2",
        "pages": PAGES
    }
    path = os.path.join(out_dir, "2507.07955v2.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"Saved {path} with {len(PAGES)} pages.")

if __name__ == "__main__":
    save()

