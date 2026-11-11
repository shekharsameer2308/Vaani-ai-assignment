"""Parsed data for 'Attention Is All You Need' (arXiv:1706.03762v7)."""

import json
import os

PAGES = [
    {
        "page_number": 1,
        "section": "Abstract",
        "text": """Attention Is All You Need
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data."""
    },
    {
        "page_number": 2,
        "section": "1 Introduction & 2 Background",
        "text": """1 Introduction
Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15].
Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states ht, as a function of the previous hidden state ht-1 and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples.
In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.
2 Background
The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU, ByteNet and ConvS2S, all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.
Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.
3 Model Architecture
Most competitive neural sequence transduction models have an encoder-decoder structure. Here, the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next."""
    },
    {
        "page_number": 3,
        "section": "3.1 Encoder and Decoder Stacks & 3.2 Attention",
        "text": """3.1 Encoder and Decoder Stacks
Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. We employ a residual connection around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel = 512.
Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.
3.2 Attention
An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key."""
    },
    {
        "page_number": 4,
        "section": "3.2.1 Scaled Dot-Product Attention & 3.2.2 Multi-Head Attention",
        "text": """3.2.1 Scaled Dot-Product Attention
We call our particular attention 'Scaled Dot-Product Attention'. The input consists of queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the query with all keys, divide each by sqrt(dk), and apply a softmax function to obtain the weights on the values.
In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V. We compute the matrix of outputs as:
Attention(Q, K, V) = softmax(Q K^T / sqrt(dk)) V
The two most commonly used attention functions are additive attention, and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of 1/sqrt(dk). Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.
While for small values of dk the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of dk. We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by 1/sqrt(dk).
3.2.2 Multi-Head Attention
Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional output values. These are concatenated and once again projected, resulting in the final values.
Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this."""
    },
    {
        "page_number": 5,
        "section": "3.2.3 Applications of Attention, 3.3 FFN, 3.4 Embeddings",
        "text": """MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
Where the projections are parameter matrices W_i^Q in R^{dmodel x dk}, W_i^K in R^{dmodel x dk}, W_i^V in R^{dmodel x dv} and W^O in R^{h dv x dmodel}.
In this work we employ h = 8 parallel attention layers, or heads. For each of these we use dk = dv = dmodel/h = 64. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.
3.2.3 Applications of Attention in our Model
The Transformer uses multi-head attention in three different ways:
1. In 'encoder-decoder attention' layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence.
2. The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder.
3. Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We prevent leftward information flow in the decoder to preserve auto-regressive property by masking out (setting to -infinity) all values in input of softmax which correspond to illegal connections.
3.3 Position-wise Feed-Forward Networks
In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between:
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality dff = 2048.
3.4 Embeddings and Softmax
We use learned embeddings to convert input and output tokens to vectors of dimension dmodel. We share the same weight matrix between the two embedding layers and the pre-softmax linear transformation. In the embedding layers, we multiply those weights by sqrt(dmodel)."""
    },
    {
        "page_number": 6,
        "section": "3.5 Positional Encoding & 4 Why Self-Attention",
        "text": """3.5 Positional Encoding
Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add 'positional encodings' to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension dmodel as the embeddings, so that the two can be summed.
In this work, we use sine and cosine functions of different frequencies:
PE(pos, 2i) = sin(pos / 10000^{2i/dmodel})
PE(pos, 2i+1) = cos(pos / 10000^{2i/dmodel})
where pos is the position and i is the dimension. The wavelengths form a geometric progression from 2pi to 10000 * 2pi. For any fixed offset k, PE_{pos+k} can be represented as a linear function of PE_{pos}.
Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations:
Self-Attention: Complexity per Layer O(n^2 * d), Sequential Operations O(1), Maximum Path Length O(1).
Recurrent: Complexity per Layer O(n * d^2), Sequential Operations O(n), Maximum Path Length O(n).
Convolutional: Complexity per Layer O(k * n * d^2), Sequential Operations O(1), Maximum Path Length O(log_k(n)).
Self-Attention (restricted): Complexity per Layer O(r * n * d), Sequential Operations O(1), Maximum Path Length O(n/r)."""
    },
    {
        "page_number": 7,
        "section": "5 Training & Hardware",
        "text": """5 Training
5.1 Training Data and Batching
We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding, which has a shared source-target vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary. Sentence pairs were batched together by approximate sequence length. Each training batch contained approximately 25000 source tokens and 25000 target tokens.
5.2 Hardware and Schedule
We trained our models on one machine with 8 NVIDIA P100 GPUs. For base models, each training step took about 0.4 seconds, trained for 100,000 steps or 12 hours. For big models, step time was 1.0 seconds, trained for 300,000 steps (3.5 days).
5.3 Optimizer
We used the Adam optimizer with beta_1 = 0.9, beta_2 = 0.98 and epsilon = 10^-9. We varied learning rate according to:
lrate = dmodel^{-0.5} * min(step_num^{-0.5}, step_num * warmup_steps^{-1.5}), where warmup_steps = 4000.
5.4 Regularization
Residual Dropout: Pdrop = 0.1 for base model.
Label Smoothing: eps_ls = 0.1 during training, which hurts perplexity as model learns to be more unsure, but improves accuracy and BLEU score."""
    },
    {
        "page_number": 8,
        "section": "6 Results: Machine Translation",
        "text": """6 Results
6.1 Machine Translation
On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. Training took 3.5 days on 8 P100 GPUs.
On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model. Transformer (big) for English-to-French used dropout rate Pdrop = 0.1 instead of 0.3.
Table 2 comparison:
Transformer (base model): BLEU EN-DE 27.3, EN-FR 38.1, Training Cost 3.3 * 10^18 FLOPs.
Transformer (big): BLEU EN-DE 28.4, EN-FR 41.8, Training Cost 2.3 * 10^19 FLOPs."""
    },
    {
        "page_number": 9,
        "section": "6.2 Model Variations & 6.3 Constituency Parsing",
        "text": """6.2 Model Variations
Table 3: Variations on Transformer architecture:
Base model: N=6, dmodel=512, dff=2048, h=8, dk=64, dv=64, Pdrop=0.1, eps_ls=0.1, 100K train steps, PPL 4.92, BLEU 25.8, 65M params.
Row (A): single head attention (h=1, dk=512) is 0.9 BLEU worse than best setting; quality also drops with too many heads (h=32).
Row (B): reducing attention key size dk hurts model quality.
Rows (C) and (D): bigger models are better, dropout is very helpful in avoiding over-fitting.
Row (E): sinusoidal positional encoding compared to learned positional embeddings produces nearly identical results (BLEU 25.7).
Big model: N=6, dmodel=1024, dff=4096, h=16, dk=64, dv=64, Pdrop=0.3, 300K train steps, PPL 4.33, BLEU 26.4, 213M params.
6.3 English Constituency Parsing
We trained a 4-layer transformer with dmodel = 1024 on the Wall Street Journal (WSJ) portion of Penn Treebank (40K training sentences). In semi-supervised setting with BerkleyParser corpora (17M sentences), Transformer achieved 92.7 F1 score, outperforming prior models."""
    },
    {
        "page_number": 10,
        "section": "7 Conclusion",
        "text": """7 Conclusion
In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.
For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art.
We are excited about the future of attention-based models and plan to apply them to other tasks, including images, audio and video. Making generation less sequential is another research goal of ours.
Code: https://github.com/tensorflow/tensor2tensor"""
    }
]

def save():
    out_dir = "data/parsed"
    os.makedirs(out_dir, exist_ok=True)
    doc = {
        "arxiv_id": "1706.03762v7",
        "title": "Attention Is All You Need",
        "version": "v7",
        "pages": PAGES
    }
    path = os.path.join(out_dir, "1706.03762v7.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"Saved {path} with {len(PAGES)} pages.")

if __name__ == "__main__":
    save()

