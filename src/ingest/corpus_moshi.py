"""Parsed data for 'Moshi: a speech-text foundation model for real-time dialogue' (arXiv:2410.00037v1)."""

import json
import os

PAGES = [
    {
        "page_number": 1,
        "section": "Abstract",
        "text": """Moshi: a speech-text foundation model for real-time dialogue
Alexandre Defossez, Laurent Mazare, Manu Orsini, Amelie Royer, Patrick Perez, Herve Jegou, Edouard Grave, Neil Zeghidour (Kyutai)
Abstract
We introduce Moshi, a speech-text foundation model and full-duplex spoken dialogue framework. Current systems for spoken dialogue rely on pipelines of independent components, namely voice activity detection, speech recognition, textual dialogue and text-to-speech. Such frameworks induce a latency of several seconds, lose non-linguistic information (emotion, non-speech sounds), and rely on turn segmentation failing on overlaps and interruptions. Moshi solves these by casting spoken dialogue as speech-to-speech generation. Starting from a text language model backbone, Moshi generates speech as tokens from the residual quantizer of a neural audio codec, while modeling separately its own speech and that of the user into parallel streams. This allows for the removal of explicit speaker turns and modeling arbitrary conversational dynamics. We extend hierarchical semantic-to-acoustic token generation to first predict time-aligned text tokens as a prefix to audio tokens ('Inner Monologue'). Theoretical latency of 160ms, 200ms in practice. Available at github.com/kyutai-labs/moshi."""
    },
    {
        "page_number": 2,
        "section": "1 Introduction",
        "text": """1 Introduction
Voice interfaces (Alexa, Siri, Google Assistant, ChatGPT voice) compound latency across pipeline components resulting in several seconds delay. Spoken conversations have an average response time of 230ms across 10 languages (Stivers et al., 2009). Furthermore, text bottleneck loses paralinguistic cues and turn-based segmentation cannot model 10-20% overlapping speech and backchanneling ('OK', 'I see').
Moshi augments a text LLM backbone with a smaller audio language model that ingests and predicts discrete audio units directly. Streaming hierarchical architecture with theoretical latency of 160ms. Multi-stream audio language model processes input and output audio jointly into two autoregressive token streams, removing explicit speaker turns."""
    },
    {
        "page_number": 3,
        "section": "Key Contributions & System Overview",
        "text": """Contributions:
1. Helium: 7B-parameter text LLM pretrained on 2.1T tokens of public English data.
2. Mimi: Neural audio codec converting 24kHz audio into discrete tokens at 12.5Hz (80ms stride) using split Residual Vector Quantization (RVQ), distilling WavLM semantic knowledge into the first level.
3. Moshi: Hierarchical streaming architecture combining Helium (Temporal Transformer) with a Depth Transformer.
4. Inner Monologue: Predicting time-aligned text tokens before audio tokens, providing scaffolding that dramatically improves linguistic quality and factuality, while enabling streaming ASR and TTS via delay tuning.
5. Full-duplex evaluation: SOTA on speech modeling and spoken question answering with several minutes of context (5 min)."""
    },
    {
        "page_number": 6,
        "section": "Figure 1 & 3 Model Overview",
        "text": """Figure 1: Overview of Moshi.
Frame rates: Audio in/out: 24kHz; Token streams: 12.5Hz.
Components:
- Mimi neural audio codec (codec in/out, stride 80ms)
- Helium Temporal Transformer (operates across time step s to s+1)
- Depth Transformer (RQ-Transformer operating across depth k at each time step)
- User audio input stream and Moshi audio output stream
- Inner Monologue: time-aligned text tokens predicted before audio tokens."""
    },
    {
        "page_number": 7,
        "section": "3.2 The Helium Text Language Model",
        "text": """3.2 The Helium Text Language Model
3.2.1 Architecture
Helium is an autoregressive Transformer language model:
- RMS normalization at input of attention blocks, feed-forward blocks, and output linear layer.
- Rotary Positional Embeddings (RoPE).
- Context length of 4,096 tokens.
- FlashAttention for efficient training.
- Gated Linear Units (GLU) with SiLU activation in feed-forward blocks.
- SentencePiece unigram tokenizer with 32,000 vocabulary targeting English, splitting all numbers into single digits and using byte-backoff.
- Optimizer: AdamW with fixed learning rate followed by cosine decay."""
    },
    {
        "page_number": 8,
        "section": "Table 1 Models Hyperparameters & Training Phases",
        "text": """Table 1: Hyperparameters for Helium and Moshi:
Temporal Transformer:
- Model dimension: 4096
- MLP dimension: 11264
- Number of heads: 32
- Number of layers: 32
- Context size: 4096 (Helium pre-training) / 3000 steps (~4 min for Moshi)
- Learning rate: 3e-4 (Helium pre-train), 3e-5 (Moshi pre-train), 3e-6 (post-train), 2e-6 (fisher & fine)
Depth Transformer:
- Model dimension: 1024
- MLP dimension: 4096
- Number of heads: 16
- Number of layers: 6
- Learning rate: 2e-4 (pre-train), 5e-5 (post-train), 4e-6 (fisher), 2e-6 (fine)
Input / Output space:
- Text cardinality: 32000
- Audio cardinality: 2048
- Frame rate: 12.5 Hz (80ms per frame)
Common parameters:
- Acoustic delay tau: 2 (pretraining), 1 (post-training, fisher, fine)
- Text delay: +/- 0.6 (pre-training), 0 (post-training, fisher, fine)
Training phases: (1) Unsupervised audio pre-training; (2) Simulated multi-stream post-training; (3) Fisher dataset fine-tuning; (4) Instruction fine-tuning on synthetic scripts."""
    },
    {
        "page_number": 10,
        "section": "3.3 Audio Tokenization & Mimi Architecture",
        "text": """3.3 Audio Tokenization: Mimi
3.3.1 Architecture
Mimi autoencoder consists of SeaNet and Residual Vector Quantizer (RVQ):
- 4 convolutional blocks with striding factors (4, 5, 6, 8) and final 1D convolution stride 2. Projects 24kHz waveform to 12.5 frames per second, dimension D = 512.
- Stride corresponds to 80ms per audio frame.
- Transformer-based bottleneck: 8 layers, 8 heads, RoPE encodings, 250 frames context (20s), GELU activations, d=512, MLP=2048, LayerScale init 0.01. Causal masking for streaming.
- Quantization rate: Q = 8 quantizers, codebook size NA = 2048 centroids. Bitrate = 1.1 kbps at 12.5Hz."""
    },
    {
        "page_number": 11,
        "section": "3.3.2 Split RVQ & Semantic-Acoustic Distillation",
        "text": """3.3.2 Learning semantic-acoustic tokens with a split RVQ
Distills semantic information from WavLM into first level of RVQ. WavLM produces 1024-dim embeddings at 50Hz from 16kHz audio; Mimi averages non-causally with stride 4 and kernel size 8 to match 12.5Hz.
Conflict between acoustic losses and semantic distillation is solved via Split RVQ:
Rather than a single RVQ with 8 levels, semantic information is distilled into a plain VQ and applied in parallel with an RVQ with 7 acoustic levels. Their outputs are summed for reconstruction."""
    },
    {
        "page_number": 13,
        "section": "3.4.1 RQ-Transformer Hierarchical Autoregressive Modeling",
        "text": """3.4.1 Hierarchical Autoregressive Modeling with RQ-Transformer
Sequence length of audio: Q=8 codebooks at 12.5Hz = 100 steps/sec. 5 min audio = 30,000 steps.
RQ-Transformer uses:
1. Temporal Transformer: maps past joint steps V_0 ... V_{s-1} to temporal context z_s in R^d.
2. Depth Transformer: maps z_s and earlier sub-sequences at current step (V_{s,1} ... V_{s,k-1}) to logits l_{s,k} in R^{N_k}.
l_{s,1} = Lin(z_s) in R^{N_1}
softmax(l_{s,k}) approximates P[V_{s,k} | V_0..V_{s-1}, V_{s,1}..V_{s,k-1}].
Number of steps in Temporal Transformer is S (not K*S), and Depth Transformer is at most K steps. Depthwise parametrization uses separate linear projection weights for each depth index k."""
    },
    {
        "page_number": 14,
        "section": "3.4.2 Audio Modeling & Acoustic Delay",
        "text": """3.4.2 Audio Modeling & Acoustic Delay
Introducing an acoustic delay tau between semantic and acoustic tokens reduces inter-dependence:
V_{s,1} = A_{s,1} (semantic)
V_{s,q} = A_{s-tau, q} if s >= tau + 1, q > 1 (acoustic)
V_{s,q} = 0 if s < tau + 1, q > 1.
Delay of tau = 1 or 2 steps (80ms or 160ms) greatly improves generation quality and stability."""
    },
    {
        "page_number": 15,
        "section": "3.4.3 Multi-Stream Modeling & 3.4.4 Inner Monologue",
        "text": """3.4.3 Multi-Stream Modeling
Two streams of audio: Moshi (A) and User (A'). Concatenated into V.
3.4.4 Inner Monologue
Moshi models the textual representation of its own speech as scaffolding. Text stream W derived from Whisper transcription tokenized with SentencePiece.
W inserted as first sub-sequence in V, acting as a prefix to semantic audio tokens.
Figure 4: Joint sequence modeled: 2Q + 1 streams total (17 streams when Q=8).
Columns: Step s. Depth: Text token (bottom) -> Moshi semantic -> Moshi acoustic (7) -> User semantic -> User acoustic (7)."""
    },
    {
        "page_number": 16,
        "section": "Aligning Text and Audio & Streaming ASR / TTS",
        "text": """Aligning text and audio tokens:
Audio operates at 12.5Hz. Words mapped to text tokens with start index t_i = start_timestamp / 12.5Hz. Special tokens PAD and EPAD. W is initialized with PAD; for word i, W_{t_i - 1} <- EPAD, W_{t_i + j} <- w_{i,j}.
PAD tokens represent ~65% of tokens in English conversational speech.
Deriving streaming ASR and TTS via token delay:
- ASR mode: Delay text by 2 seconds behind audio. Audio is fed from user, text tokens sampled freely -> Streaming ASR with 80ms word alignment.
- TTS mode: Delay audio by 2 seconds behind text. Text tokens fed into model, audio tokens sampled -> Streaming TTS.
Equation 6: Final joint sequences:
V_{s,1} = W_s (aligned text tokens)
V_{s,2} = A_{s,1} (Moshi semantic)
V_{s,1+q} = A_{s-tau, q} (Moshi acoustic, q=2..Q)
V_{s,1+Q+1} = A'_{s,1} (User semantic)
V_{s,1+Q+q} = A'_{s-tau, q} (User acoustic, q=2..Q)"""
    },
    {
        "page_number": 23,
        "section": "Table 4 Mimi Audio Quality Evaluation",
        "text": """Table 4: Audio quality evaluation for Mimi vs baselines:
- Mimi (adv. loss only): 24kHz, 12.5Hz, 1.1kbps, causal, ABX 8.7%, VisQOL 1.84, MOSNet 3.10, MUSHRA 81.0 +/- 1.3
- Mimi (non adv. only): 24kHz, 12.5Hz, 1.1kbps, causal, ABX 8.1%, VisQOL 2.82, MOSNet 2.89, MUSHRA 58.8 +/- 1.8
- RVQGAN: 24kHz, 75Hz, 1.5kbps, MUSHRA 31.3
- SpeechTokenizer: 16kHz, 50Hz, 1.5kbps, MUSHRA 45.1; 4.0kbps MUSHRA 74.3
- SemantiCodec: 16kHz, 50Hz, 1.3kbps, MUSHRA 64.8"""
    },
    {
        "page_number": 25,
        "section": "Table 5 & 6 RQ-Transformer & Delay Ablations",
        "text": """Table 5: RQ-Transformer ablations:
- Acoustic delay [0,2,2,2,2,2,2,2] without RQ-Transformer: Perplexity = 135.4
- Acoustic delay [0,2,2,2,2,2,2,2] with RQ-Transformer: Perplexity = 36.8
Table 6: Delay patterns, semantic weight, and Inner Monologue:
- Without Inner Monologue (delay [0,2,2,2,2,2,2,2], semantic weight 100, depthwise param): Transcript NLL = 3.65, length = 602 chars.
- With Inner Monologue (delay [0,2,2,2,2,2,2,2], semantic weight 100, depthwise param): Transcript NLL = 2.77, length = 1920 chars (dramatic improvement in linguistic quality and length)."""
    },
    {
        "page_number": 29,
        "section": "Table 8 Spoken Question Answering",
        "text": """Table 8: Spoken Question Answering evaluation:
- GSLM: Web Q = 1.5%, LLaMA Q = 4.0%
- AudioLM: Web Q = 2.3%, LLaMA Q = 7.0%
- SpeechGPT: Web Q = 6.5%, LLaMA Q = 21.6%, Audio Trivia QA = 14.8%
- Spectron (1B): Web Q = 6.1%, LLaMA Q = 22.9%
- Moshi (no Inner Monologue): Web Q = 9.2%, LLaMA Q = 21.0%, Audio Trivia QA = 7.3%
- Moshi (with Inner Monologue): Web Q = 26.6%, LLaMA Q = 62.3%, Audio Trivia QA = 22.8%
- Helium (text top line): Web Q = 32.3%, LLaMA Q = 75.0%, Audio Trivia QA = 56.4%"""
    },
    {
        "page_number": 32,
        "section": "5.8 Model Quantization (PTQ)",
        "text": """5.8 Compressing Moshi and Impact on Speech Quality
Post-Training Quantization (PTQ):
- Activations stored in BF16, dynamically quantized to 8 bits symmetric (AbsMax).
- Model weights quantized using asymmetric (MinMax) quantization with block size 32 or 256.
- Temporal Transformer and Depth Transformer quantized; embeddings, RMSNorms and Mimi left unquantized.
- Helium at W4A8 (block 32): 4.37GB, MMLU 52.97 (within 1.3 points of 54.3 BF16 baseline).
- Moshi at W4A8 (block 32): MMLU 45.7 (single-stream) / 42.2 (multi-stream instruct).
- W2A8 shows severe degradation (gibberish speech, noisy audio). W4A8 preserves speech quality with minimal artifacts (95.7% artifact-free)."""
    }
]

def save():
    out_dir = "data/parsed"
    os.makedirs(out_dir, exist_ok=True)
    doc = {
        "arxiv_id": "2410.00037v1",
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "version": "v1",
        "pages": PAGES
    }
    path = os.path.join(out_dir, "2410.00037v1.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"Saved {path} with {len(PAGES)} pages.")

if __name__ == "__main__":
    save()

