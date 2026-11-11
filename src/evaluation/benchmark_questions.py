from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BenchmarkQuestion:
    id: str
    query: str
    target_paper_id: Optional[str]
    expected_answerable: bool
    expected_concepts: List[str]
    category: str


BENCHMARK_QUESTIONS: List[BenchmarkQuestion] = [
    # Attention Is All You Need (1706.03762v7)
    BenchmarkQuestion(
        id="q01_attention_formula",
        query="What is the exact mathematical formula for Scaled Dot-Product Attention in the Transformer?",
        target_paper_id="1706.03762v7",
        expected_answerable=True,
        expected_concepts=["softmax(Q K^T / sqrt(d_k)) V", "d_k", "scaling factor"],
        category="attention",
    ),
    BenchmarkQuestion(
        id="q02_transformer_layers",
        query="How many layers are in the encoder and decoder stacks of the base Transformer, and what is d_model?",
        target_paper_id="1706.03762v7",
        expected_answerable=True,
        expected_concepts=["N = 6", "d_model = 512", "identical layers"],
        category="attention",
    ),
    BenchmarkQuestion(
        id="q03_positional_encoding",
        query="What sinusoidal functions are used for positional encoding in the Transformer and why were they chosen over learned embeddings?",
        target_paper_id="1706.03762v7",
        expected_answerable=True,
        expected_concepts=["sin", "cos", "10000^(2i/d_model)", "extrapolate to sequence lengths longer"],
        category="attention",
    ),
    BenchmarkQuestion(
        id="q04_transformer_hardware_steps",
        query="On what hardware was the base Transformer model trained, how many steps, and how long did it take?",
        target_paper_id="1706.03762v7",
        expected_answerable=True,
        expected_concepts=["8 NVIDIA P100 GPUs", "100,000 steps", "12 hours"],
        category="attention",
    ),

    # Moshi (2410.00037v1)
    BenchmarkQuestion(
        id="q05_moshi_latency",
        query="What is the theoretical and practical conversational latency of Moshi?",
        target_paper_id="2410.00037v1",
        expected_answerable=True,
        expected_concepts=["160ms theoretical", "200ms in practice"],
        category="moshi",
    ),
    BenchmarkQuestion(
        id="q06_mimi_codec",
        query="What is Mimi's audio sampling rate, frame rate, number of RVQ quantizers, and bitrate?",
        target_paper_id="2410.00037v1",
        expected_answerable=True,
        expected_concepts=["24kHz", "12.5Hz", "Q = 8 quantizers", "1.1kbps", "80ms"],
        category="moshi",
    ),
    BenchmarkQuestion(
        id="q07_inner_monologue",
        query="How does Inner Monologue in Moshi align text and audio tokens, and what role do PAD and EPAD tokens play?",
        target_paper_id="2410.00037v1",
        expected_answerable=True,
        expected_concepts=["time-aligned text tokens", "prefix to audio", "PAD", "EPAD", "end of padding"],
        category="moshi",
    ),
    BenchmarkQuestion(
        id="q08_streaming_asr_tts_delay",
        query="How does changing the delay between text and audio tokens allow Moshi to switch between streaming ASR and TTS?",
        target_paper_id="2410.00037v1",
        expected_answerable=True,
        expected_concepts=["audio ahead of text gives ASR", "text ahead of audio gives TTS", "2 seconds delay"],
        category="moshi",
    ),

    # Dynamic Chunking (2507.07955v2)
    BenchmarkQuestion(
        id="q09_dynamic_chunking_routing",
        query="How does the routing module in Dynamic Chunking predict chunk boundaries using cosine similarity?",
        target_paper_id="2507.07955v2",
        expected_answerable=True,
        expected_concepts=["cosine similarity", "adjacent encoder outputs", "q_t^T k_{t-1}", "p_t", "b_t"],
        category="dynamic_chunking",
    ),
    BenchmarkQuestion(
        id="q10_smoothing_module_ema",
        query="Why is the smoothing module necessary in H-Net dechunking and how does it use exponential moving average?",
        target_paper_id="2507.07955v2",
        expected_answerable=True,
        expected_concepts=["discrete boundaries impede gradient flow", "EMA", "differentiable boundary learning", "P_t z_hat + (1-P_t) z_bar"],
        category="dynamic_chunking",
    ),
    BenchmarkQuestion(
        id="q11_ratio_loss",
        query="What is the ratio loss equation L_ratio in Dynamic Chunking and what is the weight alpha used during training?",
        target_paper_id="2507.07955v2",
        expected_answerable=True,
        expected_concepts=["N / (N-1)", "F", "G", "alpha = 0.03"],
        category="dynamic_chunking",
    ),
    BenchmarkQuestion(
        id="q12_hnet_dna_efficiency",
        query="How much more data-efficient is H-Net on human genome (HG38) DNA sequence modeling compared to isotropic baselines?",
        target_paper_id="2507.07955v2",
        expected_answerable=True,
        expected_concepts=["3.6x more data efficient", "HG38", "isotropic models"],
        category="dynamic_chunking",
    ),

    # Out-of-Domain / Unanswerable (Control Refusals)
    BenchmarkQuestion(
        id="q13_quantum_surface_codes",
        query="What is the fault-tolerant threshold theorem for surface codes in superconducting topological quantum memory?",
        target_paper_id=None,
        expected_answerable=False,
        expected_concepts=["REFUSAL"],
        category="refusal_control",
    ),
    BenchmarkQuestion(
        id="q14_rlhf_direct_preference_opt",
        query="How does Direct Preference Optimization (DPO) derive the implicit reward function from the Bradley-Terry preference model?",
        target_paper_id=None,
        expected_answerable=False,
        expected_concepts=["REFUSAL"],
        category="refusal_control",
    ),
    BenchmarkQuestion(
        id="q15_bitnet_ternary_quantization",
        query="What is the Absmean quantization function used for 1.58-bit ternary weights in BitNet b1.58?",
        target_paper_id=None,
        expected_answerable=False,
        expected_concepts=["REFUSAL"],
        category="refusal_control",
    ),
]

