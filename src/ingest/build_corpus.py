"""Build parsed paper corpora with page-level and section-level metadata."""

import json
import os
import re
from typing import Dict, List, Any

PAPERS_METADATA = {
    "1706.03762v7": {
        "title": "Attention Is All You Need",
        "authors": [
            "Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit",
            "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"
        ],
        "version": "v7",
        "date": "2 Aug 2023",
        "arxiv_id": "1706.03762v7",
    },
    "2410.00037v1": {
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "authors": [
            "Alexandre Defossez", "Laurent Mazare", "Manu Orsini", "Amelie Royer",
            "Patrick Perez", "Herve Jegou", "Edouard Grave", "Neil Zeghidour"
        ],
        "version": "v1",
        "date": "17 Sep 2024",
        "arxiv_id": "2410.00037v1",
    },
    "2507.07955v2": {
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "authors": ["Sukjun Hwang", "Brandon Wang", "Albert Gu"],
        "version": "v2",
        "date": "15 Jul 2025",
        "arxiv_id": "2507.07955v2",
    },
}

