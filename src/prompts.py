# src/prompts.py
from typing import List
import re

def split_story_to_sentences(story: str, max_panels: int = 4) -> List[str]:
    """
    Naive splitter returning up to max_panels text chunks.
    Keeps order and roughly balances sentences across panels.
    """
    story = (story or "").strip().replace("\n", " ")
    if not story:
        return [""] * max_panels
    sentences = re.split(r'(?<=[.!?]) +', story)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= max_panels:
        return sentences + [""] * (max_panels - len(sentences))
    # Group into max_panels chunks
    chunks = [[] for _ in range(max_panels)]
    for i, s in enumerate(sentences):
        chunks[i % max_panels].append(s)
    panels = [" ".join(c).strip() for c in chunks]
    return panels

def build_prompt(panel_text: str, style: str = "anime", character_token: str = None) -> str:
    """
    Build a descriptive prompt for Stable Diffusion from panel text.
    """
    base = (panel_text or "").strip()
    if not base:
        base = "An expressive illustrative scene"
    if character_token:
        base = f"{character_token} in the scene: {base}"
    style_map = {
        "anime": "high-detail anime style, vibrant colors, cinematic lighting",
        "realistic": "photorealistic, ultra-detailed, natural lighting",
        "cartoon": "cartoon style, bold lines, flat colors"
    }
    style_desc = style_map.get(style, style)
    prompt = f"{base}. {style_desc}. 4k, ultra-detailed, dynamic composition, high contrast."
    return prompt
