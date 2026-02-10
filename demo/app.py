from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


st.set_page_config(page_title="DINO-WM Quantization Replay", page_icon="🎯", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Manrope:wght@400;600;700&display=swap');
      :root {
        --bg-1: #f2f4ef;
        --bg-2: #dce5d8;
        --ink: #17261f;
        --accent: #0e7c66;
        --accent-2: #c16b28;
      }
      html, body, [class*="css"]  {
        font-family: 'Manrope', sans-serif;
        color: var(--ink);
      }
      .stApp {
        background: radial-gradient(circle at 10% 10%, var(--bg-2), var(--bg-1));
      }
      .title {
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        letter-spacing: 0.04em;
        font-size: 2rem;
        margin-bottom: 0.2rem;
      }
      .subtitle {
        color: #264238;
        margin-bottom: 0.8rem;
      }
      .metric-card {
        border: 1px solid #b4c9bb;
        border-radius: 14px;
        padding: 0.8rem;
        background: #f8fbf7cc;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

repo_root = Path(__file__).resolve().parent.parent
artifacts_root = repo_root / "demo" / "demo_artifacts"
manifest_path = artifacts_root / "manifest.json"

st.markdown('<div class="title">DINO-WM Quantization Replay Viewer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Replay precomputed Wall episodes and compare FP16 vs Uniform INT8 vs Mixed INT8.</div>',
    unsafe_allow_html=True,
)

if not manifest_path.exists():
    st.error(
        f"Missing manifest: {manifest_path}. Run `python experiments/export_demo_artifacts.py` first."
    )
    st.stop()

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
variants = manifest.get("variants", [])
episodes = manifest.get("episodes", [])

if not variants or not episodes:
    st.error("Manifest is empty. Export demo artifacts first.")
    st.stop()

left, right = st.columns([1, 2])

with left:
    episode_ids = [e["episode_id"] for e in episodes]
    episode_id = st.selectbox("Episode", options=episode_ids, index=0)
    variant = st.radio("Variant", options=variants, index=0)

    ep = next(e for e in episodes if e["episode_id"] == episode_id)
    source_idx = ep.get("source_episode_idx")
    vinfo = ep["variants"][variant]

    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.write(f"**Source idx:** {source_idx}")
    st.write(f"**Status:** {vinfo.get('status', 'unknown')}")
    st.write(f"**Success:** {bool(vinfo.get('success', False))}")
    st.write(f"**Opt steps:** {manifest.get('reference_opt_steps')}")
    st.write(f"**Seed:** {manifest.get('reference_seed')}")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Compare All Variants"):
        st.session_state["compare_all"] = True

with right:
    compare_all = st.session_state.get("compare_all", False)
    if compare_all:
        cols = st.columns(len(variants))
        for col, v in zip(cols, variants):
            with col:
                info = ep["variants"][v]
                st.caption(f"{v} | {info.get('status')}")
                video_path = artifacts_root / info["video_path"]
                if video_path.exists():
                    st.video(str(video_path))
                else:
                    st.warning(f"Missing {video_path}")
        if st.button("Single Variant View"):
            st.session_state["compare_all"] = False
    else:
        video_path = artifacts_root / vinfo["video_path"]
        if video_path.exists():
            st.video(str(video_path))
        else:
            st.warning(f"Missing {video_path}")
