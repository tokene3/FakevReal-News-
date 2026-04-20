import glob
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

st.set_page_config(
    page_title="Fake vs. Real News Analysis",
    page_icon="📰",
    layout="wide",
)

# ---------------------------------
# Custom colors
# ---------------------------------
COLOR_MAP = {
    "REAL": "#2A9D8F",   # teal
    "FAKE": "#E9C46A",   # warm gold
}

ACCENT_GREEN = "#6BA292"
ACCENT_GOLD = "#D4A373"
ACCENT_PURPLE = "#7B6D8D"
ACCENT_SLATE = "#5C677D"

# -----------------------------
# Helpers
# -----------------------------
def tokenize(text: str):
    return re.findall(r"\b[a-zA-Z]{2,}\b", str(text).lower())


def get_fallback_series(df, column_name, default_value=""):
    if column_name in df.columns:
        return df[column_name].fillna(default_value).astype(str)
    return pd.Series([default_value] * len(df), index=df.index, dtype="object")


def pick_best_text(row):
    for col in ["text", "content", "abstract", "title"]:
        if col in row.index:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                return str(val)
    return ""


def top_terms(df, label, n=20):
    subset = df[df["label_name"] == label]["text_clean"].dropna().astype(str)
    tokens = []
    for doc in subset:
        tokens.extend(
            [t for t in tokenize(doc) if t not in ENGLISH_STOP_WORDS and len(t) > 2]
        )
    common = Counter(tokens).most_common(n)
    return pd.DataFrame(common, columns=["term", "count"])


def make_metric_table(df):
    metric_cols = [
        "title_word_count",
        "text_word_count",
        "title_caps_ratio",
        "title_punct_density",
        "lexical_diversity",
        "stopword_ratio",
        "repeated_word_ratio",
        "sentence_complexity",
        "punctuation_density",
        "bert_over_512_words",
    ]
    existing_metric_cols = [col for col in metric_cols if col in df.columns]

    if not existing_metric_cols or df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby("label_name")[existing_metric_cols]
        .median(numeric_only=True)
        .round(3)
        .reset_index()
    )

    if "bert_over_512_words" in df.columns:
        bert_pct = (
            df.groupby("label_name")["bert_over_512_words"]
            .mean()
            .mul(100)
            .round(1)
            .reset_index(name="bert_over_512_pct")
        )
        summary = summary.merge(bert_pct, on="label_name", how="left")

    return summary


# -----------------------------
# Load WELFake split files
# -----------------------------
@st.cache_data
def load_welfake_parts():
    search_patterns = [
         "WELFake_part_*.csv",
    os.path.join("data", "WELFake_part_*.csv"),
    os.path.join("WELFake_split_parts", "WELFake_part_*.csv"),
    os.path.join("WELFake_split_parts_pandas", "WELFake_part_*.csv"),  # ✅ THIS IS YOUR FOLDER
    ]

    file_paths = []
    for pattern in search_patterns:
        file_paths.extend(glob.glob(pattern))

    file_paths = sorted(list(set(file_paths)))

    if not file_paths:
        searched = "\n".join(search_patterns)
        return None, (
            "No split WELFake files were found.\n\n"
            "The app searched these locations:\n"
            f"{searched}\n\n"
            "Put the split files either:\n"
            "- in the same folder as RealvFake.py\n"
            "- in a folder named data\n"
            "- in a folder named WELFake_split_parts"
        )

    frames = []
    failed_files = []

    for path in file_paths:
        try:
            frames.append(pd.read_csv(path))
        except Exception as e:
            failed_files.append(f"{os.path.basename(path)}: {e}")

    if not frames:
        return None, "No WELFake files could be loaded."

    df = pd.concat(frames, ignore_index=True, sort=False)

    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    if "label" not in df.columns:
        return None, "The combined WELFake data does not contain a `label` column."

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df["label_name"] = df["label"].map({1: "FAKE", 0: "REAL"}).fillna("UNKNOWN")

    df["title_clean"] = get_fallback_series(df, "title", "")
    df["text_clean"] = df.apply(pick_best_text, axis=1)

    # Structural features
    df["title_word_count"] = df["title_clean"].apply(lambda x: len(str(x).split()))
    df["text_word_count"] = df["text_clean"].apply(lambda x: len(str(x).split()))
    df["char_count"] = df["text_clean"].apply(lambda x: len(str(x)))
    df["bert_over_512_words"] = df["text_word_count"] > 512

    def title_caps_ratio(text):
        words = str(text).split()
        if not words:
            return 0.0
        caps = sum(1 for w in words if w.isupper() and len(w) > 1)
        return caps / len(words)

    def punctuation_density(text):
        txt = str(text)
        words = txt.split()
        if not words:
            return 0.0
        punct = len(re.findall(r"[!?;:,\-]", txt))
        return punct / len(words)

    def title_punct_density(text):
        txt = str(text)
        words = txt.split()
        if not words:
            return 0.0
        punct = len(re.findall(r"[!?;:,\-]", txt))
        return punct / len(words)

    def lexical_diversity(text):
        tokens = tokenize(text)
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    def stopword_ratio(text):
        tokens = tokenize(text)
        if not tokens:
            return 0.0
        stop_ct = sum(1 for t in tokens if t in ENGLISH_STOP_WORDS)
        return stop_ct / len(tokens)

    def repeated_word_ratio(text):
        tokens = tokenize(text)
        if not tokens:
            return 0.0
        counts = Counter(tokens)
        repeated = sum(c for c in counts.values() if c > 1)
        return repeated / len(tokens)

    def sentence_complexity(text):
        txt = str(text).strip()
        if not txt:
            return 0.0
        sentences = [s.strip() for s in re.split(r"[.!?]+", txt) if s.strip()]
        words = txt.split()
        return len(words) / len(sentences) if sentences else 0.0

    df["title_caps_ratio"] = df["title_clean"].apply(title_caps_ratio)
    df["title_punct_density"] = df["title_clean"].apply(title_punct_density)
    df["lexical_diversity"] = df["text_clean"].apply(lexical_diversity)
    df["stopword_ratio"] = df["text_clean"].apply(stopword_ratio)
    df["repeated_word_ratio"] = df["text_clean"].apply(repeated_word_ratio)
    df["sentence_complexity"] = df["text_clean"].apply(sentence_complexity)
    df["punctuation_density"] = df["text_clean"].apply(punctuation_density)

    # Title signal flags
    df["has_question_mark_title"] = df["title_clean"].str.contains(r"\?", regex=True, na=False)
    df["has_exclamation_title"] = df["title_clean"].str.contains(r"!", regex=True, na=False)
    df["title_all_caps_heavy"] = df["title_caps_ratio"] >= 0.30

    warning_message = None
    if failed_files:
        warning_message = "Some files could not be loaded:\n" + "\n".join(failed_files)

    return df, warning_message


# -----------------------------
# Load data
# -----------------------------
df, load_warning = load_welfake_parts()

st.title("📰 Fake vs. Real News Analysis")
st.caption("Interactive Streamlit app built from the split WELFake dataset.")

if df is None:
    st.error(load_warning)
    st.stop()

if load_warning:
    st.warning(load_warning)

# -----------------------------
# About section
# -----------------------------
st.markdown(
    """
    ### About This Project
    This project explores differences between fake and real news using the WELFake dataset.
    It focuses on how article structure, headline style, and language patterns differ across
    the two classes. This matters because it helps reveal how misinformation is written,
    framed, and presented, which supports stronger media literacy and better detection systems.
    """
)

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Controls")
st.sidebar.markdown("FAKE and REAL are always shown together for direct comparison.")

word_min = int(df["text_word_count"].min()) if not df.empty else 0
word_max = int(df["text_word_count"].max()) if not df.empty else 100
default_upper = min(word_max, max(1500, word_min + 500))

word_range = st.sidebar.slider(
    "Article text word count",
    min_value=word_min,
    max_value=word_max,
    value=(word_min, default_upper),
)

bert_filter = st.sidebar.selectbox(
    "BERT-length pressure",
    options=["All", "Over 512 words only", "512 words or fewer"],
    index=0,
)

max_sample_upper = max(500, min(20000, len(df)))
sample_size_default = min(5000, max_sample_upper)

sample_size = st.sidebar.slider(
    "Max rows for live charts",
    min_value=500,
    max_value=max_sample_upper,
    value=sample_size_default,
    step=500,
)

balance_view = st.sidebar.checkbox(
    "Use balanced comparison sample",
    value=True,
    help="Downsamples the larger class so comparisons are more visually fair.",
)

show_raw = st.sidebar.checkbox("Show raw data preview", value=False)

filtered = df[df["text_word_count"].between(word_range[0], word_range[1])].copy()

if bert_filter == "Over 512 words only":
    filtered = filtered[filtered["bert_over_512_words"]].copy()
elif bert_filter == "512 words or fewer":
    filtered = filtered[~filtered["bert_over_512_words"]].copy()

# Live-chart downsampling for performance
chart_df = filtered.copy()
if len(chart_df) > sample_size and chart_df["label_name"].nunique() > 0:
    per_class = max(sample_size // chart_df["label_name"].nunique(), 1)
    chart_df = (
        chart_df.groupby("label_name", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), per_class), random_state=42))
        .reset_index(drop=True)
    )

# Balanced analysis set
if balance_view and filtered["label_name"].nunique() == 2:
    counts = filtered["label_name"].value_counts()
    min_count = counts.min()
    balanced_parts = []

    for lbl in counts.index:
        lbl_df = filtered[filtered["label_name"] == lbl]
        if len(lbl_df) >= min_count and min_count > 0:
            balanced_parts.append(lbl_df.sample(min_count, random_state=42))

    analysis_df = pd.concat(balanced_parts, ignore_index=True) if balanced_parts else filtered.copy()
else:
    analysis_df = filtered.copy()

# -----------------------------
# Top metrics
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total records", f"{len(df):,}")
col2.metric("Filtered records", f"{len(filtered):,}")
col3.metric("Real articles", f"{(df['label_name'] == 'REAL').sum():,}")
col4.metric("Fake articles", f"{(df['label_name'] == 'FAKE').sum():,}")

st.info(
    "WELFake supports deeper comparison than the smaller COVID dataset. "
    "This app focuses on article length, BERT pressure, headline signals, repetition, "
    "lexical diversity, punctuation, and the most common language patterns in each class."
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Executive Story",
        "Structure Signals",
        "Language Signals",
        "Headline Signals",
        "Term Explorer",
        "Article Browser",
    ]
)

with tab1:
    st.subheader("What the data says")

    left, right = st.columns([1.15, 1])

    with left:
        counts_df = (
            filtered["label_name"]
            .value_counts()
            .rename_axis("label_name")
            .reset_index(name="count")
        )

        fig = px.bar(
            counts_df,
            x="label_name",
            y="count",
            text="count",
            color="label_name",
            color_discrete_map=COLOR_MAP,
            title="Class balance in the filtered view",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        summary_tbl = make_metric_table(analysis_df)
        if not summary_tbl.empty:
            st.dataframe(summary_tbl, use_container_width=True, hide_index=True)

        st.markdown(
            """
            **How to read this app**
            - Start with class balance and median feature summaries.
            - Then compare article length, BERT pressure, and sentence complexity.
            - Then inspect language behavior like repetition, stopword usage, and lexical diversity.
            - Finally, study headline patterns and the most common terms in each class.
            """
        )

    if not analysis_df.empty:
        over_512 = (
            analysis_df.groupby("label_name")["bert_over_512_words"]
            .mean()
            .mul(100)
            .round(1)
            .to_dict()
        )
        med_text = (
            analysis_df.groupby("label_name")["text_word_count"]
            .median()
            .round(0)
            .to_dict()
        )
        med_complex = (
            analysis_df.groupby("label_name")["sentence_complexity"]
            .median()
            .round(1)
            .to_dict()
        )

        story_cols = st.columns(3)
        story_cols[0].markdown(f"**Article length**  \nMedian text words: {med_text}")
        story_cols[1].markdown(f"**BERT pressure**  \n% over 512 words: {over_512}")
        story_cols[2].markdown(f"**Sentence complexity**  \nMedian words per sentence: {med_complex}")

with tab2:
    st.subheader("Structure signals")

    if chart_df.empty:
        st.warning("No rows match the current filters.")
    else:
        left, right = st.columns(2)

        with left:
            fig = px.histogram(
                chart_df,
                x="text_word_count",
                color="label_name",
                color_discrete_map=COLOR_MAP,
                barmode="overlay",
                nbins=40,
                marginal="box",
                title="Distribution of article text length",
            )
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.box(
                analysis_df,
                x="label_name",
                y="sentence_complexity",
                color="label_name",
                color_discrete_map=COLOR_MAP,
                points="all",
                title="Sentence complexity by class",
            )
            st.plotly_chart(fig2, use_container_width=True)

        with right:
            bert_df = (
                filtered.groupby("label_name")["bert_over_512_words"]
                .mean()
                .mul(100)
                .round(1)
                .reset_index(name="percent_over_512")
            )
            fig3 = px.bar(
                bert_df,
                x="label_name",
                y="percent_over_512",
                color="label_name",
                color_discrete_map=COLOR_MAP,
                title="Percent of articles over 512 words",
                text="percent_over_512",
            )
            fig3.update_traces(texttemplate="%{text}%", textposition="outside")
            st.plotly_chart(fig3, use_container_width=True)

            fig4 = px.scatter(
                chart_df,
                x="text_word_count",
                y="char_count",
                color="label_name",
                color_discrete_map=COLOR_MAP,
                opacity=0.65,
                hover_data=["title_clean"],
                title="Text words vs character count",
            )
            st.plotly_chart(fig4, use_container_width=True)

with tab3:
    st.subheader("Language signals")

    if analysis_df.empty:
        st.warning("No rows match the current filters.")
    else:
        l1, l2 = st.columns(2)

        with l1:
            fig = px.box(
                analysis_df,
                x="label_name",
                y="lexical_diversity",
                color="label_name",
                color_discrete_map=COLOR_MAP,
                points="all",
                title="Lexical diversity",
            )
            st.plotly_chart(fig, use_container_width=True)

            fig = px.box(
                analysis_df,
                x="label_name",
                y="repeated_word_ratio",
                color="label_name",
                color_discrete_map=COLOR_MAP,
                points="all",
                title="Repeated word ratio",
            )
            st.plotly_chart(fig, use_container_width=True)

        with l2:
            fig = px.box(
                analysis_df,
                x="label_name",
                y="stopword_ratio",
                color="label_name",
                color_discrete_map=COLOR_MAP,
                points="all",
                title="Stopword ratio",
            )
            st.plotly_chart(fig, use_container_width=True)

            fig = px.box(
                analysis_df,
                x="label_name",
                y="punctuation_density",
                color="label_name",
                color_discrete_map=COLOR_MAP,
                points="all",
                title="Punctuation density",
            )
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            chart_df,
            x="text_word_count",
            y="lexical_diversity",
            color="label_name",
            color_discrete_map=COLOR_MAP,
            opacity=0.7,
            hover_data=["title_clean"],
            title="Article length vs lexical diversity",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Headline signals")

    if analysis_df.empty:
        st.warning("No rows match the current filters.")
    else:
        h1, h2 = st.columns(2)

        with h1:
            fig = px.box(
                analysis_df,
                x="label_name",
                y="title_word_count",
                color="label_name",
                color_discrete_map=COLOR_MAP,
                points="all",
                title="Headline word count",
            )
            st.plotly_chart(fig, use_container_width=True)

            fig = px.box(
                analysis_df,
                x="label_name",
                y="title_caps_ratio",
                color="label_name",
                color_discrete_map=COLOR_MAP,
                points="all",
                title="Headline capitalization ratio",
            )
            st.plotly_chart(fig, use_container_width=True)

        with h2:
            title_flags = (
                filtered.groupby("label_name")[["has_question_mark_title", "has_exclamation_title", "title_all_caps_heavy"]]
                .mean()
                .mul(100)
                .round(1)
                .reset_index()
                .melt(id_vars="label_name", var_name="signal", value_name="percent")
            )

            pretty_map = {
                "has_question_mark_title": "Question mark in title",
                "has_exclamation_title": "Exclamation mark in title",
                "title_all_caps_heavy": "Heavy ALL-CAPS title",
            }
            title_flags["signal"] = title_flags["signal"].map(pretty_map)

            fig = px.bar(
                title_flags,
                x="signal",
                y="percent",
                color="label_name",
                color_discrete_map=COLOR_MAP,
                barmode="group",
                title="Share of titles with strong headline signals",
            )
            st.plotly_chart(fig, use_container_width=True)

            fig = px.box(
                analysis_df,
                x="label_name",
                y="title_punct_density",
                color="label_name",
                color_discrete_map=COLOR_MAP,
                points="all",
                title="Headline punctuation density",
            )
            st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Term explorer")

    t1, t2 = st.columns(2)

    with t1:
        real_terms = top_terms(filtered, "REAL", 20)
        st.markdown("**Top REAL terms**")
        if not real_terms.empty:
            fig = px.bar(
                real_terms.sort_values("count"),
                x="count",
                y="term",
                orientation="h",
                title="Most common REAL terms",
                color_discrete_sequence=[ACCENT_GREEN],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No REAL terms available in the current filter.")

    with t2:
        fake_terms = top_terms(filtered, "FAKE", 20)
        st.markdown("**Top FAKE terms**")
        if not fake_terms.empty:
            fig = px.bar(
                fake_terms.sort_values("count"),
                x="count",
                y="term",
                orientation="h",
                title="Most common FAKE terms",
                color_discrete_sequence=[ACCENT_GOLD],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No FAKE terms available in the current filter.")

    keyword_search = st.text_input("Search for a word or phrase inside headlines or article text")
    if keyword_search:
        mask = (
            filtered["title_clean"].str.contains(keyword_search, case=False, na=False)
            | filtered["text_clean"].str.contains(keyword_search, case=False, na=False)
        )

        hits = filtered.loc[
            mask,
            [
                "label_name",
                "title_clean",
                "title_word_count",
                "text_word_count",
                "text_clean",
            ],
        ].copy()

        st.write(f"Matches found: {len(hits):,}")
        st.dataframe(hits.head(50), use_container_width=True, hide_index=True)

with tab6:
    st.subheader("Article browser")

    default_cols = [
        "label_name",
        "title_clean",
        "title_word_count",
        "text_word_count",
        "lexical_diversity",
        "sentence_complexity",
        "title_caps_ratio",
    ]
    valid_default_cols = [col for col in default_cols if col in filtered.columns]

    selected_cols = st.multiselect(
        "Columns to inspect",
        options=filtered.columns.tolist(),
        default=valid_default_cols,
    )

    browser_df = filtered[selected_cols].copy() if selected_cols else filtered.copy()
    st.dataframe(browser_df.head(200), use_container_width=True, hide_index=True)

    if len(filtered) > 0:
        inspect_index = st.number_input(
            "Inspect a specific filtered row index",
            min_value=0,
            max_value=len(filtered) - 1,
            value=0,
            step=1,
        )

        row = filtered.iloc[int(inspect_index)]
        st.markdown(f"**Label:** {row['label_name']}")
        st.markdown(f"**Title:** {row['title_clean']}")
        st.markdown(f"**Text word count:** {row['text_word_count']}")
        st.markdown("**Article text preview:**")
        st.write(str(row["text_clean"])[:4000])
    else:
        st.warning("No rows match the current filters.")

    st.download_button(
        "Download current filtered view as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="welfake_filtered_view.csv",
        mime="text/csv",
    )

if show_raw:
    st.subheader("Raw combined data preview")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)