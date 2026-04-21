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
    "REAL": "#2A9D8F",
    "FAKE": "#E9C46A",
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


def stratified_sample(df, label_col, max_rows, random_state=42):
    if df.empty or label_col not in df.columns:
        return df.copy()

    n_classes = df[label_col].nunique()
    if n_classes == 0 or len(df) <= max_rows:
        return df.copy()

    per_class = max(max_rows // n_classes, 1)
    sampled_parts = []

    for label in sorted(df[label_col].dropna().unique()):
        subset = df[df[label_col] == label]
        sampled_parts.append(subset.sample(min(len(subset), per_class), random_state=random_state))

    sampled = pd.concat(sampled_parts, ignore_index=True)

    if len(sampled) > max_rows:
        sampled = sampled.sample(max_rows, random_state=random_state).reset_index(drop=True)

    return sampled


def safe_plot_df(df, needed_cols):
    existing = [col for col in needed_cols if col in df.columns]
    if len(existing) != len(needed_cols):
        return pd.DataFrame(columns=needed_cols)

    plot_df = df[needed_cols].copy()

    numeric_candidates = [
        "title_word_count",
        "text_word_count",
        "char_count",
        "title_caps_ratio",
        "title_punct_density",
        "lexical_diversity",
        "stopword_ratio",
        "repeated_word_ratio",
        "sentence_complexity",
        "punctuation_density",
        "percent_over_512",
        "count",
        "percent",
    ]

    for col in numeric_candidates:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    plot_df = plot_df.dropna()
    return plot_df


# -----------------------------
# Load processed deployment file
# -----------------------------
@st.cache_data
def load_data():
    file_path = "welfake_deploy.parquet"

    if not os.path.exists(file_path):
        return None, (
            "Could not find `welfake_deploy.parquet`. "
            "Keep it in the same folder as `RealvFake.py`."
        )

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        return None, f"Error loading parquet file: {e}"

    # Safety cleanup
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Ensure expected string columns exist
    if "title_clean" in df.columns:
        df["title_clean"] = df["title_clean"].fillna("").astype(str)
    else:
        df["title_clean"] = ""

    if "text_clean" in df.columns:
        df["text_clean"] = df["text_clean"].fillna("").astype(str)
    else:
        df["text_clean"] = ""

    if "label_name" not in df.columns and "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df["label_name"] = df["label"].map({1: "FAKE", 0: "REAL"}).fillna("UNKNOWN")

    numeric_cols = [
        "title_word_count",
        "text_word_count",
        "char_count",
        "title_caps_ratio",
        "title_punct_density",
        "lexical_diversity",
        "stopword_ratio",
        "repeated_word_ratio",
        "sentence_complexity",
        "punctuation_density",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Recreate title flags if needed
    if "has_question_mark_title" not in df.columns:
        df["has_question_mark_title"] = df["title_clean"].str.contains(r"\?", regex=True, na=False)

    if "has_exclamation_title" not in df.columns:
        df["has_exclamation_title"] = df["title_clean"].str.contains(r"!", regex=True, na=False)

    if "title_all_caps_heavy" not in df.columns:
        if "title_caps_ratio" in df.columns:
            df["title_all_caps_heavy"] = df["title_caps_ratio"] >= 0.30
        else:
            df["title_all_caps_heavy"] = False

    return df, None


# -----------------------------
# Load data
# -----------------------------
df, load_warning = load_data()

st.title("📰 Fake vs. Real News Analysis")
st.caption("Interactive Streamlit app built from a processed WELFake deployment dataset.")

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
chart_df = stratified_sample(filtered, "label_name", sample_size, random_state=42)

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

        counts_plot = safe_plot_df(counts_df, ["label_name", "count"])

        if not counts_plot.empty:
            fig = px.bar(
                counts_plot,
                x="label_name",
                y="count",
                text="count",
                color="label_name",
                color_discrete_map=COLOR_MAP,
                title="Class balance in the filtered view",
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Class balance chart could not be rendered for the current filter.")

    with right:
        summary_tbl = make_metric_table(analysis_df)
        if not summary_tbl.empty:
            st.dataframe(summary_tbl, width="stretch", hide_index=True)

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
            hist_df = safe_plot_df(chart_df, ["text_word_count", "label_name"])
            if not hist_df.empty:
                fig = px.histogram(
                    hist_df,
                    x="text_word_count",
                    color="label_name",
                    color_discrete_map=COLOR_MAP,
                    barmode="overlay",
                    nbins=40,
                    marginal="box",
                    title="Distribution of article text length",
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.warning("Text length histogram could not be rendered.")

            complexity_df = safe_plot_df(analysis_df, ["label_name", "sentence_complexity"])
            if not complexity_df.empty:
                fig2 = px.box(
                    complexity_df,
                    x="label_name",
                    y="sentence_complexity",
                    color="label_name",
                    color_discrete_map=COLOR_MAP,
                    points=False,
                    title="Sentence complexity by class",
                )
                st.plotly_chart(fig2, width="stretch")
            else:
                st.warning("Sentence complexity plot could not be rendered.")

        with right:
            bert_df = (
                filtered.groupby("label_name")["bert_over_512_words"]
                .mean()
                .mul(100)
                .round(1)
                .reset_index(name="percent_over_512")
            )
            bert_plot = safe_plot_df(bert_df, ["label_name", "percent_over_512"])

            if not bert_plot.empty:
                fig3 = px.bar(
                    bert_plot,
                    x="label_name",
                    y="percent_over_512",
                    color="label_name",
                    color_discrete_map=COLOR_MAP,
                    title="Percent of articles over 512 words",
                    text="percent_over_512",
                )
                fig3.update_traces(texttemplate="%{text}%", textposition="outside")
                st.plotly_chart(fig3, width="stretch")
            else:
                st.warning("BERT pressure chart could not be rendered.")

            char_df = safe_plot_df(chart_df, ["text_word_count", "char_count", "label_name", "title_clean"])
            if not char_df.empty:
                fig4 = px.scatter(
                    char_df,
                    x="text_word_count",
                    y="char_count",
                    color="label_name",
                    color_discrete_map=COLOR_MAP,
                    opacity=0.65,
                    hover_data=["title_clean"],
                    title="Text words vs character count",
                )
                st.plotly_chart(fig4, width="stretch")
            else:
                st.warning("Character count scatterplot could not be rendered.")

with tab3:
    st.subheader("Language signals")

    if analysis_df.empty:
        st.warning("No rows match the current filters.")
    else:
        l1, l2 = st.columns(2)

        with l1:
            lex_df = safe_plot_df(analysis_df, ["label_name", "lexical_diversity"])
            if not lex_df.empty:
                fig = px.box(
                    lex_df,
                    x="label_name",
                    y="lexical_diversity",
                    color="label_name",
                    color_discrete_map=COLOR_MAP,
                    points=False,
                    title="Lexical diversity",
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.warning("Lexical diversity plot could not be rendered.")

            rep_df = safe_plot_df(analysis_df, ["label_name", "repeated_word_ratio"])
            if not rep_df.empty:
                fig = px.box(
                    rep_df,
                    x="label_name",
                    y="repeated_word_ratio",
                    color="label_name",
                    color_discrete_map=COLOR_MAP,
                    points=False,
                    title="Repeated word ratio",
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.warning("Repeated word ratio plot could not be rendered.")

        with l2:
            stop_df = safe_plot_df(analysis_df, ["label_name", "stopword_ratio"])
            if not stop_df.empty:
                fig = px.box(
                    stop_df,
                    x="label_name",
                    y="stopword_ratio",
                    color="label_name",
                    color_discrete_map=COLOR_MAP,
                    points=False,
                    title="Stopword ratio",
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.warning("Stopword ratio plot could not be rendered.")

            punct_df = safe_plot_df(analysis_df, ["label_name", "punctuation_density"])
            if not punct_df.empty:
                fig = px.box(
                    punct_df,
                    x="label_name",
                    y="punctuation_density",
                    color="label_name",
                    color_discrete_map=COLOR_MAP,
                    points=False,
                    title="Punctuation density",
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.warning("Punctuation density plot could not be rendered.")

        scatter_df = safe_plot_df(chart_df, ["text_word_count", "lexical_diversity", "label_name", "title_clean"])
        if not scatter_df.empty:
            fig = px.scatter(
                scatter_df,
                x="text_word_count",
                y="lexical_diversity",
                color="label_name",
                color_discrete_map=COLOR_MAP,
                opacity=0.7,
                hover_data=["title_clean"],
                title="Article length vs lexical diversity",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Lexical diversity scatterplot could not be rendered.")

with tab4:
    st.subheader("Headline signals")

    if analysis_df.empty:
        st.warning("No rows match the current filters.")
    else:
        h1, h2 = st.columns(2)

        with h1:
            title_wc_df = safe_plot_df(analysis_df, ["label_name", "title_word_count"])
            if not title_wc_df.empty:
                fig = px.box(
                    title_wc_df,
                    x="label_name",
                    y="title_word_count",
                    color="label_name",
                    color_discrete_map=COLOR_MAP,
                    points=False,
                    title="Headline word count",
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.warning("Headline word count plot could not be rendered.")

            title_caps_df = safe_plot_df(analysis_df, ["label_name", "title_caps_ratio"])
            if not title_caps_df.empty:
                fig = px.box(
                    title_caps_df,
                    x="label_name",
                    y="title_caps_ratio",
                    color="label_name",
                    color_discrete_map=COLOR_MAP,
                    points=False,
                    title="Headline capitalization ratio",
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.warning("Headline capitalization plot could not be rendered.")

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

            title_flags_plot = safe_plot_df(title_flags, ["label_name", "signal", "percent"])
            if not title_flags_plot.empty:
                fig = px.bar(
                    title_flags_plot,
                    x="signal",
                    y="percent",
                    color="label_name",
                    color_discrete_map=COLOR_MAP,
                    barmode="group",
                    title="Share of titles with strong headline signals",
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.warning("Headline signal bar chart could not be rendered.")

            title_punct_df = safe_plot_df(analysis_df, ["label_name", "title_punct_density"])
            if not title_punct_df.empty:
                fig = px.box(
                    title_punct_df,
                    x="label_name",
                    y="title_punct_density",
                    color="label_name",
                    color_discrete_map=COLOR_MAP,
                    points=False,
                    title="Headline punctuation density",
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.warning("Headline punctuation density plot could not be rendered.")

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
            st.plotly_chart(fig, width="stretch")
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
            st.plotly_chart(fig, width="stretch")
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
        st.dataframe(hits.head(50), width="stretch", hide_index=True)

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
    st.dataframe(browser_df.head(200), width="stretch", hide_index=True)

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
    st.dataframe(df.head(100), width="stretch", hide_index=True)
