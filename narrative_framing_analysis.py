"""
narrative_framing_analysis.py
=============================================================================
Analysis of narrative framing in Arabic discrimination/racism discourse on
Facebook, and how audiences react to each frame.

This script reproduces the framing results (RQ2) and the audience-response
results (RQ3) reported in the paper, and regenerates every figure and table
in those two sections. It works from the FANAR-annotated post files: two
10k batches that overlap slightly, so the first thing we do is merge them
and drop the duplicates, which leaves 19,299 unique posts.

What it produces, in order:
  1. The frame distribution (how often each of the six frames occurs)
  2. Human-vs-FANAR agreement on the 1k validation sample (accuracy + kappa,
     a per-frame precision/recall/F1 table, and a confusion matrix figure)
  3. Content signatures per frame (message length, URL use, link posts)
  4. The most distinctive Arabic words for each frame (class-internal TF-IDF)
  5. Average engagement per frame
  6. The reaction mix per frame (shares of Love/Care/Haha/Wow/Sad/Angry)
  7. A chi-square test linking frame to the dominant reaction, plus a check
     that the result survives restricting to posts from 2016 on (the year
     Facebook reactions launched)

Outputs land in ./output/ : PNG figures and CSV tables.

Usage:
  pip install pandas numpy matplotlib scikit-learn scipy openpyxl
  python narrative_framing_analysis.py
=============================================================================
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # render to files, no display needed
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Settings — point these at your two annotated files, then run.
# ---------------------------------------------------------------------------
# File A is the first 10k AI-annotated batch.
# File B is the second batch; it also carries the 1,000 human labels in a
# "Manual annotation" column, which we use for validation.
FILE_A = "annotated_fanar_10kposts_Discrimination.csv"
FILE_B = "Discrimination and Racism annotated_fanar_1Kposts_min10.xlsx"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The six frames, kept in a fixed order so every table and figure lines up.
FRAMES = [
    "News/Reporting",
    "Moral Condemnation",
    "Personal Testimony",
    "Advocacy/Call to Action",
    "Justification/Normalization",
    "Other/Unclear",
]
# Short labels for the figure axes, where the full names don't fit.
FRAMES_SHORT = [
    "News/\nReporting", "Moral\nCondemn.", "Personal\nTestimony",
    "Advocacy", "Justif./\nNormal.", "Other/\nUnclear",
]
# The six non-Like reactions, in the order we stack them in the bar chart.
REACTIONS = ["Love", "Care", "Haha", "Wow", "Sad", "Angry"]


# ---------------------------------------------------------------------------
# Loading the files
# ---------------------------------------------------------------------------
def load_file(path):
    """
    Read one annotated file into a dataframe.

    The CSV exports hold Arabic posts with line breaks, quotes, and commas
    inside the message field. Python's default CSV reader splits on those and
    silently loses rows, so we use the slower-but-correct python engine with
    explicit quoting. Everything is read as text; we convert numbers later,
    where we actually need them.
    """
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(
            path, engine="python", quotechar='"', doublequote=True,
            skip_blank_lines=False, encoding="utf-8-sig",
            dtype=str, keep_default_na=False,
        )
    df.columns = [c.strip() for c in df.columns]   # headers have trailing spaces
    return df


def as_number(series):
    """Turn a text column into numbers, treating anything unparseable as 0."""
    return pd.to_numeric(series, errors="coerce").fillna(0)


def post_key(df):
    """
    Build a key that identifies the same post across both files.

    The URL is the natural unique id. A few posts have no URL, so for those
    we fall back to the message text. (We deliberately do NOT use the files'
    own post_id column — it turned out to be a per-file row number, so the
    same id points to different posts in the two files.)
    """
    url = df["URL"].astype(str).str.strip()
    msg = df["Message"].astype(str).str.strip()
    has_url = (url != "") & (url.str.lower() != "nan")
    return url.where(has_url, "MSG::" + msg)


def build_corpus():
    """
    Merge the two batches into one de-duplicated set of posts.

    Where a post appears in both files we keep the copy from File B, because
    that file carries the human labels and the Other/Unclear category. Then
    we add only the File A posts that aren't already present. Sorting at the
    end makes the row order deterministic, so repeated runs give identical
    numbers.
    """
    a = load_file(FILE_A)
    b = load_file(FILE_B)
    a["_key"] = post_key(a)
    b["_key"] = post_key(b)

    shared = set(a["_key"]) & set(b["_key"])
    corpus = pd.concat([b, a[~a["_key"].isin(set(b["_key"]))]], ignore_index=True)
    corpus = corpus.drop_duplicates("_key").sort_values("_key").reset_index(drop=True)

    # Normalise the frame label: anything outside the six-frame scheme
    # (a handful of stray free-text labels) is folded into Other/Unclear.
    label = corpus["llm_label"].astype(str).str.strip()
    corpus["frame"] = label.where(label.isin(FRAMES), "Other/Unclear")

    print(f"File A: {len(a):,} posts | File B: {len(b):,} posts")
    print(f"Shared between the two: {len(shared):,}")
    print(f"Unique posts after merge: {len(corpus):,}")
    return corpus, b


# ---------------------------------------------------------------------------
# Arabic text helpers (used for the keyword analysis)
# ---------------------------------------------------------------------------
# Matches Arabic diacritics and the tatweel elongation character.
_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u0640]")

# Common Arabic function words we don't want surfacing as "keywords".
ARABIC_STOPWORDS = (
    "في من على الى الي إلى عن مع هذا هذه ذلك التي الذي الذين ما لا و او أو أم ثم "
    "قد كل بعض غير بين هو هي هم انا نحن انت كان كانت يكون به له لها لهم عليه اذا "
    "حتي حتى لكن أن إن ان يا اي كما لقد منذ عند حيث ايضا الا نعم بل عبر خلال بعد "
    "قبل هناك هنا كيف لماذا متى اين الله يوم وقد وكان فقط جدا علي عنه لم لن ولا "
    "انه هل بن"
).split()


def normalize_arabic(text):
    """
    Light normalisation so spelling variants collapse together: strip
    diacritics, unify the alef / ya / ta-marbuta / hamza forms, and drop
    anything that isn't an Arabic letter or a space.
    """
    text = _DIACRITICS.sub("", str(text)).replace("\ufeff", "")
    text = re.sub("[إأآا]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = text.replace("ؤ", "و").replace("ئ", "ي")
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# RQ2 — part 1: how the frames are distributed
# ---------------------------------------------------------------------------
def frame_distribution(corpus):
    counts = corpus["frame"].value_counts().reindex(FRAMES)
    percents = counts / len(corpus) * 100

    print("\nFrame distribution")
    for frame in FRAMES:
        print(f"  {frame:<30} {int(counts[frame]):>6,}  ({percents[frame]:4.1f}%)")

    table = pd.DataFrame({"count": counts.astype(int), "percent": percents.round(1)})
    table.to_csv(f"{OUTPUT_DIR}/frame_distribution.csv")

    # Horizontal bar chart, labelled with the percentage on each bar.
    colors = ["#2A5F8F", "#B23A48", "#3C7A3C", "#8F5F2A", "#6A4C93", "#7A7A7A"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(FRAMES))
    ax.barh(y, percents.values, color=colors)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace("\n", " ") for f in FRAMES_SHORT])
    for i, v in enumerate(percents.values):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("% of annotated posts")
    ax.set_xlim(0, percents.max() * 1.15)
    ax.set_title(f"Distribution of narrative frames (N={len(corpus):,})")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig_frame_distribution.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# RQ2 — part 2: how well FANAR agrees with the human annotator
# ---------------------------------------------------------------------------
def annotation_reliability(file_b):
    """
    Compare FANAR's labels to the human labels on the 1,000-post validation
    sample. Reports overall accuracy and Cohen's kappa, a per-frame
    precision/recall/F1 table, and a row-normalised confusion matrix figure.
    """
    from sklearn.metrics import (
        cohen_kappa_score, precision_recall_fscore_support, confusion_matrix,
    )

    labelled = file_b[file_b["Manual annotation"].notna()].copy()
    human = labelled["Manual annotation"].astype(str).str.strip()
    fanar = labelled["llm_label"].astype(str).str.strip()
    fanar = fanar.where(fanar.isin(FRAMES), "Other/Unclear")

    accuracy = (human == fanar).mean()
    kappa = cohen_kappa_score(human, fanar)
    print(f"\nValidation on {len(labelled):,} posts")
    print(f"  Agreement: {accuracy * 100:.1f}%   Cohen's kappa: {kappa:.3f}")

    precision, recall, f1, support = precision_recall_fscore_support(
        human, fanar, labels=FRAMES, zero_division=0
    )
    table = pd.DataFrame(
        {"P": precision.round(2), "R": recall.round(2),
         "F1": f1.round(2), "n": support},
        index=FRAMES,
    )
    print(table.to_string())
    table.to_csv(f"{OUTPUT_DIR}/validation_per_frame.csv")

    # Confusion matrix, each row scaled to 100% so we can read "when the human
    # said X, where did FANAR put it?".
    matrix = confusion_matrix(human, fanar, labels=FRAMES).astype(float)
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    short = ["News", "Moral", "Pers", "Advoc", "Justif", "Other"]

    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(FRAMES)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(FRAMES)))
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("FANAR label")
    ax.set_ylabel("Human label")
    for i in range(len(FRAMES)):
        for j in range(len(FRAMES)):
            shade = "white" if matrix[i, j] > 0.5 else "black"
            ax.text(j, i, f"{matrix[i, j] * 100:.0f}", ha="center",
                    va="center", color=shade, fontsize=8)
    ax.set_title("Human vs. FANAR (row-normalised %)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig_confusion_matrix.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# RQ2 — part 3: what each frame looks like (format and vocabulary)
# ---------------------------------------------------------------------------
def content_signatures(corpus):
    """Mean message length, share with a URL, and share posted as links."""
    length = corpus["Message"].astype(str).str.len()
    has_url = corpus["Message"].astype(str).str.contains(r"https?://").astype(int) * 100
    is_link = corpus["Type"].astype(str).str.strip().str.lower().eq("link").astype(int) * 100

    table = pd.DataFrame({"frame": corpus["frame"], "Length": length,
                          "URL%": has_url, "Link%": is_link})
    summary = table.groupby("frame").mean().reindex(FRAMES).round(0)
    print("\nContent signatures by frame")
    print(summary.to_string())
    summary.to_csv(f"{OUTPUT_DIR}/content_signatures.csv")


def distinctive_words(corpus):
    """
    For each frame, the Arabic words that stand out within that frame's posts,
    ranked by mean TF-IDF. This is the lexical "fingerprint" of each frame.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    text = corpus["Message"].map(normalize_arabic)
    print("\nDistinctive words per frame")
    rows = []
    for frame in FRAMES:
        frame_text = text[corpus["frame"] == frame]
        frame_text = frame_text[frame_text.str.len() > 0]
        if len(frame_text) < 5:
            continue
        vectorizer = TfidfVectorizer(
            min_df=4, stop_words=ARABIC_STOPWORDS,
            token_pattern=r"(?u)\b\w\w+\b",
        )
        tfidf = vectorizer.fit_transform(frame_text)
        mean_scores = np.asarray(tfidf.mean(axis=0)).ravel()
        top = (pd.Series(mean_scores, index=vectorizer.get_feature_names_out())
               .sort_values(ascending=False).head(7))
        print(f"  {frame}: " + " | ".join(top.index))
        rows.append({"frame": frame, "top_words": " | ".join(top.index)})
    pd.DataFrame(rows).to_csv(f"{OUTPUT_DIR}/distinctive_words.csv", index=False,
                              encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# RQ3 — part 1: how much engagement each frame draws
# ---------------------------------------------------------------------------
def engagement_by_frame(corpus):
    for col in ["Total Interactions", "Comments", "Shares"]:
        corpus[col] = as_number(corpus[col])

    summary = (corpus.groupby("frame")[["Total Interactions", "Comments", "Shares"]]
               .mean().reindex(FRAMES).round(1))
    print("\nAverage engagement by frame")
    print(summary.to_string())
    summary.to_csv(f"{OUTPUT_DIR}/engagement_by_frame.csv")

    means = corpus.groupby("frame")["Total Interactions"].mean().reindex(FRAMES)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(range(len(FRAMES)), means.values, color="#2A5F8F")
    ax.set_xticks(range(len(FRAMES)))
    ax.set_xticklabels(FRAMES_SHORT, fontsize=8)
    ax.set_ylabel("mean total interactions")
    for i, v in enumerate(means.values):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Average engagement by narrative frame")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig_engagement_by_frame.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# RQ3 — part 2: the mix of reactions each frame draws
# ---------------------------------------------------------------------------
def reaction_composition(corpus):
    """
    For each frame, average the six reactions and turn them into shares that
    sum to 100%. This shows the *emotional colour* of the response, separate
    from how much engagement there is.
    """
    for reaction in REACTIONS:
        corpus[reaction] = as_number(corpus[reaction])

    means = corpus.groupby("frame")[REACTIONS].mean().reindex(FRAMES)
    shares = means.div(means.sum(axis=1), axis=0) * 100

    print("\nReaction composition by frame (%)")
    print(shares.round(1).to_string())
    shares.round(1).to_csv(f"{OUTPUT_DIR}/reaction_composition.csv")

    reaction_colors = {
        "Love": "#E8608A", "Care": "#9B72CF", "Haha": "#F2C14E",
        "Wow": "#F0A202", "Sad": "#5B8DEF", "Angry": "#D7263D",
    }
    fig, ax = plt.subplots(figsize=(8, 4.3))
    bottom = np.zeros(len(FRAMES))
    for reaction in REACTIONS:
        ax.bar(range(len(FRAMES)), shares[reaction].values, bottom=bottom,
               label=reaction, color=reaction_colors[reaction])
        bottom += shares[reaction].values
    ax.set_xticks(range(len(FRAMES)))
    ax.set_xticklabels(FRAMES_SHORT, fontsize=8)
    ax.set_ylabel("% of reactions")
    ax.set_title("Reaction composition by narrative frame")
    ax.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig_reaction_composition.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# RQ3 — part 3: is the frame/reaction link real, and is it a time artefact?
# ---------------------------------------------------------------------------
def frame_reaction_test(corpus):
    """
    Two checks on the reaction findings.

    First, a chi-square test of independence between the frame and each post's
    dominant reaction (the reaction it got most of), over the posts that
    received at least one reaction. We report chi-square against its degrees of
    freedom, the p-value, and Cramer's V for the effect size — significance at
    this sample size is cheap, so the effect size is what tells us how strong
    the link actually is.

    Second, because Facebook only introduced these reactions in 2016, we
    recompute the reaction shares on posts from 2016 on and check they barely
    move — i.e. the pooled result isn't an artefact of the rollout.
    """
    from scipy.stats import chi2_contingency

    for reaction in REACTIONS:
        corpus[reaction] = as_number(corpus[reaction])

    with_reaction = corpus[corpus[REACTIONS].sum(axis=1) > 0].copy()
    with_reaction["dominant"] = with_reaction[REACTIONS].idxmax(axis=1)

    table = pd.crosstab(with_reaction["frame"], with_reaction["dominant"])
    chi2, p_value, dof, _ = chi2_contingency(table)
    n = int(table.values.sum())
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))

    print("\nFrame vs. dominant reaction")
    print(f"  posts with at least one reaction: {n:,}")
    print(f"  chi-square = {chi2:.0f} (df = {dof}, expected ~{dof} under "
          f"independence)")
    print(f"  p = {p_value:.2e}   Cramer's V = {cramers_v:.3f}")

    # Robustness: does restricting to 2016+ change the reaction shares?
    date_col = "Post Created" if "Post Created" in corpus.columns else "Post Created Date"
    dates = pd.to_datetime(
        corpus[date_col].astype(str).str.replace(" AST", "", regex=False),
        errors="coerce",
    )
    year = dates.dt.year
    from_2016 = (year >= 2016)
    print(f"\n  posts from 2016 on: {from_2016.sum():,} "
          f"({from_2016.mean() * 100:.1f}%)")

    def shares(mask):
        means = corpus[mask].groupby("frame")[REACTIONS].mean().reindex(FRAMES)
        return (means.div(means.sum(axis=1), axis=0) * 100).round(1)

    all_years = shares(pd.Series(True, index=corpus.index))
    recent = shares(from_2016)
    comparison = pd.DataFrame({
        "Sad (all)": all_years["Sad"], "Sad (2016+)": recent["Sad"],
        "Angry (all)": all_years["Angry"], "Angry (2016+)": recent["Angry"],
    })
    print("  reaction shares barely move when restricted to 2016+:")
    print(comparison.to_string())


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------
def main():
    corpus, file_b = build_corpus()

    print("\n" + "=" * 60 + "\nRQ2 — Narrative framing\n" + "=" * 60)
    frame_distribution(corpus)
    annotation_reliability(file_b)
    content_signatures(corpus)
    distinctive_words(corpus)

    print("\n" + "=" * 60 + "\nRQ3 — Audience response\n" + "=" * 60)
    engagement_by_frame(corpus)
    reaction_composition(corpus)
    frame_reaction_test(corpus)

    print(f"\nDone. Figures and tables are in ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
