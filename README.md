# Narrative Framing of Arabic Discrimination Discourse — Analysis Code

This repository contains the analysis code for the framing and audience-response
results in our study of discrimination and racism discourse on Arabic Facebook.
It takes the FANAR-annotated posts and reproduces the frame distribution, the
human-validation numbers, the per-frame content and vocabulary signatures, and
the reaction analysis, along with every figure and table in those sections.

## What the script does

`narrative_framing_analysis.py` runs end to end and produces:

**Framing (RQ2)**
- The distribution of the six narrative frames across the corpus
- Agreement between FANAR and a human annotator on a 1,000-post validation
  sample (accuracy, Cohen's kappa, a per-frame precision/recall/F1 table, and a
  confusion-matrix figure)
- Content signatures per frame: mean message length, share of posts with a URL,
  and share posted as links
- The most distinctive Arabic words for each frame (class-internal TF-IDF)

**Audience response (RQ3)**
- Average engagement (interactions, comments, shares) per frame
- The reaction mix per frame — the shares of Love, Care, Haha, Wow, Sad and
  Angry
- A chi-square test of the link between frame and dominant reaction, reported
  with Cramer's V for the effect size
- A robustness check confirming the reaction results hold when the data is
  restricted to 2016 onward (the year Facebook reactions launched)

The two annotated files are two 10k batches that overlap on a few hundred posts;
the script merges them on the post URL and removes the duplicates, leaving 19,299
unique posts.

## Setup

```bash
pip install pandas numpy matplotlib scikit-learn scipy openpyxl
```

## Running it

Put the two annotated files next to the script (or edit the `FILE_A` and
`FILE_B` paths at the top), then:

```bash
python narrative_framing_analysis.py
```

Figures (PNG) and tables (CSV) are written to an `output/` folder, and a summary
of every number is printed to the console.

## Input files

- `annotated_fanar_10kposts_Discrimination.csv` — first annotated batch
- `Discrimination_and_Racism_annotated_fanar_1Kposts_min10.xlsx` — second batch;
  also holds the 1,000 human labels (in a `Manual annotation` column) used for
  validation

Both files carry the FANAR frame label in an `llm_label` column plus the
CrowdTangle post metadata (message text, URL, type, reactions, engagement).

## Notes

- The CSV reader uses Python's quoting-aware engine on purpose: the posts
  contain line breaks and commas inside the message field, which the default
  reader mis-splits.
- Duplicate detection is keyed on the post URL (falling back to the message text
  for the few posts without one), not on the files' own row id.
- Results are deterministic across runs.
