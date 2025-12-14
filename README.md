# Knowledge Contribution Classification in Citation Contexts: Framework and 45-Year NLP Analysis

This repository provides the model code, evaluation code, and manually annotated dataset for the paper *"Knowledge Contribution Classification in Citation Contexts: Framework and 45-Year NLP Analysis"*.

---

## Overview

Understanding scientific knowledge flow and evolution is essential for assessing research impact and tracing disciplinary development. However, existing citation analysis methods focus on subjective citation intent, conflating heterogeneous knowledge types. Moreover, most studies analyze isolated contribution statements rather than actual citation contexts, yielding results disconnected from real knowledge dissemination scenarios.

To address these issues, we propose the **Natural Language Processing Knowledge Contribution Taxonomy (NLP-KCT)**. This framework identifies objective knowledge contribution types of cited works across four categories: **Method**, **Resource Tool**, **Empirical Findings**, and **Background**. By distinguishing core from non-core contributions, it reveals the hierarchical structure of knowledge dissemination.

**Key Results:**
- Dual-path hybrid fusion model achieves **85.5%** classification accuracy, outperforming general-purpose reasoning LLMs
- Analysis of **802,202** citations from ACL Anthology (1980–2024) reveals only 42.9% constitute core knowledge contribution edges
- Reconstructed citation networks render knowledge flow patterns more discernible
- Evolutionary analysis demonstrates that traditional methods gradually recede into disciplinary background while paradigm innovations achieve enduring influence by becoming reusable tools


---

## Knowledge Contribution Taxonomy 

Our classification framework (ACL-KCT) moves beyond subjective citation intent to focus on the objective, verifiable knowledge contribution of a cited paper. We define four categories, including three **Core Knowledge Contributions** and one **Non-core** contribution.

| Contribution Type | Definition | Example |
| :--- | :--- | :--- |
| **Method** (Core) | Citation where the citing paper ("we") directly uses, implements, improves, or extends methods, techniques, algorithms, models, systems, or evaluation metrics from the cited paper. | "We compute COMET scores [CITATION] separately for each domain with default wmt20-comet-da..." |
| **Resource Tool** (Core) | Citation where the citing paper ("we") uses datasets, corpora, or explicitly labeled toolkits/toolboxes from the cited paper. | "We first extracted opinionated and objective texts from DeReKo corpus [CITATION]" |
| **Empirical Findings** (Core) | Citation providing specific, verifiable empirical findings (results, data) used for: 1) direct comparison, 2) justifying decisions, or 3) stating empirical facts. | "Table 2 shows that... our parser first outperforms the state-of-the-art segmentator of [CITATION]" |
| **Research Background** (Non-core) | Citation providing necessary context by: 1) outlining the developmental course or 2) describing the research landscape. It is descriptive rather than operational. | "Contemporary MT evaluation measures have evolved beyond simple lexical matching... [CITATION1], [CITATION2]" |

---

## Dataset Overview (802,202 Instances)

Figure 1 shows the statistical features of the 802,202 citation instances. The data shows an exponential growth in publications (Fig. 1a), a high concentration of citations in the *Introduction* (44.0%) and *Related Work* (34.3%) sections (Fig. 1b), and a normal distribution of context length (Mean: 434 chars) (Fig. 1c).

<p align="center">
  <img width="2199" height="792" alt="Figure 1: Data Statistics" src="https://github.com/user-attachments/assets/003b1f19-46c3-4bc7-960e-3f9b71f0fef1" />
  <br>
  <em>Figure 1: Detailed statistics of the ACL Anthology data (1980-2024)</em>
</p>

## Gold Standard & Annotation Quality (2,000 Samples)

A 2,000-instance gold standard set was manually annotated to train and validate our models. The annotation quality was rigorously assessed, achieving an **Average Cohen's Kappa of 0.770** and a **Fleiss' Kappa of 0.770** (Fig. 2a, 2b), indicating "excellent" inter-annotator agreement. The distribution of this gold set is shown in Fig. 2c and 2d.

<p align="center">
  <img width="1958" height="588" alt="Figure 2: Annotation Quality" src="https://github.com/user-attachments/assets/2d4ce66d-007f-4015-b56d-1280e79bc050" />
  <br>
  <em>Figure 2: Annotation quality assessment and distribution of the 2,000-sample gold standard set</em>
</p>

---

## Data Schema (ACLKC-Dataset.csv)

The CSV file contains 19 columns. Here is a detailed breakdown using a single example (ID 446292) for illustration.

| Column Name | Description | Example (from ID 446292) |
| :--- | :--- | :--- |
| `id` | Unique identifier for the citation instance. | `446292` |
| `citing_paper_id` | ACL Anthology ID of the paper *making* the citation. | `P19-1228` |
| `citing_paper_title` | Title of the citing paper. | `Compound Probabilistic Context-Free Grammars for Grammar Induction` |
| `citing_paper_authors`| Authors of the citing paper. | `Yoon Kim; Chris Dyer; Alexander Rush` |
| `citing_paper_year` | Publication year of the citing paper. | `2019` |
| `citing_paper_abstract`| Abstract of the citing paper. | `We study a formalization of the grammar induction problem...` |
| `citation_section` | Section of the paper where the citation appeared. | `3. Compound PCFGs` |
| `citation_frequency` | Frequency count of the citation. | `4` |
| `cited_paper_title` | Title of the *cited* paper. | `Bayesian Inference for PCFGs via Markov chain Monte Carlo` |
| `cited_paper_authors` | Authors of the cited paper. | `Mark Johnson; Thomas Griffiths; Sharon Goldwater` |
| `cited_paper_year` | Publication year of the cited paper. | `2007.0` |
| `cited_paper_bib_id` | ACL Anthology ID of the paper *receiving* the citation. | `johnson-etal-2007-bayesian` |
| `cited_paper_abstract`| Abstract of the cited paper. | `This paper presents two Markov chain Monte Carlo (MCMC) algorithms...` |
| `citation_context` | The full text context (prev + current + next) of the citation. | `then the noun phrase is likely to be a movie. In contrast to the usual Bayesian treatment of PCFGs which places priors on global rule probabilities (Kurihara and Sato, 2006; CITATION Wang and Blunsom, 2013) , the compound PCFG assumes a prior on local, sentence-level rule probabilities. It is therefore closely related to the Bayesian grammars studied by Cohen et al. (2009)...` |
| `prev_sentence` | The sentence immediately preceding the citation sentence. | `then the noun phrase is likely to be a movie.` |
| `current_sentence` | The specific sentence containing the `CITATION` marker. | `In contrast to the usual Bayesian treatment of PCFGs which places priors on global rule probabilities (Kurihara and Sato, 2006; CITATION Wang and Blunsom, 2013) , the compound PCFG assumes a prior on local, sentence-level rule probabilities.` |
| `next_sentence` | The sentence immediately following the citation sentence. | `It is therefore closely related to the Bayesian grammars studied by Cohen et al. (2009) and Cohen and Smith (2009) , who also sample local rule probabilities from a logistic normal prior for training dependency models with valence (DMV) (Klein and Manning, 2004) .` |
| `period` | The pre-defined technological era of the citation. | `Period4_2017-2020` |
| `KC` | The Knowledge Contribution label (1=Method, 2=Resource, 3=Finding, 4=Background). | `1` |


---
## Quick Start

### Environment Setup
```bash
pip install torch transformers scikit-learn pandas numpy
```

### Train Main Model
```bash
python scibert+hybrid.py
```

### Run Comparison Experiments
```bash
# Traditional ML baselines
cd comparison_EXP/
python run_baselines.py

# LLM inference experiments
cd LLM_EXP/
python run_llm_inference.py
```

---

## Citation

If you use this dataset or code in your research, please cite our paper:
```bibtex
@article{quan2025knowledge,
    title = {Knowledge Contribution Classification in Citation Contexts: Framework and 45-Year NLP Analysis},
    author = {Zhibang Quan and Jin Mao and Gang Li and Zhentao Liang and Jinyu Wei and Jie Sun},
    journal = {...},
    year = {2025},
    publisher = {...}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
