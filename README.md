# Knowledge Contribution Classification in Citation Contexts: A 45-Year Empirical Study of NLP Literature

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://huggingface.co/transformers/"><img src="https://img.shields.io/badge/🤗_Transformers-4.0+-yellow.svg" alt="Transformers"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>


<p align="center">
  <a href="#dataset"><img src="https://img.shields.io/badge/📊_Dataset-Released-success.svg" alt="Dataset Released"></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/🚀_Quick_Start-Guide-blue.svg" alt="Quick Start"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/📝_Paper-Coming_Soon-lightgrey.svg" alt="Paper"></a>
</p>

---

This repository provides the model code, evaluation code, and manually annotated dataset for the paper *"Knowledge Contribution Classification in Citation Contexts: A 45-Year Empirical Study of NLP Literature"*.

---

## 📋 Overview

Understanding scientific knowledge flow and evolution is essential for assessing research impact and tracing disciplinary development. However, existing citation analysis methods focus on subjective citation intent, conflating heterogeneous knowledge types. Moreover, most studies analyze isolated contribution statements rather than actual citation contexts, yielding results disconnected from real knowledge dissemination scenarios.

To address these issues, we propose the **Knowledge Contribution Taxonomy (NLP-KCT)**. This framework identifies objective knowledge contribution types of cited works across four categories: **Method**, **Resource Tool**, **Empirical Findings**, and **Background**. By distinguishing core from non-core contributions, it reveals the hierarchical structure of knowledge dissemination.

---

## 🏷️ Knowledge Contribution Taxonomy (NLP-KCT)

Our classification framework (NLP-KCT) moves beyond subjective citation intent to focus on the objective, verifiable knowledge contribution of a cited paper. We define four categories, including three **Core Knowledge Contributions** and one **Non-core** contribution.

| Type | Category | Definition | Example |
| :---: | :--- | :--- | :--- |
| 🔧 | **Method** (Core) | Citation where the citing paper directly uses, implements, improves, or extends methods, techniques, algorithms, models, systems, or evaluation metrics from the cited paper. | "We compute COMET scores [CITATION] separately for each domain..." |
| 📦 | **Resource Tool** (Core) | Citation where the citing paper uses datasets, corpora, or explicitly labeled toolkits/toolboxes from the cited paper. | "We first extracted opinionated and objective texts from DeReKo corpus [CITATION]" |
| 📈 | **Empirical Findings** (Core) | Citation providing specific, verifiable empirical findings used for comparison, justifying decisions, or stating empirical facts. | "Table 2 shows that... our parser first outperforms the state-of-the-art segmentator of [CITATION]" |
| 📖 | **Research Background** (Non-core) | Citation providing necessary context by outlining development or describing research landscape. Descriptive rather than operational. | "Contemporary MT evaluation measures have evolved beyond simple lexical matching... [CITATION1], [CITATION2]" |

---

## 📁 Repository Structure

```
📦 NLP-KCT
├── 📂 data/
│   ├── 📄 train_split.csv              # Training set
│   ├── 📄 val_split.csv                # Validation set
│   └── 📄 test_split.csv               # Test set
├── 🐍 scibert+hybrid.py                # Main model: Dual-path hybrid fusion
├── 📂 comparison_EXP/
│   ├── 🐍 run_comparisons.py           # Baseline comparison experiments
│   ├── 📄 comparison_results_summary.csv
│   └── 📄 comparison_results.jsonl
├── 📂 LLM_EXP/
│   └── 🐍 llm_validation_standalone.py # LLM inference experiments
└── 📄 README.md
```

---

## 📊 Dataset

### 🗃️ Data Source: ACL Anthology (802,202 Instances)

We collected and preprocessed **802,202 citation instances** from the ACL Anthology, spanning 45 years of NLP research (1980–2024). Figure 1 shows the statistical features: exponential growth in publications (Fig. 1a), high concentration of citations in the *Introduction* (44.0%) and *Related Work* (34.3%) sections (Fig. 1b), and a normal distribution of context length (Mean: 434 chars) (Fig. 1c).

<p align="center">
  <img width="2199" height="792" alt="Figure 1: Data Statistics" src="https://github.com/user-attachments/assets/003b1f19-46c3-4bc7-960e-3f9b71f0fef1" />
  <br>
  <em>Figure 1: Detailed statistics of the ACL Anthology data (1980-2024)</em>
</p>

### 🏆 Released Dataset: Gold Standard (2,000 Instances)

From the above 802,202 instances, we performed **stratified sampling** to obtain **2,000 citation instances** for manual annotation. This gold standard dataset is released for model training, evaluation, and reproduction.

The annotation quality was rigorously assessed, achieving an **Average Cohen's Kappa of 0.770** and a **Fleiss' Kappa of 0.770** (Fig. 2a, 2b), indicating "excellent" inter-annotator agreement.

<p align="center">
  <img width="1958" height="588" alt="Figure 2: Annotation Quality" src="https://github.com/user-attachments/assets/2d4ce66d-007f-4015-b56d-1280e79bc050" />
  <br>
  <em>Figure 2: Annotation quality assessment and distribution of the 2,000-sample gold standard set</em>
</p>

---

## 📋 Data Schema

The CSV file contains 19 columns. Here is a detailed breakdown using a single example (ID 446292) for illustration.

<details>
<summary>📌 Click to expand full schema</summary>

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
| `citation_context` | The full text context (prev + current + next) of the citation. | `then the noun phrase is likely to be a movie. In contrast to the usual Bayesian treatment of PCFGs...` |
| `prev_sentence` | The sentence immediately preceding the citation sentence. | `then the noun phrase is likely to be a movie.` |
| `current_sentence` | The specific sentence containing the `CITATION` marker. | `In contrast to the usual Bayesian treatment of PCFGs which places priors on global rule probabilities...` |
| `next_sentence` | The sentence immediately following the citation sentence. | `It is therefore closely related to the Bayesian grammars studied by Cohen et al. (2009)...` |
| `period` | The pre-defined technological era of the citation. | `Period4_2017-2020` |
| `KC` | The Knowledge Contribution label (1=Method, 2=Resource, 3=Finding, 4=Background). | `1` |

</details>

---

## 🚀 Quick Start

### ⚙️ Environment Setup
```bash
pip install torch transformers scikit-learn pandas numpy requests tqdm
```

### 🏋️ Train Main Model
```bash
python scibert+hybrid.py
```

### 📊 Run Baseline Comparisons
```bash
cd comparison_EXP/
python run_comparisons.py
```

### 🤖 Run LLM Experiments

The `llm_validation_standalone.py` script supports both zero-shot and few-shot evaluation with any OpenAI-compatible API.

**📝 Zero-shot evaluation**
```bash
cd LLM_EXP/
python llm_validation_standalone.py \
    --dataset ../data/test_split.csv \
    --base-url https://api.openai.com/v1/chat/completions \
    --model-id gpt-4o \
    --api-key-env OPENAI_API_KEY \
    --mode zero \
    --output ./outputs
```

**📝 Few-shot evaluation**
```bash
python llm_validation_standalone.py \
    --dataset ../data/test_split.csv \
    --base-url https://api.openai.com/v1/chat/completions \
    --model-id gpt-4o \
    --api-key-env OPENAI_API_KEY \
    --mode few \
    --fewshot-file ./fewshot_examples.txt \
    --output ./outputs
```

**Key arguments:**

| Argument | Description |
| :--- | :--- |
| `--dataset` | Path to input CSV (requires columns: `id`, `KC`, `citation_section`, `citation_context`) |
| `--base-url` | API endpoint (OpenAI-compatible) |
| `--model-id` | Model identifier (e.g., `gptmini-5.1`, `claude-4.5-sonnet`, `gemini-2.5`) |
| `--api-key-env` | Environment variable name containing the API key |
| `--mode` | `zero` for zero-shot, `few` for few-shot |
| `--fewshot-file` | Path to few-shot examples file (required when `--mode few`) |
| `--limit` | Limit number of samples for quick testing |
| `--output` | Output directory for predictions and metrics |

---

## 🙏 Acknowledgments

We extend our sincere gratitude to all collaborators who contributed to this project. From data collection and preprocessing to the meticulous annotation of 2,000 citation instances, their dedication and effort were indispensable. The painstaking work of our annotators in carefully labeling each citation context forms the foundation upon which this research is built.

---

## 📝 Citation

If you use this dataset or code in your research, please cite our paper:

```bibtex
@article{quan2025knowledge,
    title = {Knowledge Contribution Classification in Citation Contexts: A 45-Year Empirical Study of NLP Literature},
    author = {Zhibang Quan and Jin Mao and Gang Li and Zhentao Liang and Ming Ma and Jinyu Wei and Jie Sun},
    journal = {...},
    year = {2025},
    publisher = {...}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <i>If you find this work helpful, please consider giving it a ⭐!</i>
</p>
