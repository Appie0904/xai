# Functions Reference

## `bios_bias_explainability_report.py`

### `set_seed(seed: int) -> None`
Sets random seeds for Python, NumPy, and PyTorch.

### `load_and_sample_dataset(config: ExperimentConfig) -> DatasetDict`
Loads `LabHC/bias_in_bios` and creates deterministic sampled train/dev/test splits.

### `tokenize_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer, max_length: int) -> DatasetDict`
Tokenizes biographies and adds `labels` for profession classification.

### `build_model_and_tokenizer(model_name: str, num_labels: int) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]`
Builds tokenizer and sequence-classification model.

### `train_model(model, tokenized_ds, output_dir: str, config: ExperimentConfig) -> Trainer`
Trains the classifier using Hugging Face `Trainer` and returns the trained trainer object.

### `predict_proba_texts(model, tokenizer, texts: List[str], max_length: int, batch_size: int, device: torch.device) -> np.ndarray`
Runs batched inference and returns class probabilities.

### `normalize_token(token: str) -> str`
Normalizes tokens for cross-method attribution comparisons.

### `build_gender_lexicon() -> set`
Returns a lexicon of gendered words used for bias-focused alignment metrics.

### `contains_gendered_token(text: str, gender_terms: set) -> bool`
Checks whether input text contains any gender lexicon term.

### `swap_gender_terms(text: str) -> str`
Builds a gender-swapped counterfactual text by replacing gendered words.

### `infer_swap_direction(text: str) -> str`
Classifies counterfactual direction as `male->female`, `female->male`, or `mixed`.

### `aggregate_wordpiece_scores(tokens: List[str], scores: np.ndarray) -> Dict[str, float]`
Aggregates subword-level scores to normalized word-level attribution scores.

### `attention_word_importance(model, tokenizer, text: str, max_length: int, device: torch.device) -> Dict[str, float]`
Extracts last-layer CLS-attention token attributions for a text sample.

### `lime_word_importance(explainer: LimeTextExplainer, predict_fn, text: str, pred_label: int, num_features: int, num_samples: int) -> Dict[str, float]`
Computes LIME token importance for a predicted class.

### `shap_word_importance(shap_explainer: shap.Explainer, text: str, pred_label: int, max_evals: int) -> Dict[str, float]`
Computes SHAP token importance for a predicted class.

### `attribution_alignment_metrics(importance: Dict[str, float], gender_terms: set, top_k: int) -> Dict[str, float]`
Computes human-alignment proxies:
- attribution mass on gender terms
- top-k hit rate
- first gender-token rank
- reciprocal rank (MRR)

### `save_tables_and_plots(model_metrics_df, profession_gap_df, counterfactual_df, explanation_records_df, output_dir: Path) -> None`
Exports report-ready tables and figures.

### `write_report_markdown(output_dir: Path, model_metrics_df, method_summary_df, counterfactual_df) -> None`
Writes markdown summary answering the explainability question with measured metrics.

### `main() -> None`
Runs the full experiment pipeline end-to-end.
