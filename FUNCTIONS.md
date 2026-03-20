# Functions Reference

## `bios_bias_explainability_report.py`

### `set_seed(seed: int) -> None`
Sets random seeds for Python, NumPy, and PyTorch.

### `simple_tokenize(text: str) -> List[str]`
Tokenizes biography text with a lightweight regex tokenizer.

### `load_and_sample_dataset(config: ExperimentConfig) -> DatasetDict`
Loads `LabHC/bias_in_bios` and creates deterministic sampled train/dev/test splits.

### `build_vocab(texts: List[str], max_vocab_size: int) -> Dict[str, int]`
Builds a capped vocabulary from training texts.

### `encode_text(text: str, vocab: Dict[str, int], max_length: int) -> List[int]`
Encodes text to fixed-length token IDs with padding/truncation.

### `prepare_tensors(split, vocab: Dict[str, int], max_length: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
Converts one dataset split into tensors (`input_ids`, `profession`, `gender`).

### `AttentionBiLSTM.forward(input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`
Runs the attention-based BiLSTM and returns logits + token attention weights.

### `train_epoch(model, loader, optimizer, criterion, device) -> float`
Runs one training epoch and returns mean training loss.

### `evaluate(model, loader, device) -> Tuple[np.ndarray, np.ndarray]`
Runs inference on a dataloader and returns probabilities and class predictions.

### `predict_proba_texts(model, texts: List[str], vocab: Dict[str, int], max_length: int, batch_size: int, device: torch.device) -> np.ndarray`
Batched probability inference for raw text inputs.

### `build_gender_lexicon() -> set`
Returns the gendered lexicon used for counterfactuals and alignment metrics.

### `contains_gendered_token(text: str, gender_terms: set) -> bool`
Checks whether text includes any gendered term.

### `swap_gender_terms(text: str) -> str`
Creates a gender-swapped counterfactual text.

### `infer_swap_direction(text: str) -> str`
Infers direction category (`male->female`, `female->male`, `mixed`) for a sample.

### `normalize_token(token: str) -> str`
Normalizes tokens to lowercase alphabetic forms for cross-method aggregation.

### `attention_word_importance(model, text: str, vocab: Dict[str, int], max_length: int, device: torch.device) -> Dict[str, float]`
Extracts model-internal attention weights at token level.

### `lime_word_importance(explainer: LimeTextExplainer, predict_fn, text: str, pred_label: int, num_features: int, num_samples: int) -> Dict[str, float]`
Computes LIME token attributions for the predicted class.

### `shap_word_importance(text: str, pred_label: int, predict_fn, token_limit: int, nsamples: int) -> Dict[str, float]`
Computes local token attributions using KernelSHAP via token masking.

### `attribution_alignment_metrics(importance: Dict[str, float], gender_terms: set, top_k: int) -> Dict[str, float]`
Computes human-intuition alignment proxies:
- gender attribution mass
- top-k gender-token hit rate
- first gender-token rank
- reciprocal rank (MRR)

### `save_tables_and_plots(model_metrics_df, profession_gap_df, counterfactual_df, explanation_df, output_dir: Path) -> pd.DataFrame`
Exports report-ready CSV tables and PNG figures; returns method summary table.

### `write_report_markdown(output_dir: Path, model_metrics_df, method_summary_df, counterfactual_df) -> None`
Writes a concise markdown report with findings and artifact links.

### `main() -> None`
Runs the full pipeline end-to-end (training, bias checks, explanations, exports).
