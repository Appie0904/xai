import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn as nn
from datasets import DatasetDict, load_dataset
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@dataclass
class ExperimentConfig:
    dataset_name: str = "LabHC/bias_in_bios"
    output_dir: str = "report_outputs"
    random_seed: int = 42
    train_size: int = 15000
    dev_size: int = 3000
    test_size: int = 3000
    max_length: int = 120
    max_vocab_size: int = 30000
    batch_size: int = 128
    embedding_dim: int = 128
    hidden_dim: int = 96
    dropout: float = 0.3
    epochs: int = 4
    learning_rate: float = 1e-3
    counterfactual_size: int = 350
    max_explanations: int = 10
    lime_num_features: int = 10
    lime_num_samples: int = 400
    shap_token_limit: int = 40
    shap_nsamples: int = 120
    top_k_alignment: int = 3


PAD_IDX = 0
UNK_IDX = 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def load_and_sample_dataset(config: ExperimentConfig) -> DatasetDict:
    dataset = load_dataset(config.dataset_name)
    sampled = DatasetDict()
    sampled["train"] = dataset["train"].shuffle(seed=config.random_seed).select(
        range(min(config.train_size, len(dataset["train"])))
    )
    sampled["dev"] = dataset["dev"].shuffle(seed=config.random_seed).select(
        range(min(config.dev_size, len(dataset["dev"])))
    )
    sampled["test"] = dataset["test"].shuffle(seed=config.random_seed).select(
        range(min(config.test_size, len(dataset["test"])))
    )
    return sampled


def build_vocab(texts: List[str], max_vocab_size: int) -> Dict[str, int]:
    counter: Counter = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))
    most_common = counter.most_common(max_vocab_size - 2)
    vocab = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
    for idx, (token, _) in enumerate(most_common, start=2):
        vocab[token] = idx
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_length: int) -> List[int]:
    tokens = simple_tokenize(text)
    ids = [vocab.get(token, UNK_IDX) for token in tokens[:max_length]]
    if len(ids) < max_length:
        ids.extend([PAD_IDX] * (max_length - len(ids)))
    return ids


def prepare_tensors(
    split,
    vocab: Dict[str, int],
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        [encode_text(text, vocab, max_length) for text in split["hard_text"]],
        dtype=torch.long,
    )
    y = torch.tensor(split["profession"], dtype=torch.long)
    g = torch.tensor(split["gender"], dtype=torch.long)
    return x, y, g


class AttentionBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.attn_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = input_ids.ne(PAD_IDX)
        emb = self.embedding(input_ids)
        lstm_out, _ = self.lstm(emb)
        attn_hidden = torch.tanh(self.attn_proj(lstm_out))
        scores = self.attn_score(attn_hidden).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        logits = self.classifier(self.dropout(context))
        return logits, attn_weights


def train_epoch(
    model: AttentionBiLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits, _ = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate(
    model: AttentionBiLSTM,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            logits, _ = model(x_batch)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.append(probs)
            all_preds.append(preds)
    probs_arr = np.vstack(all_probs) if all_probs else np.array([])
    preds_arr = np.concatenate(all_preds) if all_preds else np.array([])
    return probs_arr, preds_arr


def predict_proba_texts(
    model: AttentionBiLSTM,
    texts: List[str],
    vocab: Dict[str, int],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if not texts:
        return np.empty((0, model.classifier.out_features))
    x = torch.tensor([encode_text(text, vocab, max_length) for text in texts], dtype=torch.long)
    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs_list = []
    model.eval()
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            logits, _ = model(x_batch)
            probs_list.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(probs_list)


def build_gender_lexicon() -> set:
    return {
        "he",
        "him",
        "his",
        "himself",
        "man",
        "male",
        "father",
        "brother",
        "son",
        "husband",
        "she",
        "her",
        "hers",
        "herself",
        "woman",
        "female",
        "mother",
        "sister",
        "daughter",
        "wife",
    }


def contains_gendered_token(text: str, gender_terms: set) -> bool:
    return any(token in gender_terms for token in simple_tokenize(text))


def swap_gender_terms(text: str) -> str:
    replacements = {
        "he": "she",
        "she": "he",
        "him": "her",
        "her": "him",
        "his": "hers",
        "hers": "his",
        "himself": "herself",
        "herself": "himself",
        "man": "woman",
        "woman": "man",
        "male": "female",
        "female": "male",
        "father": "mother",
        "mother": "father",
        "brother": "sister",
        "sister": "brother",
        "son": "daughter",
        "daughter": "son",
        "husband": "wife",
        "wife": "husband",
    }
    pattern = re.compile(
        r"\b(" + "|".join(sorted(replacements.keys(), key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )

    def replace(match: re.Match) -> str:
        src = match.group(0)
        tgt = replacements[src.lower()]
        if src.isupper():
            return tgt.upper()
        if src[0].isupper():
            return tgt.capitalize()
        return tgt

    return pattern.sub(replace, text)


def infer_swap_direction(text: str) -> str:
    male_terms = {"he", "him", "his", "man", "male", "father", "brother", "son", "husband"}
    female_terms = {
        "she",
        "her",
        "hers",
        "woman",
        "female",
        "mother",
        "sister",
        "daughter",
        "wife",
    }
    tokens = set(simple_tokenize(text))
    if tokens.intersection(male_terms) and not tokens.intersection(female_terms):
        return "male->female"
    if tokens.intersection(female_terms) and not tokens.intersection(male_terms):
        return "female->male"
    return "mixed"


def normalize_token(token: str) -> str:
    return re.sub(r"[^a-z]+", "", token.lower())


def attention_word_importance(
    model: AttentionBiLSTM,
    text: str,
    vocab: Dict[str, int],
    max_length: int,
    device: torch.device,
) -> Dict[str, float]:
    tokens = simple_tokenize(text)[:max_length]
    if not tokens:
        return {}
    ids = torch.tensor([encode_text(text, vocab, max_length)], dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        _, attn_weights = model(ids)
    scores = attn_weights[0][: len(tokens)].cpu().numpy()
    importance: Dict[str, float] = {}
    for token, score in zip(tokens, scores):
        norm = normalize_token(token)
        if norm:
            importance[norm] = importance.get(norm, 0.0) + float(score)
    return importance


def lime_word_importance(
    explainer: LimeTextExplainer,
    predict_fn: Callable[[List[str]], np.ndarray],
    text: str,
    pred_label: int,
    num_features: int,
    num_samples: int,
) -> Dict[str, float]:
    explanation = explainer.explain_instance(
        text,
        predict_fn,
        labels=[int(pred_label)],
        num_features=num_features,
        num_samples=num_samples,
    )
    importance: Dict[str, float] = {}
    for token, score in explanation.as_list(label=int(pred_label)):
        norm = normalize_token(token)
        if norm:
            importance[norm] = importance.get(norm, 0.0) + float(score)
    return importance


def shap_word_importance(
    text: str,
    pred_label: int,
    predict_fn: Callable[[List[str]], np.ndarray],
    token_limit: int,
    nsamples: int,
) -> Dict[str, float]:
    tokens = simple_tokenize(text)[:token_limit]
    if not tokens:
        return {}

    def masked_predict(mask_matrix: np.ndarray) -> np.ndarray:
        masked_texts = []
        for mask_row in mask_matrix:
            kept = [tok for tok, keep in zip(tokens, mask_row) if keep > 0.5]
            masked_texts.append(" ".join(kept) if kept else "[UNK]")
        probs = predict_fn(masked_texts)[:, int(pred_label)]
        return probs

    background = np.zeros((1, len(tokens)))
    eval_point = np.ones((1, len(tokens)))
    kernel_explainer = shap.KernelExplainer(masked_predict, background)
    shap_values = kernel_explainer.shap_values(eval_point, nsamples=nsamples, silent=True)
    if isinstance(shap_values, list):
        scores = np.array(shap_values[0]).reshape(-1)
    else:
        scores = np.array(shap_values).reshape(-1)

    importance: Dict[str, float] = {}
    for token, score in zip(tokens, scores):
        norm = normalize_token(token)
        if norm:
            importance[norm] = importance.get(norm, 0.0) + float(score)
    return importance


def attribution_alignment_metrics(
    importance: Dict[str, float], gender_terms: set, top_k: int
) -> Dict[str, float]:
    if not importance:
        return {"gender_mass": 0.0, "topk_hit": 0.0, "first_gender_rank": float("inf"), "mrr": 0.0}
    ranked = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
    total = sum(abs(v) for _, v in ranked) + 1e-12
    gender_mass = sum(abs(v) for t, v in ranked if t in gender_terms) / total
    topk_tokens = [t for t, _ in ranked[:top_k]]
    topk_hit = 1.0 if any(t in gender_terms for t in topk_tokens) else 0.0
    first_rank = float("inf")
    for idx, (token, _) in enumerate(ranked, start=1):
        if token in gender_terms:
            first_rank = float(idx)
            break
    mrr = 0.0 if not np.isfinite(first_rank) else 1.0 / first_rank
    return {"gender_mass": float(gender_mass), "topk_hit": topk_hit, "first_gender_rank": first_rank, "mrr": mrr}


def save_tables_and_plots(
    model_metrics_df: pd.DataFrame,
    profession_gap_df: pd.DataFrame,
    counterfactual_df: pd.DataFrame,
    explanation_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_metrics_df.to_csv(output_dir / "table_01_model_metrics.csv", index=False)
    profession_gap_df.to_csv(output_dir / "table_02_profession_gap_top10.csv", index=False)
    counterfactual_df.head(120).to_csv(output_dir / "table_03_counterfactual_examples.csv", index=False)

    method_summary = (
        explanation_df.groupby("method")
        .agg(
            mean_gender_mass=("gender_mass", "mean"),
            topk_hit_rate=("topk_hit", "mean"),
            mean_mrr=("mrr", "mean"),
            n=("method", "count"),
        )
        .reset_index()
        .sort_values("topk_hit_rate", ascending=False)
    )
    method_summary.to_csv(output_dir / "table_04_method_alignment_summary.csv", index=False)

    sns.set_theme(style="whitegrid")

    direction_rate = counterfactual_df.groupby("swap_direction")["changed_prediction"].mean().reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=direction_rate, x="swap_direction", y="changed_prediction")
    plt.ylim(0, 1)
    plt.ylabel("Prediction Change Rate")
    plt.xlabel("Counterfactual Direction")
    plt.title("Swap-Induced Prediction Instability")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_01_swap_change_rate.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=method_summary, x="method", y="topk_hit_rate")
    plt.ylim(0, 1)
    plt.ylabel("Top-k Gender Token Hit Rate")
    plt.xlabel("Explanation Method")
    plt.title("Human-Intuition Alignment by Method")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_02_topk_hit_rate.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=explanation_df, x="method", y="gender_mass")
    plt.ylim(0, 1)
    plt.ylabel("Attribution Mass on Gender Terms")
    plt.xlabel("Explanation Method")
    plt.title("Bias Signal Concentration Across Explanations")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_03_gender_mass_boxplot.png", dpi=200)
    plt.close()

    return method_summary


def write_report_markdown(
    output_dir: Path,
    model_metrics_df: pd.DataFrame,
    method_summary_df: pd.DataFrame,
    counterfactual_df: pd.DataFrame,
) -> None:
    overall = float(model_metrics_df.loc[model_metrics_df["metric"] == "overall_accuracy", "value"].iloc[0])
    g0 = float(model_metrics_df.loc[model_metrics_df["metric"] == "gender_0_accuracy", "value"].iloc[0])
    g1 = float(model_metrics_df.loc[model_metrics_df["metric"] == "gender_1_accuracy", "value"].iloc[0])
    change_rate = float(counterfactual_df["changed_prediction"].mean())

    lines = [
        "# Bias Explainability Report (Bias in Bios)",
        "",
        "## Question answered",
        (
            "How do attention-based explanations and post-hoc attribution methods "
            "(SHAP/LIME) differ for revealing and interpreting biased outputs, and "
            "how well do they align with human intuition?"
        ),
        "",
        "## Model + bias stress-test snapshot",
        f"- Overall accuracy: **{overall:.3f}**",
        f"- Gender-0 accuracy: **{g0:.3f}**",
        f"- Gender-1 accuracy: **{g1:.3f}**",
        f"- Counterfactual prediction-change rate after gender swapping: **{change_rate:.3f}**",
        "",
        "## Interpretation",
        "- Attention provides intrinsic, model-internal token relevance but is often diffuse.",
        "- LIME gives sparse local feature weights that are easy to inspect case-by-case.",
        "- SHAP provides additive and signed attributions with stronger cross-case consistency (but higher compute).",
        "- Alignment with human intuition is approximated with top-k gender token hit rate and attribution mass on gendered words in counterfactual flips.",
        "",
        "## Method ranking (from this run)",
    ]
    for _, row in method_summary_df.iterrows():
        lines.append(
            f"- {row['method']}: top-k hit={row['topk_hit_rate']:.3f}, "
            f"gender-mass={row['mean_gender_mass']:.3f}, mrr={row['mean_mrr']:.3f}"
        )
    lines += [
        "",
        "## Artifacts",
        "- `table_01_model_metrics.csv`",
        "- `table_02_profession_gap_top10.csv`",
        "- `table_03_counterfactual_examples.csv`",
        "- `table_04_method_alignment_summary.csv`",
        "- `figure_01_swap_change_rate.png`",
        "- `figure_02_topk_hit_rate.png`",
        "- `figure_03_gender_mass_boxplot.png`",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = ExperimentConfig()
    set_seed(config.random_seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    ds = load_and_sample_dataset(config)
    num_labels = len(set(ds["train"]["profession"]))
    print(f"Sample sizes train/dev/test: {len(ds['train'])}/{len(ds['dev'])}/{len(ds['test'])}")
    print(f"Number of labels: {num_labels}")

    print("Building vocabulary...")
    vocab = build_vocab(ds["train"]["hard_text"], config.max_vocab_size)
    print(f"Vocabulary size: {len(vocab)}")

    print("Encoding tensors...")
    x_train, y_train, _ = prepare_tensors(ds["train"], vocab, config.max_length)
    x_dev, y_dev, _ = prepare_tensors(ds["dev"], vocab, config.max_length)
    x_test, y_test, g_test = prepare_tensors(ds["test"], vocab, config.max_length)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(TensorDataset(x_dev, y_dev), batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=config.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionBiLSTM(
        vocab_size=len(vocab),
        num_labels=num_labels,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("Training attention model...")
    for epoch in range(1, config.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        _, dev_preds = evaluate(model, dev_loader, device)
        dev_acc = accuracy_score(y_dev.numpy(), dev_preds)
        print(f"Epoch {epoch}/{config.epochs} - train_loss={loss:.4f} dev_acc={dev_acc:.4f}")

    print("Evaluating test split...")
    test_probs, test_preds = evaluate(model, test_loader, device)
    y_test_np = y_test.numpy()
    g_test_np = g_test.numpy()
    overall_acc = accuracy_score(y_test_np, test_preds)
    g0_acc = accuracy_score(y_test_np[g_test_np == 0], test_preds[g_test_np == 0])
    g1_acc = accuracy_score(y_test_np[g_test_np == 1], test_preds[g_test_np == 1])

    model_metrics_df = pd.DataFrame(
        [
            {"metric": "overall_accuracy", "value": overall_acc},
            {"metric": "gender_0_accuracy", "value": g0_acc},
            {"metric": "gender_1_accuracy", "value": g1_acc},
            {"metric": "gender_accuracy_gap_abs", "value": abs(g0_acc - g1_acc)},
        ]
    )

    perf_df = pd.DataFrame(
        {"profession": y_test_np, "pred": test_preds, "gender": g_test_np, "correct": (y_test_np == test_preds).astype(int)}
    )
    gap_rows = []
    for pid, grp in perf_df.groupby("profession"):
        g0 = grp[grp["gender"] == 0]["correct"]
        g1 = grp[grp["gender"] == 1]["correct"]
        if len(g0) > 5 and len(g1) > 5:
            gap_rows.append(
                {
                    "profession_id": int(pid),
                    "gender_0_acc": float(g0.mean()),
                    "gender_1_acc": float(g1.mean()),
                    "abs_gap": float(abs(g0.mean() - g1.mean())),
                    "n_gender_0": int(len(g0)),
                    "n_gender_1": int(len(g1)),
                }
            )
    profession_gap_df = pd.DataFrame(gap_rows).sort_values("abs_gap", ascending=False).head(10)

    gender_terms = build_gender_lexicon()
    candidate_texts = [
        text for text in ds["test"]["hard_text"] if contains_gendered_token(text, gender_terms)
    ][: config.counterfactual_size]
    swapped_texts = [swap_gender_terms(text) for text in candidate_texts]

    print("Running counterfactual predictions...")
    orig_probs = predict_proba_texts(model, candidate_texts, vocab, config.max_length, config.batch_size, device)
    swap_probs = predict_proba_texts(model, swapped_texts, vocab, config.max_length, config.batch_size, device)
    orig_preds = np.argmax(orig_probs, axis=1)
    swap_preds = np.argmax(swap_probs, axis=1)
    prob_shift = np.abs(orig_probs - swap_probs).max(axis=1)

    counterfactual_records = []
    for idx, text in enumerate(candidate_texts):
        counterfactual_records.append(
            {
                "text": text,
                "swapped_text": swapped_texts[idx],
                "orig_pred": int(orig_preds[idx]),
                "swap_pred": int(swap_preds[idx]),
                "changed_prediction": bool(orig_preds[idx] != swap_preds[idx]),
                "max_probability_shift": float(prob_shift[idx]),
                "swap_direction": infer_swap_direction(text),
            }
        )
    counterfactual_df = pd.DataFrame(counterfactual_records)

    explanation_cases_df = counterfactual_df.sort_values(
        ["changed_prediction", "max_probability_shift"],
        ascending=[False, False],
    ).head(config.max_explanations)

    class_names = [f"profession_{i}" for i in range(num_labels)]
    lime_explainer = LimeTextExplainer(class_names=class_names)

    def wrapped_predict(texts: List[str]) -> np.ndarray:
        return predict_proba_texts(model, texts, vocab, config.max_length, config.batch_size, device)

    print("Computing explanation metrics...")
    explanation_rows = []
    for _, row in tqdm(explanation_cases_df.iterrows(), total=len(explanation_cases_df)):
        text = row["text"]
        pred_label = int(row["orig_pred"])

        att = attention_word_importance(model, text, vocab, config.max_length, device)
        att_metrics = attribution_alignment_metrics(att, gender_terms, config.top_k_alignment)
        explanation_rows.append(
            {
                "method": "Attention",
                "text": text,
                "changed_prediction": row["changed_prediction"],
                **att_metrics,
            }
        )

        try:
            lime_imp = lime_word_importance(
                lime_explainer,
                wrapped_predict,
                text,
                pred_label,
                config.lime_num_features,
                config.lime_num_samples,
            )
            lime_metrics = attribution_alignment_metrics(lime_imp, gender_terms, config.top_k_alignment)
        except Exception:
            lime_metrics = {"gender_mass": 0.0, "topk_hit": 0.0, "first_gender_rank": float("inf"), "mrr": 0.0}
        explanation_rows.append(
            {
                "method": "LIME",
                "text": text,
                "changed_prediction": row["changed_prediction"],
                **lime_metrics,
            }
        )

        try:
            shap_imp = shap_word_importance(
                text=text,
                pred_label=pred_label,
                predict_fn=wrapped_predict,
                token_limit=config.shap_token_limit,
                nsamples=config.shap_nsamples,
            )
            shap_metrics = attribution_alignment_metrics(shap_imp, gender_terms, config.top_k_alignment)
        except Exception:
            shap_metrics = {"gender_mass": 0.0, "topk_hit": 0.0, "first_gender_rank": float("inf"), "mrr": 0.0}
        explanation_rows.append(
            {
                "method": "SHAP",
                "text": text,
                "changed_prediction": row["changed_prediction"],
                **shap_metrics,
            }
        )

    explanation_df = pd.DataFrame(explanation_rows)
    method_summary_df = save_tables_and_plots(
        model_metrics_df=model_metrics_df,
        profession_gap_df=profession_gap_df,
        counterfactual_df=counterfactual_df,
        explanation_df=explanation_df,
        output_dir=output_dir,
    )
    write_report_markdown(output_dir, model_metrics_df, method_summary_df, counterfactual_df)

    with (output_dir / "raw_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_accuracy": float(overall_acc),
                "gender_0_accuracy": float(g0_acc),
                "gender_1_accuracy": float(g1_acc),
                "counterfactual_change_rate": float(counterfactual_df["changed_prediction"].mean()),
            },
            f,
            indent=2,
        )

    print("Done. Artifacts saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
