import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
from datasets import DatasetDict, load_dataset
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

matplotlib.use("Agg")


@dataclass
class ExperimentConfig:
    dataset_name: str = "LabHC/bias_in_bios"
    model_name: str = "prajjwal1/bert-tiny"
    output_dir: str = "report_outputs"
    random_seed: int = 42
    train_size: int = 12000
    dev_size: int = 3000
    test_size: int = 4000
    counterfactual_size: int = 400
    max_length: int = 192
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    learning_rate: float = 2e-5
    max_explanations: int = 12
    lime_num_features: int = 10
    lime_num_samples: int = 700
    shap_max_evals: int = 300
    top_k_alignment: int = 3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def tokenize_dataset(
    dataset: DatasetDict, tokenizer: AutoTokenizer, max_length: int
) -> DatasetDict:
    def preprocess(batch: Dict[str, List]) -> Dict[str, List]:
        tokenized = tokenizer(
            batch["hard_text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        tokenized["labels"] = batch["profession"]
        return tokenized

    tokenized_ds = dataset.map(preprocess, batched=True)
    tokenized_ds = tokenized_ds.remove_columns(["hard_text", "profession", "gender"])
    tokenized_ds.set_format(type="torch")
    return tokenized_ds


def build_model_and_tokenizer(
    model_name: str, num_labels: int
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    return tokenizer, model


def train_model(
    model: AutoModelForSequenceClassification,
    tokenized_ds: DatasetDict,
    output_dir: str,
    config: ExperimentConfig,
) -> Trainer:
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "training_artifacts"),
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        report_to=[],
        seed=config.random_seed,
    )

    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["dev"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer


def predict_proba_texts(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probs_list = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encodings = {k: v.to(device) for k, v in encodings.items()}
            outputs = model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            probs_list.append(probs)
    return np.vstack(probs_list) if probs_list else np.empty((0, model.num_labels))


def normalize_token(token: str) -> str:
    token = token.replace("##", "").strip().lower()
    token = re.sub(r"[^a-z]+", "", token)
    return token


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
        "husband",
    }


def contains_gendered_token(text: str, gender_terms: set) -> bool:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return any(token in gender_terms for token in tokens)


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
        repl = replacements[src.lower()]
        if src.isupper():
            return repl.upper()
        if src[0].isupper():
            return repl.capitalize()
        return repl

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
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    if tokens.intersection(male_terms) and not tokens.intersection(female_terms):
        return "male->female"
    if tokens.intersection(female_terms) and not tokens.intersection(male_terms):
        return "female->male"
    return "mixed"


def aggregate_wordpiece_scores(tokens: List[str], scores: np.ndarray) -> Dict[str, float]:
    word_scores: Dict[str, float] = {}
    current_word = ""
    current_score = 0.0

    def flush() -> None:
        nonlocal current_word, current_score
        if current_word:
            norm = normalize_token(current_word)
            if norm:
                word_scores[norm] = word_scores.get(norm, 0.0) + float(current_score)
        current_word = ""
        current_score = 0.0

    for token, score in zip(tokens, scores):
        if token in {"[CLS]", "[SEP]", "[PAD]"}:
            flush()
            continue
        if token.startswith("##"):
            current_word += token[2:]
            current_score += float(score)
        else:
            flush()
            current_word = token
            current_score = float(score)
    flush()
    return word_scores


def attention_word_importance(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    max_length: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc, output_attentions=True)
    attentions = outputs.attentions[-1][0]  # heads x seq x seq
    cls_attention = attentions[:, 0, :].mean(dim=0).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].cpu().numpy())
    return aggregate_wordpiece_scores(tokens, cls_attention)


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
    importance = {}
    for token, score in explanation.as_list(label=int(pred_label)):
        norm = normalize_token(token)
        if norm:
            importance[norm] = importance.get(norm, 0.0) + float(score)
    return importance


def shap_word_importance(
    shap_explainer: shap.Explainer,
    text: str,
    pred_label: int,
    max_evals: int,
) -> Dict[str, float]:
    shap_values = shap_explainer([text], max_evals=max_evals)
    tokens_raw = shap_values.data[0]
    if isinstance(tokens_raw, str):
        tokens = re.findall(r"\b\w+\b", tokens_raw.lower())
    else:
        tokens = [str(t) for t in tokens_raw]

    values = shap_values.values
    if values.ndim == 3:
        token_scores = values[0, :, int(pred_label)]
    elif values.ndim == 2:
        token_scores = values[0, :]
    else:
        token_scores = np.array([])

    importance: Dict[str, float] = {}
    for token, score in zip(tokens, token_scores):
        norm = normalize_token(token)
        if norm:
            importance[norm] = importance.get(norm, 0.0) + float(score)
    return importance


def attribution_alignment_metrics(
    importance: Dict[str, float], gender_terms: set, top_k: int
) -> Dict[str, float]:
    if not importance:
        return {
            "gender_mass": 0.0,
            "topk_hit": 0.0,
            "first_gender_rank": float("inf"),
            "mrr": 0.0,
        }
    sorted_items = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
    total_mass = sum(abs(v) for _, v in sorted_items) + 1e-12
    gender_mass = sum(abs(v) for t, v in sorted_items if t in gender_terms) / total_mass
    topk_tokens = [token for token, _ in sorted_items[:top_k]]
    topk_hit = 1.0 if any(token in gender_terms for token in topk_tokens) else 0.0

    first_gender_rank = float("inf")
    for idx, (token, _) in enumerate(sorted_items, start=1):
        if token in gender_terms:
            first_gender_rank = float(idx)
            break
    mrr = 0.0 if not np.isfinite(first_gender_rank) else 1.0 / first_gender_rank
    return {
        "gender_mass": float(gender_mass),
        "topk_hit": topk_hit,
        "first_gender_rank": first_gender_rank,
        "mrr": mrr,
    }


def save_tables_and_plots(
    model_metrics_df: pd.DataFrame,
    profession_gap_df: pd.DataFrame,
    counterfactual_df: pd.DataFrame,
    explanation_records_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_metrics_df.to_csv(output_dir / "table_01_model_metrics.csv", index=False)
    profession_gap_df.to_csv(output_dir / "table_02_profession_gap_top10.csv", index=False)
    counterfactual_df.to_csv(output_dir / "table_03_counterfactual_examples.csv", index=False)

    method_summary = (
        explanation_records_df.groupby("method")
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

    direction_rate = (
        counterfactual_df.groupby("swap_direction")["changed_prediction"].mean().reset_index()
    )
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
    plt.title("Human-Intuition Alignment (Top-k Hit)")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_02_topk_hit_rate.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=explanation_records_df, x="method", y="gender_mass")
    plt.ylim(0, 1)
    plt.ylabel("Attribution Mass on Gender Terms")
    plt.xlabel("Explanation Method")
    plt.title("Bias Signal Concentration Across Methods")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_03_gender_mass_boxplot.png", dpi=200)
    plt.close()


def write_report_markdown(
    output_dir: Path,
    model_metrics_df: pd.DataFrame,
    method_summary_df: pd.DataFrame,
    counterfactual_df: pd.DataFrame,
) -> None:
    overall_acc = float(
        model_metrics_df.loc[model_metrics_df["metric"] == "overall_accuracy", "value"].iloc[0]
    )
    male_acc = float(
        model_metrics_df.loc[model_metrics_df["metric"] == "gender_0_accuracy", "value"].iloc[0]
    )
    female_acc = float(
        model_metrics_df.loc[model_metrics_df["metric"] == "gender_1_accuracy", "value"].iloc[0]
    )
    change_rate = float(counterfactual_df["changed_prediction"].mean())

    best_hit_method = method_summary_df.iloc[0]["method"]
    best_hit_value = float(method_summary_df.iloc[0]["topk_hit_rate"])
    low_mass_method = method_summary_df.sort_values("mean_gender_mass").iloc[0]["method"]
    low_mass_value = float(method_summary_df.sort_values("mean_gender_mass").iloc[0]["mean_gender_mass"])

    lines = [
        "# Bias Explainability Report (Bias in Bios)",
        "",
        "## Core question",
        (
            "How do attention-based explanations and post-hoc attribution methods "
            "(SHAP, LIME) differ in revealing and interpreting biased outputs, and how "
            "well do they align with human intuition?"
        ),
        "",
        "## Model and bias stress test",
        f"- Overall profession-classification accuracy: **{overall_acc:.3f}**",
        f"- Gender-0 accuracy: **{male_acc:.3f}**",
        f"- Gender-1 accuracy: **{female_acc:.3f}**",
        f"- Prediction change rate after gender swapping: **{change_rate:.3f}**",
        "",
        "## Explainability findings",
        (
            f"- Highest human-alignment (top-k gender-token hit rate): "
            f"**{best_hit_method} ({best_hit_value:.3f})**"
        ),
        (
            f"- Lowest attribution mass on gender tokens: "
            f"**{low_mass_method} ({low_mass_value:.3f})**"
        ),
        "",
        "## Interpretation",
        (
            "Attention is model-internal and often diffuse: it can reveal that gender "
            "tokens are being attended to, but it is less localized and can be harder "
            "to interpret causally."
        ),
        (
            "LIME is sparse and local: it typically highlights a short list of words "
            "driving a single decision, which can be intuitive for case-level bias review."
        ),
        (
            "SHAP gives additive token-level attributions with sign and magnitude, often "
            "providing stronger consistency across cases at higher computational cost."
        ),
        (
            "Alignment with human intuition can be approximated by whether methods rank "
            "gendered terms among top explanatory features in counterfactual flips."
        ),
        "",
        "## Report artifacts",
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

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and sampling dataset...")
    dataset = load_and_sample_dataset(config)

    num_labels = len(set(dataset["train"]["profession"]))
    print(f"Number of labels: {num_labels}")

    tokenizer, model = build_model_and_tokenizer(config.model_name, num_labels)
    tokenized_ds = tokenize_dataset(dataset, tokenizer, config.max_length)

    print("Training classifier...")
    trainer = train_model(model, tokenized_ds, config.output_dir, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Running test predictions...")
    test_texts = dataset["test"]["hard_text"]
    test_labels = np.array(dataset["test"]["profession"])
    test_gender = np.array(dataset["test"]["gender"])
    test_probs = predict_proba_texts(
        model=model,
        tokenizer=tokenizer,
        texts=test_texts,
        max_length=config.max_length,
        batch_size=config.per_device_eval_batch_size,
        device=device,
    )
    test_preds = np.argmax(test_probs, axis=1)

    overall_acc = accuracy_score(test_labels, test_preds)
    gender0_acc = accuracy_score(test_labels[test_gender == 0], test_preds[test_gender == 0])
    gender1_acc = accuracy_score(test_labels[test_gender == 1], test_preds[test_gender == 1])

    model_metrics_df = pd.DataFrame(
        [
            {"metric": "overall_accuracy", "value": overall_acc},
            {"metric": "gender_0_accuracy", "value": gender0_acc},
            {"metric": "gender_1_accuracy", "value": gender1_acc},
            {"metric": "gender_accuracy_gap_abs", "value": abs(gender0_acc - gender1_acc)},
        ]
    )

    performance_df = pd.DataFrame(
        {
            "profession": test_labels,
            "pred": test_preds,
            "gender": test_gender,
            "correct": (test_labels == test_preds).astype(int),
        }
    )
    profession_gap_rows = []
    for profession_id, group in performance_df.groupby("profession"):
        g0 = group[group["gender"] == 0]["correct"]
        g1 = group[group["gender"] == 1]["correct"]
        if len(g0) > 5 and len(g1) > 5:
            acc0 = g0.mean()
            acc1 = g1.mean()
            profession_gap_rows.append(
                {
                    "profession_id": int(profession_id),
                    "gender_0_acc": acc0,
                    "gender_1_acc": acc1,
                    "abs_gap": abs(acc0 - acc1),
                    "n_gender_0": len(g0),
                    "n_gender_1": len(g1),
                }
            )
    profession_gap_df = (
        pd.DataFrame(profession_gap_rows).sort_values("abs_gap", ascending=False).head(10)
    )

    gender_terms = build_gender_lexicon()
    candidate_texts = [
        text for text in test_texts if contains_gendered_token(text=text, gender_terms=gender_terms)
    ][: config.counterfactual_size]
    swapped_texts = [swap_gender_terms(text) for text in candidate_texts]

    print("Scoring counterfactual pairs...")
    orig_probs = predict_proba_texts(
        model=model,
        tokenizer=tokenizer,
        texts=candidate_texts,
        max_length=config.max_length,
        batch_size=config.per_device_eval_batch_size,
        device=device,
    )
    swap_probs = predict_proba_texts(
        model=model,
        tokenizer=tokenizer,
        texts=swapped_texts,
        max_length=config.max_length,
        batch_size=config.per_device_eval_batch_size,
        device=device,
    )
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
        ["changed_prediction", "max_probability_shift"], ascending=[False, False]
    ).head(config.max_explanations)

    class_names = [f"profession_{i}" for i in range(num_labels)]
    lime_explainer = LimeTextExplainer(class_names=class_names)

    def wrapped_predict(texts: List[str]) -> np.ndarray:
        return predict_proba_texts(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            max_length=config.max_length,
            batch_size=config.per_device_eval_batch_size,
            device=device,
        )

    print("Preparing SHAP explainer...")
    shap_explainer = shap.Explainer(wrapped_predict, tokenizer)

    print("Computing explanation alignment metrics...")
    explanation_rows = []
    for _, row in tqdm(explanation_cases_df.iterrows(), total=len(explanation_cases_df)):
        text = row["text"]
        pred_label = int(row["orig_pred"])

        att_imp = attention_word_importance(
            model=model,
            tokenizer=tokenizer,
            text=text,
            max_length=config.max_length,
            device=device,
        )
        att_metrics = attribution_alignment_metrics(att_imp, gender_terms, config.top_k_alignment)
        explanation_rows.append(
            {
                "method": "Attention",
                "text": text,
                **att_metrics,
                "changed_prediction": row["changed_prediction"],
                "swap_direction": row["swap_direction"],
            }
        )

        lime_imp = lime_word_importance(
            explainer=lime_explainer,
            predict_fn=wrapped_predict,
            text=text,
            pred_label=pred_label,
            num_features=config.lime_num_features,
            num_samples=config.lime_num_samples,
        )
        lime_metrics = attribution_alignment_metrics(lime_imp, gender_terms, config.top_k_alignment)
        explanation_rows.append(
            {
                "method": "LIME",
                "text": text,
                **lime_metrics,
                "changed_prediction": row["changed_prediction"],
                "swap_direction": row["swap_direction"],
            }
        )

        shap_imp = shap_word_importance(
            shap_explainer=shap_explainer,
            text=text,
            pred_label=pred_label,
            max_evals=config.shap_max_evals,
        )
        shap_metrics = attribution_alignment_metrics(shap_imp, gender_terms, config.top_k_alignment)
        explanation_rows.append(
            {
                "method": "SHAP",
                "text": text,
                **shap_metrics,
                "changed_prediction": row["changed_prediction"],
                "swap_direction": row["swap_direction"],
            }
        )

    explanation_records_df = pd.DataFrame(explanation_rows)
    method_summary_df = (
        explanation_records_df.groupby("method")
        .agg(
            mean_gender_mass=("gender_mass", "mean"),
            topk_hit_rate=("topk_hit", "mean"),
            mean_mrr=("mrr", "mean"),
            n=("method", "count"),
        )
        .reset_index()
        .sort_values("topk_hit_rate", ascending=False)
    )

    save_tables_and_plots(
        model_metrics_df=model_metrics_df,
        profession_gap_df=profession_gap_df,
        counterfactual_df=counterfactual_df,
        explanation_records_df=explanation_records_df,
        output_dir=out_dir,
    )
    write_report_markdown(
        output_dir=out_dir,
        model_metrics_df=model_metrics_df,
        method_summary_df=method_summary_df,
        counterfactual_df=counterfactual_df,
    )

    with (out_dir / "raw_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_accuracy": float(overall_acc),
                "gender_0_accuracy": float(gender0_acc),
                "gender_1_accuracy": float(gender1_acc),
                "counterfactual_change_rate": float(counterfactual_df["changed_prediction"].mean()),
            },
            f,
            indent=2,
        )

    print("Done. Artifacts saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
