import pandas as pd
import torch
import optuna

from sklearn.preprocessing import LabelEncoder

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from peft import LoraConfig, get_peft_model, TaskType

# Dataset para single-label (task 1 y 2)
from Basic_Functions import SexismDataset

# Funciones de evaluación para cada tarea
from Basic_Functions import compute_metrics_1, compute_metrics_2, compute_metrics_3

# Pipelines LoRA específicos para cada tarea
from Tasks_LoRA_Pipelines import (
    sexism_classification_pipeline_task1_LoRA,
    sexism_classification_pipeline_task2_LoRA,
    sexism_classification_pipeline_task3_LoRA
)

def run_lora_experiments(
    task_num: int,
    model_names: list,
    params: dict,
    trainInfo,
    devInfo,
    testInfo=None
) -> pd.DataFrame:
    """
    Runs a batch of LoRA fine-tuning experiments for task 1, 2 or 3.

    Args:
        task_num:       1, 2 or 3 – which sexism_classification_pipeline_taskX_LoRA to call.
        model_names:    List of HF model names (strings).
        params:         Dict of hyperparameters to pass as **kwargs.
        trainInfo:      Your train tuple (ids, texts, labels).
        devInfo:        Your dev tuple (ids, texts, labels).
        testInfo:       Optional test tuple (ids, texts) for predictions.

    Returns:
        DataFrame with one row per model, columns = ['model', *eval_metrics..., 'epoch'].
    """
    # map task number → (pipeline_fn, nlabels, problem_type)
    pipeline_map = {
        1: (sexism_classification_pipeline_task1_LoRA, 2, "single_label_classification"),
        2: (sexism_classification_pipeline_task2_LoRA, 4, "single_label_classification"),
        3: (sexism_classification_pipeline_task3_LoRA, 6, "multi_label_classification"),
    }
    if task_num not in pipeline_map:
        raise ValueError(f"Unsupported task {task_num}. Choose 1, 2 or 3.")

    pipeline_fn, nlabels, ptype = pipeline_map[task_num]
    metrics_list = []

    for model_name in model_names:
        print(f"→ Running task{task_num} LoRA on {model_name!r} …")
        # pipeline returns (model, eval_results) if testInfo is None
        _, eval_results = pipeline_fn(
            trainInfo,
            devInfo,
            testInfo,
            model_name,
            nlabels,
            ptype,
            **params
        )
        # eval_results is a dict like {'eval_accuracy':…, 'eval_f1':…, …, 'epoch':…}
        row = {"model": model_name}
        row.update(eval_results)
        metrics_list.append(row)

    df = pd.DataFrame(metrics_list)
    # optional: reorder cols so 'model' comes first
    cols = ["model"] + [c for c in df.columns if c != "model"]
    return df[cols]


def select_best_model(df: pd.DataFrame, task_num: int) -> str:
    """
    Given the DataFrame of metrics (as returned by run_lora_experiments),
    computes a composite 'score' depending on task and returns the best model name.
    """
    if task_num == 1:
        df["score"] = 0.6 * df["eval_f1"] + 0.4 * df["eval_accuracy"]
    elif task_num == 2:
        df["score"] = (
            0.5 * df["eval_f1"] +
            0.3 * df["eval_accuracy"] +
            0.2 * df["eval_precision"]
        )
    elif task_num == 3:
        df["score"] = (
            0.4 * df["eval_micro_f1"] +
            0.4 * df["eval_macro_f1"] +
            0.2 * df["eval_subset_accuracy"]
        )
    else:
        raise ValueError(f"Unknown task {task_num}")

    best_row = df.sort_values("score", ascending=False).iloc[0]
    print(f"→ Winning model for task{task_num}: {best_row['model']} (score={best_row['score']:.4f})")
    return best_row["model"]


def optimize_lora_hyperparams(
    task_num: int,
    best_model_name: str,
    params_base: dict,
    trainInfo,
    devInfo,
    n_trials: int = 20
) -> dict:
    """
    Runs an Optuna search to tune LoRA hyperparameters for the given task
    and model checkpoint. Returns the optimized params dict.

    Args:
        task_num:         1, 2 or 3 – selects number of labels & compute_metrics.
        best_model_name:  HF model checkpoint chosen after initial comparison.
        params_base:      Base params dict (num_train_epochs, batch_sizes, etc.).
        trainInfo:        Tuple (ids, texts, labels) for training.
        devInfo:          Tuple (ids, texts, labels) for validation.
        n_trials:         Number of Optuna trials to run.
    """
    # 1) Map task → (nlabels, compute_metrics_fn)
    compute_map = {
        1: (2, compute_metrics_1),
        2: (4, compute_metrics_2),
        3: (6, compute_metrics_3),
    }
    if task_num not in compute_map:
        raise ValueError("task_num must be 1, 2, or 3")
    nlabels, compute_metrics = compute_map[task_num]

    # 2) Prepare tokenizer, label encoder, and PyTorch datasets
    tokenizer = AutoTokenizer.from_pretrained(best_model_name)
    label_enc = LabelEncoder()
    y_train = label_enc.fit_transform(trainInfo[2])
    y_dev   = label_enc.transform(devInfo[2])

    train_dataset = SexismDataset(
        texts=trainInfo[1],
        labels=y_train,
        ids=[int(x) for x in trainInfo[0]],
        tokenizer=tokenizer
    )
    eval_dataset = SexismDataset(
        texts=devInfo[1],
        labels=y_dev,
        ids=[int(x) for x in devInfo[0]],
        tokenizer=tokenizer
    )

    # 3) Define Optuna objective
    def objective(trial):
        # 3.1) Suggest hyperparameters
        lr         = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        r          = trial.suggest_int("r", 8, 128, log=True)
        lora_alpha = trial.suggest_int("lora_alpha", 8, 64, log=True)

        # 3.2) Load base model & apply LoRA
        base_model = AutoModelForSequenceClassification.from_pretrained(
            best_model_name, num_labels=nlabels
        )
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            target_modules=["query", "value"],
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            bias="none"
        )
        peft_model = get_peft_model(base_model, lora_cfg)

        # 3.3) Quick-training arguments
        args = TrainingArguments(
            output_dir=f"optuna_trial_{trial.number}",
            num_train_epochs=10,
            per_device_train_batch_size=params_base["per_device_train_batch_size"],
            per_device_eval_batch_size=params_base["per_device_eval_batch_size"],
            learning_rate=lr,
            logging_steps=params_base.get("logging_steps", 100),
            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            report_to="none"
        )

        # 3.4) Trainer
        trainer = Trainer(
            model=peft_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics
        )

        # 3.5) Train & evaluate
        trainer.train()
        res = trainer.evaluate()
        return res["eval_f1"]

    # 4) Run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    print("✅ Optimized LoRA hyperparameters:", best_params)

    # 5) Merge with base params and return
    optimized_params = params_base.copy()
    optimized_params.update(best_params)
    return optimized_params
