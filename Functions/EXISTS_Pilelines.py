import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)



def sexism_classification_pipeline_task1(trainInfo, devInfo, testInfo=None, model_name='roberta-base', nlabels=2, ptype="single_label_classification", **args):
    # Model and Tokenizer
    labelEnc= LabelEncoder()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=nlabels,
        problem_type=ptype
    )

    # Prepare datasets
    train_dataset = SexismDataset(trainInfo[1], labelEnc.fit_transform(trainInfo[2]),[int(x) for x in trainInfo[0]], tokenizer )
    val_dataset = SexismDataset(devInfo[1], labelEnc.transform(devInfo[2]), [int(x) for x in devInfo[0]], tokenizer)

    # Training Arguments
    training_args = TrainingArguments(
        report_to="none", # alt: "wandb", "tensorboard" "comet_ml" "mlflow" "clearml"
        output_dir= args.get('output_dir', './results'),
        num_train_epochs= args.get('num_train_epochs', 5),
        learning_rate=args.get('learning_rate', 5e-5),
        per_device_train_batch_size=args.get('per_device_train_batch_size', 16),
        per_device_eval_batch_size=args.get('per_device_eval_batch_size', 64),
        warmup_steps=args.get('warmup_steps', 500),
        weight_decay=args.get('weight_decay',0.01),
        logging_dir=args.get('logging_dir', './logs'),
        logging_steps=args.get('logging_steps', 10),
        eval_strategy=args.get('eval_strategy','epoch'),
        save_strategy=args.get('save_strategy', "epoch"),
        load_best_model_at_end=args.get('load_best_model_at_end', True),
        metric_for_best_model=args.get('metric_for_best_model',"f1")
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_1,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.get("early_stopping_patience",3))]
    )

    # Fine-tune the model
    trainer.train()

    # Evaluate on validation set
    eval_results = trainer.evaluate()
    print("Validation Results:", eval_results)

    # If there is a test dataset
    if testInfo is not None:
        # Prepare test dataset for prediction
        test_dataset = SexismDataset(testInfo[1], [0] * len(testInfo[1]),  [int(x) for x in testInfo[0]],   tokenizer)

        # Predict test set labels
        predictions = trainer.predict(test_dataset)
        predicted_labels = np.argmax(predictions.predictions, axis=1)

        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'id': testInfo[0],
            'label': labelEnc.inverse_transform(predicted_labels),
            "test_case": ["EXIST2025"]*len(predicted_labels)
        })
        submission_df.to_csv('sexism_predictions_task1.csv', index=False)
        print("Prediction for TASK 1 completed. Results saved to sexism_predictions_task1.csv")
        return model, submission_df
    return model, eval_results


def sexism_classification_pipeline_task2(trainInfo, devInfo, testInfo=None, model_name='bert-base-uncased', nlabels=3, ptype="single_label_classification", **args):
    # Model and Tokenizer
    labelEnc= LabelEncoder()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=nlabels,
        problem_type=ptype
    )

    # Prepare datasets
    train_dataset = SexismDataset(trainInfo[1], labelEnc.fit_transform(trainInfo[2]),[int(x) for x in trainInfo[0]], tokenizer )
    val_dataset = SexismDataset(devInfo[1], labelEnc.transform(devInfo[2]), [int(x) for x in devInfo[0]], tokenizer)

    # Training Arguments
    training_args = TrainingArguments(
        report_to="none", # alt: "wandb", "tensorboard" "comet_ml" "mlflow" "clearml"
        output_dir= args.get('output_dir', './results'),
        num_train_epochs= args.get('num_train_epochs', 5),
        learning_rate=args.get('learning_rate', 5e-5),
        per_device_train_batch_size=args.get('per_device_train_batch_size', 16),
        per_device_eval_batch_size=args.get('per_device_eval_batch_size', 64),
        warmup_steps=args.get('warmup_steps', 500),
        weight_decay=args.get('weight_decay',0.01),
        logging_dir=args.get('logging_dir', './logs'),
        logging_steps=args.get('logging_steps', 10),
        eval_strategy=args.get('eval_strategy','epoch'),
        save_strategy=args.get('save_strategy', "epoch"),
        save_total_limit=args.get('save_total_limit', 1),
        load_best_model_at_end=args.get('load_best_model_at_end', True),
        #metric_for_best_model=args.get('metric_for_best_model',"ICM")
        metric_for_best_model=args.get('metric_for_best_model',"f1")
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_2,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.get("early_stopping_patience",3))]
    )

    # Fine-tune the model
    trainer.train()

    # Evaluate on validation set
    eval_results = trainer.evaluate()
    print("Validation Results:", eval_results)

    # If there is a test dataset
    if testInfo is not None:
        # Prepare test dataset for prediction
        test_dataset = SexismDataset(testInfo[1], [0] * len(testInfo[1]),  [int(x) for x in testInfo[0]],   tokenizer)

        # Predict test set labels
        predictions = trainer.predict(test_dataset)
        predicted_labels = np.argmax(predictions.predictions, axis=1)

        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'id': testInfo[0],
            'label': labelEnc.inverse_transform(predicted_labels),
            "test_case": ["EXIST2025"]*len(predicted_labels)

        })
        submission_df.to_csv('sexism_predictions_task2.csv', index=False)
        print("Prediction TASK2 completed. Results saved to sexism_predictions_task2.csv")
        return model, submission_df
    return model, eval_results
