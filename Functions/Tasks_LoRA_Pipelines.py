import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from functools import partial

from Basic_Functions import compute_metrics_1, compute_metrics_2, compute_metrics_3
from Basic_Functions import SexismDataset

#######################################TASK 1###############################################

######################################CHANGE###############################################
from peft import LoraConfig, get_peft_model, TaskType
###########################################################################################

def sexism_classification_pipeline_task1_LoRA(trainInfo, devInfo, testInfo=None, model_name='roberta-base', nlabels=2, ptype="single_label_classification", **args):
    # Model and Tokenizer
    labelEnc = LabelEncoder()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=nlabels,
        problem_type=ptype
    )

    ######################################CHANGE###############################################
    # Configure LoRA
    lora_config = LoraConfig(
    task_type= args.get("task_type", TaskType.SEQ_CLS),
    target_modules= args.get("target_modules", ["query", "value"]),
    r= args.get("rank", 64),  # Rank of LoRA adaptation
    lora_alpha=args.get("lora_alpha", 32),  # Scaling factor
    lora_dropout=args.get("lora_dropout", 0.1),
    bias=args.get("bias", "none")
)
    ###########################################################################################

    ######################################CHANGE###############################################
    # Prepare LoRA model
    peft_model = get_peft_model(model, lora_config)

    ###########################################################################################
    # Prepare datasets
    train_dataset = SexismDataset(trainInfo[1], labelEnc.fit_transform(trainInfo[2]),[int(x) for x in trainInfo[0]], tokenizer )
    val_dataset = SexismDataset(devInfo[1], labelEnc.transform(devInfo[2]), [int(x) for x in devInfo[0]], tokenizer)

    # Training Arguments
    training_args = TrainingArguments(
        report_to="none", # alt: "wandb", "tensorboard" "comet_ml" "mlflow" "clearml"
        output_dir= args.get('output_dir', './results_task1_LoRA0'),
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
        metric_for_best_model=args.get('metric_for_best_model',"f1")
    )

    # Initialize Trainer
    trainer = Trainer(
        ######################################CHANGE###############################################
        # Prepare LoRA model
        model=peft_model,
        ###########################################################################################
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

    ######################################CHANGE###############################################
    #Saving the new weigths for the LoRA model
    trainer.save_model('./final_best_model_LoRA')
    # Notice that, in this case only the LoRA matrices are saved.
    # The weigths for the classification head are not saved.
    ###########################################################################################

    ######################################CHANGE###############################################
    #Mixing the LoRA matrices with the weigths of the base model used
    mixModel=peft_model.merge_and_unload()
    mixModel.save_pretrained("./final_best_model_mixpeft")
    # IN this case the full model is saved.
    ###########################################################################################

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
        return mixModel, submission_df
    return mixModel, eval_results



def sexism_classification_pipeline_task2_LoRA(trainInfo, devInfo, testInfo=None, model_name='roberta-base', nlabels=4, ptype="single_label_classification", **args):
    # Model and Tokenizer
    labelEnc = LabelEncoder()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=nlabels,
        problem_type=ptype
    )
    ######################################CHANGE###############################################
    # Configure LoRA
    lora_config = LoraConfig(
    task_type= args.get("task_type", TaskType.SEQ_CLS),
    target_modules= args.get("target_modules", ["query", "value"]),
    r= args.get("rank", 64),  # Rank of LoRA adaptation
    lora_alpha=args.get("lora_alpha", 32),  # Scaling factor
    lora_dropout=args.get("lora_dropout", 0.1),
    bias=args.get("bias", "none"),
)
    ###########################################################################################

    ######################################CHANGE###############################################
    # Prepare LoRA model
    peft_model = get_peft_model(model, lora_config)

    ###########################################################################################

    # Prepare datasets
    train_dataset = SexismDataset(trainInfo[1], labelEnc.fit_transform(trainInfo[2]),[int(x) for x in trainInfo[0]], tokenizer )
    val_dataset = SexismDataset(devInfo[1], labelEnc.transform(devInfo[2]), [int(x) for x in devInfo[0]], tokenizer)

    # Training Arguments
    training_args = TrainingArguments(
        report_to="none", # alt: "wandb", "tensorboard" "comet_ml" "mlflow" "clearml"
        output_dir= args.get('output_dir', './results_task2_LoRA0'),
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
        metric_for_best_model=args.get('metric_for_best_model',"f1") # F1 para el concurso
    )

    # Initialize Trainer
    trainer = Trainer(
        ######################################CHANGE###############################################
        # Prepare LoRA model
        model=peft_model,
        ###########################################################################################
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

    ######################################CHANGE###############################################
    #Saving the new weigths for the LoRA model
    trainer.save_model('./final_best_model_LoRA_2')
    # Notice that, in this case only the LoRA matrices are saved.
    # The weigths for the classification head are not saved.
    ###########################################################################################

    ######################################CHANGE###############################################
    #Mixing the LoRA matrices with the weigths of the base model used
    mixModel=peft_model.merge_and_unload()
    mixModel.save_pretrained("./final_best_model_mixpeft_2")
    # IN this case the full model is saved.
    ###########################################################################################

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
        print("Prediction for TASK 2 completed. Results saved to sexism_predictions_task1.csv")
        return mixModel, submission_df
    return mixModel, eval_results


def sexism_classification_pipeline_task3_LoRA(trainInfo, devInfo, testInfo=None,
                                              model_name='roberta-base',
                                              nlabels=6,
                                              ptype="multi_label_classification",
                                              **args):
    """
    Pipeline para clasificación multi-label (TASK 3) con adaptación LoRA.

    Parámetros:
      - trainInfo, devInfo, testInfo: estructuras con información, donde se espera que:
          trainInfo = (lista_ids, lista_textos, lista_etiquetas)
          devInfo   = (lista_ids, lista_textos, lista_etiquetas)
          testInfo  = (lista_ids, lista_textos, _dummy_)  [para test se generan etiquetas dummy]
      - model_name: nombre (o path) del modelo base.
      - nlabels: número de etiquetas.
      - ptype: tipo de problema para el modelo ('multi_label_classification').
      - **args: permite ajustar hiperparámetros (por ejemplo, learning_rate, num_train_epochs, etc.) y la configuración LoRA.

    Retorna:
      - mixModel: modelo final (con LoRA fusionado con los pesos base).
      - submission_df o eval_results: según se proporcione o no testInfo.
    """
    # Codificador de etiquetas para multi-label
    labelEnc = MultiLabelBinarizer()


    # Modelo y Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=nlabels,
        problem_type=ptype
    )

    ###################################### CONFIGURACIÓN LoRA ###############################################
    lora_config = LoraConfig(
        task_type=args.get("task_type", TaskType.SEQ_CLS),
        target_modules=args.get("target_modules", ["query", "value"]),
        r=args.get("rank", 64),                # Rango de adaptación LoRA
        lora_alpha=args.get("lora_alpha", 32),  # Factor de escalado LoRA
        lora_dropout=args.get("lora_dropout", 0.1),
        bias=args.get("bias", "none")
    )
    # Aplicar LoRA al modelo base
    peft_model = get_peft_model(base_model, lora_config)
    #######################################################################################################

    # Preparación de datasets usando SexismDatasetMulti para clasificación multi-label
    train_dataset = SexismDatasetMulti(
        texts=trainInfo[1],
        labels=labelEnc.fit_transform(trainInfo[2]),
        ids=[int(x) for x in trainInfo[0]],
        tokenizer=tokenizer
    )
    val_dataset = SexismDatasetMulti(
        texts=devInfo[1],
        labels=labelEnc.transform(devInfo[2]),
        ids=[int(x) for x in devInfo[0]],
        tokenizer=tokenizer
    )

    # Configuración de los argumentos de entrenamiento
    training_args = TrainingArguments(
        report_to="none",  # Alternativas: "wandb", "tensorboard", etc.
        output_dir=args.get('output_dir', './results_task3_LoRA'),
        num_train_epochs=args.get('num_train_epochs', 5),
        learning_rate=args.get('learning_rate', 5e-5),
        per_device_train_batch_size=args.get('per_device_train_batch_size', 16),
        per_device_eval_batch_size=args.get('per_device_eval_batch_size', 64),
        warmup_steps=args.get('warmup_steps', 500),
        weight_decay=args.get('weight_decay', 0.01),
        logging_dir=args.get('logging_dir', './logs'),
        logging_steps=args.get('logging_steps', 10),
        eval_strategy=args.get('eval_strategy', 'epoch'),
        save_strategy=args.get('save_strategy', "epoch"),
        save_total_limit=args.get('save_total_limit', 1),
        load_best_model_at_end=args.get('load_best_model_at_end', True),
        metric_for_best_model=args.get('metric_for_best_model', "ICM")
    )

    # Inicialización del Trainer con función de métricas propia para multi-label (compute_metrics_3)
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=partial(compute_metrics_3, lencoder=labelEnc),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.get("early_stopping_patience", 3))]
    )

    # Entrenamiento
    trainer.train()

    # Evaluación en el conjunto de validación
    eval_results = trainer.evaluate()
    print("Validation Results:", eval_results)

    ###################################### GUARDADO DEL MODELO ###############################################
    # Se guardan los pesos de LoRA (solo las matrices LoRA)
    trainer.save_model('./final_best_model_LoRA')
    # Se realiza la fusión de las matrices LoRA con los pesos base para obtener el modelo completo
    mixModel = peft_model.merge_and_unload()
    mixModel.save_pretrained("./final_best_model_mixpeft_3")
    #######################################################################################################

    if testInfo is not None:
        # Preparación del dataset de test (se crean etiquetas dummy, pues en test solo se hacen predicciones)
        test_dataset = SexismDatasetMulti(
            texts=testInfo[1],
            labels=[[0] * nlabels for _ in range(len(testInfo[1]))],
            ids=[int(x) for x in testInfo[0]],
            tokenizer=tokenizer
        )

        # Realizar predicciones
        predictions = trainer.predict(test_dataset)
        # Se aplican la función sigmoide y se define un umbral (0.5) para obtener las etiquetas finales
        predicted_probs = torch.sigmoid(torch.tensor(predictions.predictions)).numpy()
        predicted_labels = (predicted_probs >= 0.5).astype(int)

        # Creación del DataFrame para la submission
        submission_df = pd.DataFrame({
            'id': testInfo[0],
            'label': labelEnc.inverse_transform(predicted_labels),
            'test_case': ["EXIST2025"] * len(predicted_labels)
        })
        submission_df.to_csv('sexism_predictions_task3_LoRA.csv', index=False)
        print("Prediction for TASK 3 (LoRA) completed. Results saved to sexism_predictions_task3_LoRA.csv")
        return mixModel, submission_df

    return mixModel, eval_results