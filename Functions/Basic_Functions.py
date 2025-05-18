import os
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from pyevall.evaluation import PyEvALLEvaluation
from pyevall.metrics.metricfactory import MetricFactory
from pyevall.reports.reports import PyEvALLReport
from pyevall.utils.utils import PyEvALLUtils

class SexismDataset(Dataset):
    def __init__(self, texts, labels, ids, tokenizer, max_len=128, pad="max_length", trunc=True,rt='pt'):
        self.texts = texts.tolist()
        self.labels = labels
        self.ids = ids
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad = pad
        self.trunc = trunc
        self.rt = rt

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,padding=self.pad, truncation=self.trunc,
            return_tensors=self.rt
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'id': torch.tensor(self.ids[idx], dtype=torch.long)
        }

def compute_metrics_1(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def compute_metrics_2(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def compute_metrics_3(pred, lencoder):
    labels = pred.label_ids
    #preds = pred.predictions.argmax(-1)
    preds = torch.sigmoid(torch.tensor(pred.predictions)).numpy()
    preds_binary = (preds >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds_binary, average=None, zero_division=0
    )
    acc = accuracy_score(labels, preds_binary)
    icm= ICMWrapper(lencoder.inverse_transform(preds_binary), lencoder.inverse_transform(labels), multi=True)
    # Macro averages
    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    f1_macro = np.mean(f1)
    metrics = {}
    metrics.update({
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'ICM': icm
    })
    return metrics

def ICMWrapper(pred, labels, multi=False,ids=None):
    test = PyEvALLEvaluation()
    metrics=[MetricFactory.ICM.value]
    params= dict()
    fillLabel=None
    if multi:
        params[PyEvALLUtils.PARAM_REPORT]="embedded"
        hierarchy={"True":['IDEOLOGICAL-INEQUALITY', 'STEREOTYPING-DOMINANCE', 'MISOGYNY-NON-SEXUAL-VIOLENCE', 'OBJECTIFICATION', 'SEXUAL-VIOLENCE'],
        "False":[]}
        params[PyEvALLUtils.PARAM_HIERARCHY]=hierarchy
        fillLabel = lambda x: ["False"] if len(x)== 0 else x
    else:
        params[PyEvALLUtils.PARAM_REPORT]="simple"
        fillLabel = lambda x: str(x)


    truth_name, predict_name=None, None
    if ids is None:
        ids=list(range(len(labels)))

    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as truth:
        truth_name=truth.name
        truth_df=pd.DataFrame({'test_case': ['EXIST2025']*len(labels),
                        'id': [str(x) for x in ids],
                        'value': [fillLabel(x) for x in labels]})
        if multi==True:
            truth_df=truth_df.astype('object')
        truth.write(truth_df.to_json(orient="records"))

    with  tempfile.NamedTemporaryFile(mode='w', delete=False) as predict:
        predict_name=predict.name
        predict_df=pd.DataFrame({'test_case': ['EXIST2025']*len(pred),
                        'id': [str(x) for x in ids],
                        'value': [fillLabel(x) for x in pred]})
        if multi==True:
            predict_df=predict_df.astype('object')
        predict.write(predict_df.to_json(orient="records"))

    report = test.evaluate(predict_name, truth_name, metrics, **params)
    os.unlink(truth_name)
    os.unlink(predict_name)

    icm = None
    if 'metrics' in report.report:
        if 'ICM' in report.report["metrics"]: icm=float(report.report["metrics"]['ICM']["results"]["average_per_test_case"])
    return icm