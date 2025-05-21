import os
import json
from collections import Counter
import pandas as pd
import sys 

class EXISTReader:
    # --------------------------------------------------
    # Hard‐labeling helpers
    # --------------------------------------------------
    def __hardlabeling1(self, annotations):
        cnt = Counter(annotations).most_common()
        if cnt[0][1] > 3: return cnt[0][0]
        if len(cnt) == 2:  return "AMBIGUOUS"

    def __hardlabeling2(self, annotations):
        cnt = Counter(annotations).most_common()
        if len(cnt) == 1: return cnt[0][0]
        if cnt[0][1] > 2 and cnt[1][1] != cnt[0][1]:
            return cnt[0][0]
        return "AMBIGUOUS"

    def __hardlabeling3(self, annotations):
        # multi‐label union
        union = sorted({lbl for ann in annotations for lbl in ann})
        # drop UNKNOWN but keep "-" as a valid label
        union = [lbl for lbl in union if lbl != "UNKNOWN"]
        if not union: return "AMBIGUOUS"
        return union

    # --------------------------------------------------
    # Constructor: detect automatically if there *are* labels
    # --------------------------------------------------
    def __init__(self, file_path, task=1):
        self.task = f"task{task}"
        # estructuras temporales
        buckets = {"EN": {"id": [], "text": [], "label1": [], "label2": [], "label3": []},
                   "ES": {"id": [], "text": [], "label1": [], "label2": [], "label3": []}}
        
        with open(file_path, 'r', encoding='utf8') as f:
            entries = json.load(f)
        
        splits = set()
        for eid, entry in entries.items():
            split, lang = entry['split'].split("_")
            splits.add(split)
            buckets[lang]["id"].append(eid)
            buckets[lang]["text"].append(entry["tweet"])
            
            # Si vienen etiquetas, procesa; si no, déjalos vacíos
            key1 = f"labels_{self.task}_1"
            if key1 in entry:
                l1 = self.__hardlabeling1(entry[key1])
                l2 = self.__hardlabeling2(entry[f"labels_{self.task}_2"])
                l3 = self.__hardlabeling3(entry[f"labels_{self.task}_3"])
                buckets[lang]["label1"].append(l1)
                buckets[lang]["label2"].append(l2)
                buckets[lang]["label3"].append(l3)
        
        assert len(splits) == 1
        
        # Determina si hay etiquetas (train/dev) o no (test)
        self.hasLabels = len(buckets["EN"]["label1"]) > 0
        
        # Construye allData según haya o no etiquetas
        if self.hasLabels:
            allData = {
                "id":    buckets["EN"]["id"]    + buckets["ES"]["id"],
                "text":  buckets["EN"]["text"]  + buckets["ES"]["text"],
                "label1":buckets["EN"]["label1"]+ buckets["ES"]["label1"],
                "label2":buckets["EN"]["label2"]+ buckets["ES"]["label2"],
                "label3":buckets["EN"]["label3"]+ buckets["ES"]["label3"],
                "language": ["EN"]*len(buckets["EN"]["id"]) + ["ES"]*len(buckets["ES"]["id"])
            }
            cols = ["id","text","label1","label2","label3","language"]
        else:
            allData = {
                "id": buckets["EN"]["id"]   + buckets["ES"]["id"],
                "text": buckets["EN"]["text"] + buckets["ES"]["text"],
                "language": ["EN"]*len(buckets["EN"]["id"]) + ["ES"]*len(buckets["ES"]["id"])
            }
            cols = ["id","text","language"]
        
        self.dataframe = pd.DataFrame(allData, columns=cols)

    # --------------------------------------------------
    # get(): devuelve tuplas según subtask y presencia de etiquetas
    # --------------------------------------------------
    def get(self, lang="EN", subtask="1", preprocess_fn=None):
        df = self.dataframe[self.dataframe["language"] == lang.upper()]
        if not self.hasLabels:
            # Test set: sólo id,text
            return df["id"], df["text"]
        
        # Train/dev:
        if subtask == "1":
            df = df[df["label1"].isin(["YES","NO"])]
            return df["id"], df["text"], df["label1"]
        
        if subtask == "2":
            df = df[df["label1"].isin(["YES","NO"])]
            df = df[df["label2"].isin(["-","JUDGEMENTAL","REPORTED","DIRECT"])]
            # map "-" → "NO" si quieres
            df["label2"] = df["label2"].replace({"-": "NO"})
            return df["id"], df["text"], df["label2"]
        
        if subtask == "3":
            df = df[df["label1"].isin(["YES","NO"])]
            # label3 ya es lista, incluye "-" si estaba
            df = df[df["label3"] != "AMBIGUOUS"]
            # aplanar o dejar como lista, según tu pipeline:
            return df["id"], df["text"], df["label3"]



