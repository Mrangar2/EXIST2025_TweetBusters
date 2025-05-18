import os
import json
import sys
from collections import Counter
import pandas as pd

# TASK1: 'NO', 'YES'
# TASK2: '-', 'JUDGEMENTAL', 'REPORTED', 'DIRECT', 'UNKNOWN'
# TASK3:  'OBJECTIFICATION', 'STEREOTYPING-DOMINANCE', '-', 'MISOGYNY-NON-SEXUAL-VIOLENCE', 'IDEOLOGICAL-INEQUALITY', 'SEXUAL-VIOLENCE', 'UNKNOWN'

class EXISTReader:
    # Task 1, the class annotated by more than 3 annotators is selected
    # Task 2, the class annotated by more than 2 annotators is selected
    # Task 3 the classes annotated by more than 1 annotator are selected
    def __hardlabeling1(self, annotations):
        count  = Counter(annotations).most_common()
        if count[0][1]  > 3: return count[0][0]
        if len(count) == 2: return "AMBIGUOUS"

    def __hardlabeling2(self, annotations):
        count = counter = Counter(annotations).most_common()
        if len(count) == 1: return counter[0][0]
        if len(count) > 1 and count[0][1] > 2 and  count[1][1]!=count[0][1] : return counter[0][0]
        return "AMBIGUOUS"

    def __hardlabeling3(self, annotations):
        unionAnnotations = []
        for anno in annotations: unionAnnotations+=anno
        unionAnnotations = list(set(unionAnnotations))
        unionAnnotations.sort()
        if "-" in unionAnnotations: unionAnnotations.remove("-")
        if "UNKNOWN" in unionAnnotations:  unionAnnotations.remove("UNKNOWN")
        if len(unionAnnotations) == 0: return "AMBIGUOUS"
        return unionAnnotations

    def __init__(self, file, task=1):
        self.file = file
        self.task = f"task{task}"
        mapLabelToId = {"task1": {'NO': 0, 'YES': 1, "AMBIGUOUS": 2},
                        "task2": {'-': 4, 'JUDGEMENTAL': 0, 'REPORTED': 1, 'DIRECT': 2, 'UNKNOWN': 3, "AMBIGUOUS": 5},
                        "task3": {'OBJECTIFICATION': 0, 'STEREOTYPING-DOMINANCE': 1, 'MISOGYNY-NON-SEXUAL-VIOLENCE': 2,
                                  'IDEOLOGICAL-INEQUALITY': 3, 'SEXUAL-VIOLENCE': 4, 'UNKNOWN': 5, '-': 6,
                                  "AMBIGUOUS": 7}}

        mapIdToLabel = {"task1": {0: 'NO', 1: 'YES', 2: "AMBIGUOUS"},
                        "task2": {4: '-', 0: 'JUDGEMENTAL', 1: 'REPORTED', 2: 'DIRECT', 3: 'UNKNOWN', 4: "AMBIGUOUS"},
                        "task3": {0: 'OBJECTIFICATION', 1: 'STEREOTYPING-DOMINANCE', 2: 'MISOGYNY-NON-SEXUAL-VIOLENCE',
                                  3: 'IDEOLOGICAL-INEQUALITY', 4: 'SEXUAL-VIOLENCE', 5: 'UNKNOWN', 6: '-',
                                  7: "AMBIGUOUS"}}

        self.dataset = {"ES":{'id': [], 'text': [], 'label1': [], "label2": [], "label3": []},
                      "EN": {'id': [], 'text': [], 'label1': [], "label2": [], "label3": []}
                      }

        splits = set()
        with open(self.file, 'r', encoding='utf-8') as mytrain:
            entries = json.load(mytrain)
            for entryid in entries:
                entry = entries[entryid]
                #print(entry)
                
                split, lang = entry['split'].split("_")
                splits.add(split)

                self.dataset[lang]['id'].append(entryid)
                self.dataset[lang]['text'].append(entry["tweet"])

                if f"labels_{self.task}_1" in entry:
                    
                    label1 = entry[f"labels_{self.task}_1"]
                    label2 = entry[f"labels_{self.task}_2"]
                    label3 = entry[f"labels_{self.task}_3"]

                    label1 = self.__hardlabeling1(label1)
                    label2 = self.__hardlabeling2(label2)
                    label3 = self.__hardlabeling3(label3)
                    
                    self.dataset[lang]["label1"].append(label1)
                    self.dataset[lang]["label2"].append(label2)
                    self.dataset[lang]["label3"].append(label3)

        assert len(splits) == 1
        split = list(splits)
        allData= {
            "id": self.dataset["EN"]["id"] + self.dataset["ES"]["id"],
            "text": self.dataset["EN"]["text"] + self.dataset["ES"]["text"],
            "label1": self.dataset["EN"]["label1"] + self.dataset["ES"]["label1"],
            "label2": self.dataset["EN"]["label2"] + self.dataset["ES"]["label2"],
            "label3": self.dataset["EN"]["label3"] + self.dataset["ES"]["label3"],
            "language": ["EN"] * len(self.dataset["EN"]["id"]) + ["ES"] * len(self.dataset["ES"]["id"]),

        }

        self.dataframe = pd.DataFrame(allData, columns=["id", "text", "label1", "label2", "label3", "language"], index=None)


    def get(self, lang="EN", subtask="1"):
        data=None
        if subtask == "1":
            data = self.dataframe[self.dataframe["language"] == lang.upper()]
            data = data[data["label1"].isin(["YES","NO"])]
            return data["id"], data["text"], data["label1"]

        if subtask == "2":
            data= self.dataframe[self.dataframe["language"] == lang.upper()]
            data = data[data["label1"].isin(["YES","NO"])]
            data = data[data["label2"].isin(['-','JUDGEMENTAL','REPORTED', 'DIRECT'])]
            return data["id"], data["text"], data["label2"]

        if subtask == "3":
            data= self.dataframe[self.dataframe["language"] == lang.upper()]
            data = data[data["label1"].isin(["YES","NO"])]
            data = data[data["label3"] != 'AMBIGUOUS']
            return data["id"], data["text"], data["label3"]



if __name__ == '__main__':
    reader = EXISTReader("EXIST_2025_Dataset_V0.2/EXIST2025_training.json")
    T1 = reader.get(lang="ES", subtask="1")
    print(T1)


