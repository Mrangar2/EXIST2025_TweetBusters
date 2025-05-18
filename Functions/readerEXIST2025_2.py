import os
import json
import sys
from collections import Counter
import pandas as pd

class AnnotationReader:
    """
    Lee un diccionario de anotaciones con la estructura dada y construye un DataFrame de pandas.
    Campos: id_EXIST, lang, tweet, number_annotators, annotators, gender_annotators,
            age_annotators, ethnicities_annotators, study_levels_annotators, countries_annotators, split
    """
    def __init__(self, data: dict):
        # data puede venir de json.load(...) o de un diccionario en memoria
        self.data = data
        self.df = self._build_dataframe()

    def _build_dataframe(self) -> pd.DataFrame:
        rows = []
        for entry_id, entry in self.data.items():
            rows.append({
                'id_EXIST': entry.get('id_EXIST'),
                'lang': entry.get('lang'),
                'tweet': entry.get('tweet'),
                'number_annotators': entry.get('number_annotators'),
                'annotators': entry.get('annotators'),
                'gender_annotators': entry.get('gender_annotators'),
                'age_annotators': entry.get('age_annotators'),
                'ethnicities_annotators': entry.get('ethnicities_annotators'),
                'study_levels_annotators': entry.get('study_levels_annotators'),
                'countries_annotators': entry.get('countries_annotators'),
                'split': entry.get('split')
            })
        return pd.DataFrame(rows)

    def get_by_lang(self, lang: str) -> pd.DataFrame:
        """Devuelve sólo las filas cuyo campo 'lang' coincide."""
        return self.df[self.df['lang'] == lang]

    def get_by_split(self, split: str) -> pd.DataFrame:
        """Devuelve sólo las filas cuyo campo 'split' coincide."""
        return self.df[self.df['split'] == split]


class EXISTReader:
    """
    Reader para el dataset EXIST2025. Permite generar etiquetas duras (hard-labeling)
    según la sub-tarea seleccionada (1, 2 ó 3), y acceder tanto a metadatos como a
    DataFrame final etiquetado.
    """
    def __hardlabeling1(self, annotations):
        # Más de 3 anotadores
        count = Counter(annotations).most_common()
        if count[0][1] > 3:
            return count[0][0]
        if len(count) == 2:
            return "AMBIGUOUS"
        return None

    def __hardlabeling2(self, annotations):
        # Más de 2 anotadores
        count = Counter(annotations).most_common()
        if len(count) == 1:
            return count[0][0]
        if len(count) > 1 and count[0][1] > 2 and count[1][1] != count[0][1]:
            return count[0][0]
        return "AMBIGUOUS"

    def __hardlabeling3(self, annotations):
        # Unión de etiquetas (más de 1 anotador)
        union = []
        for anno in annotations:
            union += anno
        union = sorted(set(union) - {'-', 'UNKNOWN'})
        if not union:
            return "AMBIGUOUS"
        return union

    def __init__(self, file_path, task=1):
        self.file = file_path
        self.task = f"task{task}"
        # Carga de datos crudos
        with open(self.file, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        # DataFrame con metadatos
        self.raw_df = AnnotationReader(entries).df

        # Mapeos de etiquetas a IDs (opcional)
        self.mapLabelToId = {
            'task1': {'NO': 0, 'YES': 1, 'AMBIGUOUS': 2},
            'task2': {'-': 4, 'JUDGEMENTAL': 0, 'REPORTED': 1, 'DIRECT': 2, 'UNKNOWN': 3, 'AMBIGUOUS': 5},
            'task3': {
                'OBJECTIFICATION': 0,
                'STEREOTYPING-DOMINANCE': 1,
                'MISOGYNY-NON-SEXUAL-VIOLENCE': 2,
                'IDEOLOGICAL-INEQUALITY': 3,
                'SEXUAL-VIOLENCE': 4,
                'UNKNOWN': 5,
                '-': 6,
                'AMBIGUOUS': 7
            }
        }

        # Estructura de almacenamiento
        self.dataset = {
            'ES': {'id': [], 'text': [], 'label1': [], 'label2': [], 'label3': []},
            'EN': {'id': [], 'text': [], 'label1': [], 'label2': [], 'label3': []}
        }
        splits = set()

        # Procesamiento principal con manejo de faltantes en test
        for entry_id, entry in entries.items():
            split, lang = entry['split'].split("_")  # p.ej. 'TEST_ES' -> ('TEST','ES')
            splits.add(split)
            # Añadir metadatos básicos
            self.dataset[lang]['id'].append(entry_id)
            self.dataset[lang]['text'].append(entry['tweet'])
            # Extraer labels si existen, sino None
            key1 = f"labels_{self.task}_1"
            if key1 in entry:
                raw1 = entry[key1]
                raw2 = entry[f"labels_{self.task}_2"]
                raw3 = entry[f"labels_{self.task}_3"]
                l1 = self.__hardlabeling1(raw1)
                l2 = self.__hardlabeling2(raw2)
                l3 = self.__hardlabeling3(raw3)
            else:
                l1, l2, l3 = None, None, None
            self.dataset[lang]['label1'].append(l1)
            self.dataset[lang]['label2'].append(l2)
            self.dataset[lang]['label3'].append(l3)

        assert len(splits) == 1, f"Se esperaba un solo split, pero se encontraron: {splits}"

        # Construcción de DataFrame final
        all_ids = self.dataset['EN']['id'] + self.dataset['ES']['id']
        all_text = self.dataset['EN']['text'] + self.dataset['ES']['text']
        all_l1 = self.dataset['EN']['label1'] + self.dataset['ES']['label1']
        all_l2 = self.dataset['EN']['label2'] + self.dataset['ES']['label2']
        all_l3 = self.dataset['EN']['label3'] + self.dataset['ES']['label3']
        all_lang = ['EN'] * len(self.dataset['EN']['id']) + ['ES'] * len(self.dataset['ES']['id'])

        self.dataframe = pd.DataFrame(
            {
                'id': all_ids,
                'text': all_text,
                'label1': all_l1,
                'label2': all_l2,
                'label3': all_l3,
                'language': all_lang
            },
            columns=['id', 'text', 'label1', 'label2', 'label3', 'language']
        )

    def get(self, lang='EN', subtask='1'):
        """
        Retorna tuplas (ids, texts, labels) filtradas según idioma y sub-tarea.
        subtask: '1', '2' o '3'.
        """
        df = self.dataframe[self.dataframe['language'] == lang.upper()]
        df = df[df['label1'].isin(['YES','NO'])]
        if subtask == '1':
            return df['id'], df['text'], df['label1']
        df = df[df['label2'].isin(['-','JUDGEMENTAL','REPORTED','DIRECT'])]
        if subtask == '2':
            return df['id'], df['text'], df['label2']
        df = df[df['label3'] != 'AMBIGUOUS']
        return df['id'], df['text'], df['label3']


if __name__ == '__main__':
    # Ejemplo de uso
    reader = EXISTReader('EXIST2025_training.json', task=1)
    print("Raw metadata head:")
    print(reader.raw_df.head())
    ids, texts, labels = reader.get(lang='ES', subtask='1')
    print("Primeras 5 etiquetas para ES, subtask 1:", list(labels[:5]))
