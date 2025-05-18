import os
import sys
import tempfile
import ast

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score

from transformers import AutoTokenizer, AutoModelForCausalLM

from pyevall.evaluation import PyEvALLEvaluation
from pyevall.metrics.metricfactory import MetricFactory
from pyevall.reports.reports import PyEvALLReport
from pyevall.utils.utils import PyEvALLUtils

from Basic_Functions import ICMWrapper
from huggingface_hub import login

def get_model(model_path):

    HF_TOKEN = "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"
    login(token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map='auto',
    #token=HF_TOKEN
    )
    return tokenizer, model

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
    
def simple_prompting_model(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Generate responses on pretrained model
    outputs = model.generate(
        **inputs,
        max_new_tokens=64, #Specify the new tokens that must be generated
        num_return_sequences=1, # Determines the number of different sequences the model should generate
        temperature=0.8 # Controls the randomness of the generated text. A higher temperature leads to more diverse and creative outputs, while a lower temperature results in more focused and deterministic outputs.
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

#######################################################################################################

def perform_incontext_classification(model, tokenizer, prompt, ntokens=8, nseq=1, temp=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=ntokens,
        num_return_sequences=nseq,
        temperature=temp
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def create_incontext_zero_prompt(task_description, query, context=None, role=None, output=None):
    prompt = f""
    if role!=None:
        prompt+= f"You are an expert {role}.\n\n"

    prompt += f"Task: {task_description}\n\n"
    if context!=None:
         prompt+= f"Context: {context}\n\n"
    if output != None:
         prompt+= f"Output Format: {output} \n\n"
    # Add query
    prompt+=f"Input: {query}\n"
    prompt+=f"Output: "
    return prompt


def output_postprocessing_incontext_zero_s1(output):
    outputp=output.rsplit("Output: ", 1)[-1].strip()
    #print(outputp)
    for line in outputp.split('\n'):
        line=line.strip()
        if line!="":
             if line.upper().startswith("YES"): return "YES"
             if line.upper().startswith("NO"): return "NO"
    return "UNK"

def incontext_zero_pipeline_task1(model, tokenizer, devData, testData, postprocess, **params):

    role= "in social psychology and linguistics with vast experience analyzing social media content and discriminative and harmful language"
    task ="""Sexist identification task is a binary text classification task which aim at determining
    whether or not a given tweet expresses ideas related to sexism in any of the three forms: it is sexist itself,
    it describes a sexist situation in which discrimination towards women occurs, or criticizes a sexist behaviour.
    The tweet is sexist (YES) or describes or criticizes a sexist situation. Not sexist. The tweet is not sexist (NO),
    nor it describes or criticizes a sexist situation."""
    output= "The output must be YES/NO."
    context=None

    ntokens=params.get("max_new_tokens", 8)
    nseq=params.get("num_return_sequences", 1)
    temp=params.get("temperature", 0.7)

    idqueries= devData[0].tolist()
    textqueries= devData[1].tolist()
    labelqueries=devData[2].tolist()

    predictions=[]
    for i in range(len(textqueries)):
        query=textqueries[i]
        filledPrompt=create_incontext_zero_prompt(task, query, context=None, role=role, output=output)
        #print("PROMPT >>> ", filledPrompt)
        response= perform_incontext_classification(model, tokenizer, filledPrompt, ntokens, nseq, temp)
        #print("#"*25, "ANSWER", "#"*25)
        #print(response)
        pred=postprocess(response)
        if pred in ["YES", "NO"]:
            predictions.append(pred)
        else:
            predictions.append("NO")
        print("ID: ", idqueries[i], "Ground Truth:", labelqueries[i], "Predicted: ", pred)

    # Create dev output DataFrame
    dev_df = pd.DataFrame({ 'id': idqueries,  'label': predictions, "tweet": textqueries, "test_case": ["EXIST2025"]*len(predictions) })
    dev_df.to_csv('sexism_dev_predictions_task1.csv', index=False)
    print("Prediction TASK1 completed. Results saved to sexism_dev_predictions_task1.csv")

    #Computing the quality measure for the subtask 1
    f1= f1_score(labelqueries, predictions, labels=None, pos_label="YES", average='binary')
    print(f"\nF1-Score Sexism: {f1}\n\n")

    idqueries= testData[0].tolist()
    textqueries= testData[1].tolist()
    test_predictions=[]
    for i in range(len(textqueries)):
        query=textqueries[i]
        filledPrompt=create_incontext_zero_prompt(task, query, context=None, role=role, output=output)
        response= perform_incontext_classification(model, tokenizer, filledPrompt,  ntokens, nseq, temp)
        pred=postprocess(response)
        if pred in ["YES", "NO"]:test_predictions.append(pred)
        else: test_predictions.append("NO")
        print("ID: ", idqueries[i],  "Predicted: ", pred)
        # Create submission DataFrame
    submission_df = pd.DataFrame({'id': idqueries, 'label': test_predictions, "test_case": ["EXIST2025"]*len(test_predictions) })
    submission_df.to_csv('sexism_predictions_task1.csv', index=False)
    print("Prediction TASK1 completed. Results saved to sexism_predictions_task1.csv")


def output_postprocessing_incontext_zero_s2(output):
    outputp=output.rsplit("Output: ", 1)[-1].strip()
    for line in outputp.split('\n'):
        line=line.strip()
        if line!="":
             if line.upper().startswith("DIRECT"): return "DIRECT"
             if line.upper().startswith("REPORTED"): return "REPORTED"
             if line.upper().startswith("JUDGEMENTAL"): return "JUDGEMENTAL"
    return "UNK"

def create_incontext_zero_prompt(task_description, query, context=None, role=None, output=None):
    prompt = f""
    if role!=None:
        prompt+= f"You are an expert {role}.\n\n"

    prompt += f"Task: {task_description}\n\n"
    if context!=None:
         prompt+= f"Context: {context}\n\n"
    if output != None:
         prompt+= f"Output Format: {output} \n\n"
    # Add query
    prompt+=f"Input: {query}\n"
    prompt+=f"Output: "
    return prompt


def incontext_zero_pipeline_task2(model, tokenizer, devData, testData, postprocessing, **params):

    role= "in social psychology and linguistics with vast experience analyzing social media content and discriminative and harmful language"
    task ="""Sexism Source Intention in tweets is a three class classification task aims to categorize
    the sexist tweets according to the intention of the author. This distinction allow us to differentiate
    sexism that is actually taking place online from sexism which is being suffered by women in other situations
    but that is being reported in social networks with the aim of complaining and fighting against sexism.
    The task include the following ternary classification of tweets.

    REPORTED: the intention is to report and share a sexist situation suffered by a woman or women.
    JUDGEMENTAL the intention is to condemn and critizise sexist situations or behaviours.
    DIRECT: The intention is to write a message that is sexist by itself or incites to be sexist.
     """
    output= "The output is a single label: REPORTED/JUDGEMENTAL/DIRECT/-."
    context=None

    ntokens=params.get("max_new_tokens", 8)
    nseq=params.get("num_return_sequences", 1)
    temp=params.get("temperature", 0.7)

    idqueries= devData[0].tolist()
    textqueries= devData[1].tolist()
    labelqueries=devData[2].tolist()

    predictions=[]
    for i in range(len(textqueries)):
        query=textqueries[i]
        filledPrompt=create_incontext_zero_prompt(task, query, context=None, role=role, output=output)
        #print("PROMPT >>> ", filledPrompt)
        response= perform_incontext_classification(model, tokenizer, filledPrompt, ntokens, nseq, temp)
        #print("#"*25, "ANSWER", "#"*25)
        #print(response)
        pred=postprocessing(response)
        if pred in ["DIRECT", "REPORTED", "JUDGEMENTAL"]: predictions.append(pred)
        else: predictions.append("DIRECT")
        print("ID: ", idqueries[i], "Ground Truth:", labelqueries[i], "Predicted: ", pred)

    # Create dev output DataFrame
    dev_df = pd.DataFrame({
        'id': idqueries,
        'label': predictions,
        "tweet": textqueries,
        "test_case": ["EXIST2025"]*len(predictions) })
    dev_df.to_csv('/content/drive/MyDrive/EXISTS2025/Results/sexism_dev_predictions_task2.csv', index=False)
    print("Evaluation TASK2 completed. Results saved to sexism_dev_predictions_task2.csv")

    #Computing the quality measure for the subtask 2
    f1= f1_score(labelqueries, predictions, labels=None, average='macro')
    print(f"\nMacro Average F1-Score Sexism: {f1}\n\n")

    idqueries= testData[0].tolist()
    textqueries= testData[1].tolist()
    test_predictions=[]
    for i in range(len(textqueries)):
        query=textqueries[i]
        filledPrompt=create_incontext_zero_prompt(task, query, context=None, role=role, output=output)
        #print("PROMPT >>> ", filledPrompt)
        response= perform_incontext_classification(model, tokenizer, filledPrompt,  ntokens, nseq, temp)
        #print("#"*25, "ANSWER", "#"*25)
        #print(response)
        pred=postprocessing(response)
        if pred in["DIRECT", "REPORTED", "JUDGEMENTAL"]:test_predictions.append(pred)
        else: test_predictions.append("DIRECT")
        print("ID: ", idqueries[i], "Predicted: ", pred)

    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': idqueries,
        'label': test_predictions,
        "test_case": ["EXIST2025"]*len(test_predictions) })
    submission_df.to_csv('sexism_predictions_task2.csv', index=False)
    print("Prediction TASK2 completed. Results saved to sexism_predictions_task2.csv")

def output_postprocessing_incontext_zero_s3(output):
    cLabels=['IDEOLOGICAL-INEQUALITY', 'STEREOTYPING-DOMINANCE', 'MISOGYNY-NON-SEXUAL-VIOLENCE', 'OBJECTIFICATION', 'SEXUAL-VIOLENCE']
    labels=[]
    outputp=output.rsplit("Output: ", 1)[-1].strip()
    #print(outputp)
    for line in outputp.split('\n'):
        line=line.strip()
        if line!="":
             #print(line)
             if line.upper().startswith("["):
                try:
                    labels = [x.upper() for  x in ast.literal_eval(line) if x.upper() in cLabels]
                    return labels
                except:
                    labels=[]
    return labels

def create_incontext_zero_prompt(task_description, query, context=None, role=None, output=None):
    prompt = f""
    if role!=None:
        prompt+= f"You are an expert {role}.\n\n"

    prompt += f"Task: {task_description}\n\n"
    if context!=None:
         prompt+= f"Context: {context}\n\n"
    if output != None:
         prompt+= f"Output Format: {output} \n\n"
    # Add query
    prompt+=f"Input: {query}\n"
    prompt+=f"Output: "
    return prompt



def incontext_zero_pipeline_task3(model, tokenizer, devData, testData, postprocessing, **params):

    role= "in social psychology and linguistics with vast experience analyzing social media content and discriminative and harmful language"
    task =""" Many facets of a woman's life may be the focus of sexist attitudes including domestic and parenting roles,  career opportunities, sexual image, and life expectations, to name a few. According to this, each sexist tweet must be assigned one or more of the five categories.

    IDEOLOGICAL-AND-INEQUALITY: it includes messages that discredit the feminist movement. It also includes messages that reject inequality between men and women, or present men as victims of gender-based oppression.

    STEREOTYPING-DOMINANCE: it includes messages that express false ideas about women that suggest they are more suitable or inappropriate for certain tasks, and somehow inferior to men.

    OBJECTIFICATION: it includes messages where women are presented as objects apart from their dignity and personal aspects. We also include messages that assume or describe certain physical ºqualities that women must have in order to fulfill traditional gender roles

    SEXUAL-VIOLENCE: it includes messages where sexual suggestions, requests or harassment of a sexual nature (rape or sexual assault) are made.

    MISOGYNY-NON-SEXUAL-VIOLENCE: it includes expressions of hatred and violence towards women."""

    output= "The output must be a list of the assigned label. Possible Labels: ['IDEOLOGICAL-INEQUALITY', 'STEREOTYPING-DOMINANCE', 'MISOGYNY-NON-SEXUAL-VIOLENCE', 'OBJECTIFICATION', 'SEXUAL-VIOLENCE']"
    context=None

    ntokens=params.get("max_new_tokens", 32)
    nseq=params.get("num_return_sequences", 1)
    temp=params.get("temperature", 0.7)

    idqueries= devData[0].tolist()
    textqueries= devData[1].tolist()
    labelqueries=devData[2].tolist()

    predictions=[]
    for i in range(len(textqueries)):
        query=textqueries[i]
        filledPrompt=create_incontext_zero_prompt(task, query, context=None, role=role, output=output)
        #print("PROMPT >>> ", filledPrompt)
        response= perform_incontext_classification(model, tokenizer, filledPrompt, ntokens, nseq, temp)
        #print("#"*25, "ANSWER", "#"*25)
        #print(response)
        pred=postprocessing(response)
        predictions.append(pred)
        print("ID: ", idqueries[i], "Ground Truth:", labelqueries[i], "Predicted: ", pred)

    # Create dev output DataFrame
    dev_df = pd.DataFrame({
        'id': idqueries,
        'label': predictions,
        "tweet": textqueries,
        "test_case": ["EXIST2025"]*len(predictions) })
    dev_df.to_csv('sexism_dev_predictions_task3.csv', index=False)
    print("Evaluation TASK3 completed. Results saved to sexism_dev_predictions_task3.csv")

    #Computing the quality measure for the subtask 1
    icm= ICMWrapper(predictions, labelqueries, multi=True)
    print(f"\nICM : {icm}\n\n")

    idqueries= testData[0].tolist()
    textqueries= testData[1].tolist()
    test_predictions=[]
    for i in range(len(textqueries)):
        query=textqueries[i]
        filledPrompt=create_incontext_zero_prompt(task, query, context=None, role=role, output=output)
        #print("PROMPT >>> ", filledPrompt)
        response= perform_incontext_classification(model, tokenizer, filledPrompt,  ntokens, nseq, temp)
        #print("#"*25, "ANSWER", "#"*25)
        #print(response)
        pred=postprocessing(response)
        test_predictions.append(pred)
        print("ID: ", idqueries[i], "Predicted: ", pred)

        # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': idqueries,
        'label': test_predictions,
        "test_case": ["EXIST2025"]*len(test_predictions) })
    submission_df.to_csv('sexism_predictions_task3.csv', index=False)
    print("Prediction TASK3 completed. Results saved to sexism_predictions_task3.csv")

def create_incontext_few_prompt(task_description, exemplars, query, context=None, role=None, output=None):
    prompt = f""
    if role!=None:
        prompt+= f"You are an expert {role}.\n\n"

    prompt += f"Task: {task_description}\n\n"
    if context!=None:
         prompt+= f"Context: {context}\n\n"

    prompt += "Examples in Input and Output format:\n\n"
    # Add exemplars
    for exemplar in exemplars:
        prompt += f"Input: {exemplar['input']}\n"
        prompt += f"Output: {exemplar['output']}\n\n"

    if output != None:
         prompt+= f"Output Format: {output} \n\n"
    # Add query
    prompt+=f"Input: {query}\nOutput:"
    return prompt

def output_postprocessing_incontext_few_s1(output):
    outputp=output.rsplit("\nOutput:", 1)[-1].strip()
    #print(outputp)
    for line in outputp.split('\n'):
        line=line.strip()
        if line!="":
             if line.upper().startswith("YES"): return "YES"
             if line.upper().startswith("NO"): return "NO"
    return "UNK"


def stratified_sample(df, label_column, sample_size=None, sample_frac=None):
    """
    Stratified sampling of a DataFrame based on a label column.

    Args:
        df (pd.DataFrame): The DataFrame to sample.
        label_column (str): The name of the column to stratify by.
        sample_size (int, optional): The desired sample size per stratum.
        sample_frac (float, optional): The fraction of samples to take per stratum.
            Either `sample_size` or `sample_frac` must be provided.

    Returns:
        pd.DataFrame: The stratified sample of the DataFrame.

    Raises:
        ValueError: If neither `sample_size` nor `sample_frac` is provided.
    """

    if sample_size is None and sample_frac is None:
        raise ValueError("Either `sample_size` or `sample_frac` must be provided.")
    # Group the DataFrame by the label column
    grouped = df.groupby(label_column)
    # Determine the sample size or fraction for each group
    if sample_size is not None:
        sample_sizes = {label: sample_size for label in grouped.groups}
    else:
        sample_fracs = {label: sample_frac for label in grouped.groups}
    # Sample each group
    samples = []
    for label, group in grouped:
        if sample_size is not None:
            sample = group.sample(n=sample_sizes[label], random_state=12324)
        else:
            sample = group.sample(frac=sample_fracs[label], random_state=12344)
        samples.append(sample)

    # Concatenate the samples
    stratified_sample = pd.concat(samples, ignore_index=True)
    return stratified_sample


#This function aims at selecting the examples used in the in-context learning
# using few-shot. This is a naive a papproach to inform the propmt with examples in the trainig dataset
def sampling_few_instances(trainData, nexamples):
    idqueries= trainData[0].tolist()
    textqueries= trainData[1].tolist()
    labelqueries=trainData[2].tolist()
    df = pd.DataFrame({'id': idqueries, 'tweet': textqueries, 'label': labelqueries})
    samples=stratified_sample(df, 'label', nexamples)
    examples=[]
    for i, row in samples.iterrows():
        examples.append({'input': row['tweet'], 'output': row['label']})
    return examples



def incontext_few_pipeline_task1(model, tokenizer, trainData, devData, testData, postprocessing,**params):

    role= "in social psychology and linguistics with vast experience analyzing social media content and discriminative and harmful language"
    task ="""Sexist identification task is a binary text classification task which aim at determining
    whether or not a given tweet expresses ideas related to sexism in any of the three forms: it is sexist itself,
    it describes a sexist situation in which discrimination towards women occurs, or criticizes a sexist behaviour.
    The tweet is sexist (YES) or describes or criticizes a sexist situation. Not sexist. The tweet is not sexist (NO),
    nor it describes or criticizes a sexist situation."""
    output= "The output must be YES/NO."
    context=None

    ntokens=params.get("max_new_tokens", 8)
    nseq=params.get("num_return_sequences", 1)
    temp=params.get("temperature", 0.7)

    idqueries= devData[0].tolist()
    textqueries= devData[1].tolist()
    labelqueries=devData[2].tolist()

    exemplars = sampling_few_instances(trainData, 3)

    predictions=[]
    for i in range(len(textqueries)):
        query=textqueries[i]
        filledPrompt=create_incontext_few_prompt(task, exemplars, query, context=None, role=role, output=output)
        #print("PROMPT >>> ", filledPrompt)
        response= perform_incontext_classification(model, tokenizer, filledPrompt, ntokens, nseq, temp)
        #print("#"*25, "ANSWER", "#"*25)
        #print(response)
        pred=postprocessing(response)
        if pred in ["YES", "NO"]:predictions.append(pred)
        else: predictions.append("NO")
        print("ID: ", idqueries[i], "Ground Truth:", labelqueries[i], "Predicted: ", pred)

    # Create dev output DataFrame
    dev_df = pd.DataFrame({'id': idqueries,  'label': predictions, "tweet": textqueries, "test_case": ["EXIST2025"]*len(predictions) })
    dev_df.to_csv('sexism_dev_predictions_task1_few.csv', index=False)
    print("Prediction TASK1 completed. Results saved to sexism_dev_predictions_task1_few.csv")

    #Computing the quality measure for the subtask 1
    f1= f1_score(labelqueries, predictions, labels=None, pos_label="YES", average='binary')
    print(f"\nF1-Score Sexism: {f1}\n\n")

    idqueries= testData[0].tolist()
    textqueries= testData[1].tolist()
    test_predictions=[]
    for i in range(len(textqueries)):
        query=textqueries[i]
        filledPrompt=create_incontext_few_prompt(task, exemplars, query, context=None, role=role, output=output)
        #print("PROMPT >>> ", filledPrompt)
        response= perform_incontext_classification(model, tokenizer, filledPrompt,  ntokens, nseq, temp)
        #print("#"*25, "ANSWER", "#"*25)
        #print(response)
        pred=postprocessing(response)
        if pred in ["YES", "NO"]:test_predictions.append(pred)
        else: test_predictions.append("NO")
        print("ID: ", idqueries[i], "Predicted: ", pred)

        # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': idqueries,
        'label': test_predictions,
        "test_case": ["EXIST2025"]*len(test_predictions) })
    submission_df.to_csv('/content/drive/MyDrive/EXISTS2025/results/sexism_predictions_task1_few.csv', index=False)
    print("Prediction TASK1 completed. Results saved to sexism_predictions_task1_few.csv")

def output_postprocessing_incontext_few_s2(output):
    outputp=output.rsplit("\nOutput:", 1)[-1].strip()
    for line in outputp.split('\n'):
        line=line.strip()
        if line!="":
             if line.upper().startswith("DIRECT"): return "DIRECT"
             if line.upper().startswith("REPORTED"): return "REPORTED"
             if line.upper().startswith("JUDGEMENTAL"): return "JUDGEMENTAL"
    return "UNK"


def stratified_sample(df, label_column, sample_size=None, sample_frac=None):
    """
    Stratified sampling of a DataFrame based on a label column.

    Args:
        df (pd.DataFrame): The DataFrame to sample.
        label_column (str): The name of the column to stratify by.
        sample_size (int, optional): The desired sample size per stratum.
        sample_frac (float, optional): The fraction of samples to take per stratum.
            Either `sample_size` or `sample_frac` must be provided.

    Returns:
        pd.DataFrame: The stratified sample of the DataFrame.

    Raises:
        ValueError: If neither `sample_size` nor `sample_frac` is provided.
    """

    if sample_size is None and sample_frac is None:
        raise ValueError("Either `sample_size` or `sample_frac` must be provided.")
    # Group the DataFrame by the label column
    grouped = df.groupby(label_column)
    # Determine the sample size or fraction for each group
    if sample_size is not None:
        sample_sizes = {label: sample_size for label in grouped.groups}
    else:
        sample_fracs = {label: sample_frac for label in grouped.groups}
    # Sample each group
    samples = []
    for label, group in grouped:
        if sample_size is not None:
            sample = group.sample(n=sample_sizes[label], random_state=1234)
        else:
            sample = group.sample(frac=sample_fracs[label], random_state=1234)
        samples.append(sample)

    # Concatenate the samples
    stratified_sample = pd.concat(samples, ignore_index=True)
    return stratified_sample


#This function aims at selecting the examples used in the in-context learning
# using few-shot. This is a naive a approach to inform the prompt with examples in the training dataset
def sampling_few_instances(trainData, nexamples):
    idqueries= trainData[0].tolist()
    textqueries= trainData[1].tolist()
    labelqueries=trainData[2].tolist()
    df = pd.DataFrame({'id': idqueries, 'tweet': textqueries, 'label': labelqueries})
    samples=stratified_sample(df, 'label', nexamples)
    examples=[]
    for i, row in samples.iterrows():
        examples.append({'input': row['tweet'], 'output': row['label']})
    return examples



def incontext_few_pipeline_task2(model, tokenizer, trainData, devData, testData, postprocessing, **params):
    role= "in social psychology and linguistics with vast experience analyzing social media content and discriminative and harmful language"
    task ="""Sexism Source Intention in tweets is a three class classification task aims to categorize
    the sexist tweets according to the intention of the author. This distinction allow us to differentiate
    sexism that is actually taking place online from sexism which is being suffered by women in other situations
    but that is being reported in social networks with the aim of complaining and fighting against sexism.
    The task include the following ternary classification of tweets.

    REPORTED: the intention is to report and share a sexist situation suffered by a woman or women.
    JUDGEMENTAL the intention is to condemn and critizise sexist situations or behaviours.
    DIRECT: The intention is to write a message that is sexist by itself or incites to be sexist.
     """
    output= "The output is a single label: REPORTED/JUDGEMENTAL/DIRECT."
    context=None

    ntokens=params.get("max_new_tokens", 8)
    nseq=params.get("num_return_sequences", 1)
    temp=params.get("temperature", 0.7)

    idqueries= devData[0].tolist()
    textqueries= devData[1].tolist()
    labelqueries=devData[2].tolist()

    exemplars = sampling_few_instances(trainData, 2)

    predictions=[]
    for i in range(len(textqueries)):
        query=textqueries[i]
        filledPrompt=create_incontext_few_prompt(task, exemplars, query, context=None, role=role, output=output)
        #print("PROMPT >>> ", filledPrompt)
        response= perform_incontext_classification(model, tokenizer, filledPrompt, ntokens, nseq, temp)
        #print("#"*25, "ANSWER", "#"*25)
        #print(response)
        pred=postprocessing(response)
        if pred in ["DIRECT", "REPORTED", "JUDGEMENTAL"]:predictions.append(pred)
        else: predictions.append("DIRECT")
        print("ID: ", idqueries[i],"Ground Truth:", labelqueries[i], "Predicted: ", pred)

    # Create dev output DataFrame
    dev_df = pd.DataFrame({'id': idqueries,  'label': predictions, "tweet": textqueries, "test_case": ["EXIST2025"]*len(predictions) })
    dev_df.to_csv('sexism_dev_predictions_task2_few.csv', index=False)
    print("Prediction TASK2 completed. Results saved to sexism_dev_predictions_task2_few.csv")

    #Computing the quality measure for the subtask 1
    f1= f1_score(labelqueries, predictions, average='macro')
    print(f"\nMacro average F1-Score Sexism: {f1}\n\n")

    idqueries= testData[0].tolist()
    textqueries= testData[1].tolist()
    test_predictions=[]
    for i in range(len(textqueries)):
        query=textqueries[i]
        filledPrompt=create_incontext_few_prompt(task, exemplars, query, context=None, role=role, output=output)
        #print("PROMPT >>> ", filledPrompt)
        response= perform_incontext_classification(model, tokenizer, filledPrompt,  ntokens, nseq, temp)
        #print("#"*25, "ANSWER", "#"*25)
        #print(response)
        pred=postprocessing(response)
        if pred in ["DIRECT", "REPORTED", "JUDGEMENTAL"]:test_predictions.append(pred)
        else: test_predictions.append("DIRECT")
        print("ID: ", idqueries[i], "Predicted: ", pred)

        # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': idqueries,
        'label': test_predictions,
        "test_case": ["EXIST2025"]*len(test_predictions) })
    submission_df.to_csv('sexism_predictions_task2_few.csv', index=False)
    print("Prediction TASK2 completed. Results saved to sexism_predictions_task2_few.csv")

def output_postprocessing_incontext_zero_s3(output):
    cLabels=['IDEOLOGICAL-INEQUALITY', 'STEREOTYPING-DOMINANCE', 'MISOGYNY-NON-SEXUAL-VIOLENCE', 'OBJECTIFICATION', 'SEXUAL-VIOLENCE']
    labels=[]
    outputp=output.rsplit("\nOutput:", 1)[-1].strip()
    #print(outputp)
    for line in outputp.split('\n'):
        line=line.strip()
        if line!="":
             #print(line)
             if line.upper().startswith("["):
                try:
                    labels = [x.upper() for  x in ast.literal_eval(line) if x.upper() in cLabels]
                    return labels
                except:
                    labels=[]
    return labels


#This function aims at selecting the examples used in the in-context learning
# using few-shot. This is a naive a papproach to inform the propmt with examples in the trainig dataset
def sampling_few_instances_multilabel(trainData, nexamples):
    idqueries= trainData[0].tolist()
    textqueries= trainData[1].tolist()
    labelqueries=trainData[2].tolist()
    df = pd.DataFrame({'id': idqueries, 'tweet': textqueries, 'label': labelqueries})
    samples= df.sample(n=min(nexamples, len(df)), random_state=1234)
    examples=[]
    for i, row in samples.iterrows():
        examples.append({'input': row['tweet'], 'output': row['label']})
    return examples



def incontext_few_pipeline_task3(model, tokenizer, trainData, devData, testData, postprocessing, **params):

    role= "in social psychology and linguistics with vast experience analyzing social media content and discriminative and harmful language"
    task =""" Many facets of a woman’s life may be the focus of sexist attitudes including domestic and parenting roles,  career opportunities, sexual image, and life expectations, to name a few. According to this, each sexist tweet must be assigned one or more of the five categories.

    IDEOLOGICAL-AND-INEQUALITY: it includes messages that discredit the feminist movement. It also includes messages that reject inequality between men and women, or present men as victims of gender-based oppression.

    STEREOTYPING-DOMINANCE: it includes messages that express false ideas about women that suggest they are more suitable or inappropriate for certain tasks, and somehow inferior to men.

    OBJECTIFICATION: it includes messages where women are presented as objects apart from their dignity and personal aspects. We also include messages that assume or describe certain physical ºqualities that women must have in order to fulfill traditional gender roles

    SEXUAL-VIOLENCE: it includes messages where sexual suggestions, requests or harassment of a sexual nature (rape or sexual assault) are made.

    MISOGYNY-NON-SEXUAL-VIOLENCE: it includes expressions of hatred and violence towards women."""

    output= "The output must be a list of the assigned label. Possible Labels: ['IDEOLOGICAL-INEQUALITY', 'STEREOTYPING-DOMINANCE', 'MISOGYNY-NON-SEXUAL-VIOLENCE', 'OBJECTIFICATION', 'SEXUAL-VIOLENCE']"
    context=None

    ntokens=params.get("max_new_tokens", 48)
    nseq=params.get("num_return_sequences", 1)
    temp=params.get("temperature", 0.4)

    idqueries= devData[0].tolist()
    textqueries= devData[1].tolist()
    labelqueries=devData[2].tolist()

    exemplars = sampling_few_instances_multilabel(trainData, 6)

    predictions=[]
    for i in range(len(textqueries)):
        query=textqueries[i]
        filledPrompt=create_incontext_few_prompt(task, exemplars, query, context=None, role=role, output=output)
        #print("PROMPT >>> ", filledPrompt)
        response= perform_incontext_classification(model, tokenizer, filledPrompt, ntokens, nseq, temp)
        #print("#"*25, "ANSWER", "#"*25)
        #print(response)
        pred=postprocessing(response)
        predictions.append(pred)
        print("ID: ", idqueries[i],"Ground Truth:", labelqueries[i], "Predicted: ", pred)

    # Create dev output DataFrame
    dev_df = pd.DataFrame({'id': idqueries,  'label': predictions, "tweet": textqueries, "test_case": ["EXIST2025"]*len(predictions) })
    dev_df.to_csv('sexism_dev_predictions_task3_few.csv', index=False)
    print("Prediction TASK3 completed. Results saved to sexism_dev_predictions_task3_few.csv")

    #Computing the quality measure for the subtask 3
    icm= ICMWrapper(predictions, labelqueries, multi=True)
    print(f"\nICM : {icm}\n\n")

    idqueries= testData[0].tolist()
    textqueries= testData[1].tolist()
    test_predictions=[]
    for i in range(len(textqueries)):
        query=textqueries[i]
        filledPrompt=create_incontext_few_prompt(task, exemplars, query, context=None, role=role, output=output)
        #print("PROMPT >>> ", filledPrompt)
        response= perform_incontext_classification(model, tokenizer, filledPrompt,  ntokens, nseq, temp)
        #print("#"*25, "ANSWER", "#"*25)
        #print(response)
        pred=postprocessing(response)
        test_predictions.append(pred)
        print("ID: ", idqueries[i],  "Predicted: ", pred)

    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': idqueries,
        'label': test_predictions,
        "test_case": ["EXIST2025"]*len(test_predictions) })
    submission_df.to_csv('sexism_predictions_task3_few.csv', index=False)
    print("Prediction TASK3 completed. Results saved to sexism_predictions_task3_few.csv")

 