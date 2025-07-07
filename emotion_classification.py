import os
import zipfile
import pickle
import requests

# モデルが解凍されていなければ saved_model.zip を展開
if not os.path.exists("saved_model"):
    with zipfile.ZipFile("saved_model.zip", "r") as zip_ref:
        zip_ref.extractall("saved_model")

# transformersから読み込み
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "saved_model",
    local_files_only=True,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("saved_model", local_files_only=True)

# extra_info.pkl の読み込み
with open("saved_model/extra_info.pkl", "rb") as f:
    extra_info = pickle.load(f)

emotion_names = extra_info["emotion_names"]
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import re
import spacy
import japanize_matplotlib

from transformers import BertJapaneseTokenizer, BertForSequenceClassification

# Softmax関数
def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

# 感情分析関数
def analyze_emotion(text):
    model.eval()
    tokens = tokenizer(text, truncation=True, return_tensors="pt")
    tokens.to(model.device)
    preds = model(**tokens)
    prob = np_softmax(preds.logits.cpu().detach().numpy()[0])
    return {n: p for n, p in zip(emotion_names, prob)}

# GiNZAモデルの読み込み
@st.cache_resource
def load_nlp():
    return spacy.load("ja_ginza")

nlp = load_nlp()

# 文分割関数（spaCy + GiNZA）
def split_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

st.title("日本語文＆感情分析")

user_input = st.text_area("日本語の文章を入力してください","今のあなたの気持ちはどのようなものでしょう。調査してみましょう。", height=200)

if st.button("文を分析する"):
    if not user_input.strip():
        st.warning("入力がありません・・・")
    else:
        paragraphs = [p.strip() for p in user_input.split('\n') if p.strip()]
        all_sentences = []
        for para in paragraphs:
            all_sentences.extend(split_sentences(para))

        if not all_sentences:
            st.warning("文が見つかりませんでした。改行または句読点のある文を入力してください。")
        else:
            all_results = []
            for idx, sentence in enumerate(all_sentences):
                result = analyze_emotion(sentence)
                result['文'] = sentence
                result['ID'] = idx + 1
                all_results.append(result)

            df_result = pd.DataFrame(all_results)
            df_result = df_result[['ID', '文'] + emotion_names]

            st.success(f"{len(df_result)} 文を検出しました。")
            st.dataframe(df_result, use_container_width=True)

        mean_emotions = df_result[emotion_names].mean()
        if isinstance(mean_emotions, pd.Series):
            mean_emotions = mean_emotions.reset_index()
            mean_emotions.columns = ['感情', '平均確率']
        else:
            # 1列だけの場合
            mean_emotions = pd.DataFrame({'感情': [emotion_names[0]], '平均確率': [mean_emotions]})

        st.write("### 全体の平均感情スコア")
        plt.figure(figsize=(8, 3))
        sns.barplot(x='感情', y='平均確率', data=mean_emotions)
        st.pyplot(plt.gcf())
