import os
import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import spacy
import japanize_matplotlib
import requests
import io

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login

# HFトークンは環境変数から取得し、あればログイン
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

model_repo = "yusuke-31/streamlit_deploy1"  # ご自身のリポジトリに書き換え

@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer():
    model = AutoModelForSequenceClassification.from_pretrained(model_repo)
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    return model, tokenizer

@st.cache_resource(show_spinner=True)
def load_extra_info():
    extra_info_url = f"https://huggingface.co/{model_repo}/resolve/main/extra_info.pkl"
    response = requests.get(extra_info_url)
    response.raise_for_status()
    return pickle.load(io.BytesIO(response.content))

@st.cache_resource(show_spinner=True)
def load_nlp():
    return spacy.load("ja_ginza")

model, tokenizer = load_model_and_tokenizer()
extra_info = load_extra_info()
emotion_names = extra_info["emotion_names"]
nlp = load_nlp()

def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def analyze_emotion(text):
    model.eval()
    tokens = tokenizer(text, truncation=True, return_tensors="pt")
    tokens = {k: v.to(model.device) for k, v in tokens.items()}
    with torch.no_grad():
        preds = model(**tokens)
    prob = np_softmax(preds.logits.cpu().numpy()[0])
    return {n: p for n, p in zip(emotion_names, prob)}

def split_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

st.title("日本語文＆感情分析")

user_input = st.text_area(
    "日本語の文章を入力してください",
    "今のあなたの気持ちはどのようなものでしょう。調査してみましょう。",
    height=200
)

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

            mean_emotions = df_result[emotion_names].mean().reset_index()
            mean_emotions.columns = ['感情', '平均確率']

            st.write("### 全体の平均感情スコア")
            plt.figure(figsize=(8, 3))
            sns.barplot(x='感情', y='平均確率', data=mean_emotions)
            st.pyplot(plt.gcf())
