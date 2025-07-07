# 必要なモジュールのインストール
# 基本形
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# transformers系
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

# transfomersを使って学習訓練を実行
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer

# from datasets import load_metric
import evaluate
from datasets import Dataset

# データセットのURL
url = "https://github.com/ids-cv/wrime/raw/master/wrime-ver2.tsv"

# ダウンロードしたファイルの保存先
file_name = "wrime-ver2.tsv"

# HTTP GETリクエストを送信してファイルをダウンロード
response = requests.get(url)

# ファイルを保存
with open(file_name, 'wb') as file:
    file.write(response.content)

print(f"{file_name} downloaded successfully.")

# データフレーム型で保存
df_wrime = pd.read_table('wrime-ver2.tsv')
print(df_wrime.info())

# Plutchikの8つの基本感情
# （喜び・悲しみ・期待・驚き・怒り・恐れ・嫌悪・信頼）
emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger','Fear','Disgust','Trust']
num_labels = len(emotion_names)

# 客観感情の平均（"Avg. Readers_*"） の値をlist化し、新しい列として定義する
df_wrime['readers_emotion_intensities'] = df_wrime.apply(lambda x: [x['Avg. Readers_' + name] for name in emotion_names], axis=1)

# 感情強度が低いサンプルは除外する
is_target = df_wrime['readers_emotion_intensities'].map(lambda x: max(x) >= 2)
df_wrime_target = df_wrime[is_target]

# train / test に分割する
df_groups = df_wrime_target.groupby('Train/Dev/Test')
df_train = df_groups.get_group('train')
df_test = pd.concat([df_groups.get_group('dev'), df_groups.get_group('test')])
print('train :', len(df_train))  # train : 12662
print('test :', len(df_test))    # test : 2102

# 使用するモデルを指定して、トークナイザとモデルを読み込む
checkpoint = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=8)

# 1. Transformers用のデータセット形式に変換
target_columns = ['Sentence', 'readers_emotion_intensities']
train_dataset = Dataset.from_pandas(pd.DataFrame(df_train[target_columns]))
test_dataset = Dataset.from_pandas(pd.DataFrame(df_test[target_columns]))

# 2. Tokenizerを適用（モデル入力のための前処理）
def tokenize_function(batch):
    """Tokenizerを適用 （感情強度の正規化も同時に実施する）."""
    tokenized_batch = tokenizer(batch['Sentence'], truncation=True, padding='max_length')
    tokenized_batch['labels'] = [x / np.sum(x) for x in batch['readers_emotion_intensities']] # 総和=1に正規化
    return tokenized_batch

# 3. map関数で前処理（tokenize_function）を適用
train_tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
test_tokenized_dataset = test_dataset.map(tokenize_function, batched=True)

# メトリクスのロード
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    label_ids = np.argmax(labels, axis=-1)
    result = metric.compute(predictions=predictions, references=label_ids)
    if result is None:
        return {}
    return result

# モデル内の全てのテンソルを連続化する関数
def make_model_contiguous(model):
    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

# Trainerの実行前にこの関数を呼び出して、モデルのすべてのテンソルを連続化
make_model_contiguous(model)

# 訓練時の設定
training_args = TrainingArguments(
    output_dir="test_trainer",
    per_device_train_batch_size=8, # バッチサイズ　大きいと処理速度が上がるがGPU次第 標準は8
    num_train_epochs=1.0,
    fp16=True,
    report_to="none"  # ← wandb などの外部サービスへの出力を抑制
)

# Trainerの生成
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=test_tokenized_dataset,
    compute_metrics=compute_metrics,
)

# モデルの全パラメータを連続化してから訓練を実行
make_model_contiguous(trainer.model)
trainer.train()

# モデルの保存時にもテンソルを連続化
make_model_contiguous(trainer.model)
trainer.save_model("path_to_save_model")

# モデルとトークナイザを読み込み
model = AutoModelForSequenceClassification.from_pretrained(trainer.model)
tokenizer = AutoTokenizer.from_pretrained(trainer.model)

# extra_info.pkl を読み込み
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
