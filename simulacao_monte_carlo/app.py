import streamlit as st
import os
import nltk
from nltk.tokenize import word_tokenize
import re
from collections import Counter
from scipy.stats import rv_discrete
import numpy as np
import random
from stqdm import stqdm


nltk.download('punkt')
nltk.download('punkt_tab')

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def clean_text(text):
    # Lowercase
    text = text.lower()

    # Remover pontuação
    text = re.sub('[^\s\w_]+', '', text)

    # Remover números
    text = re.sub(r'\d+', '', text)

    # Tokenize
    words = word_tokenize(text)

    return words

class NGramModel:
    def __init__(self, n=3, top_k=10):
        self.n = n
        self.top_k = top_k

    def generate_n_grams_freq(self, n, tokens):
        n_grams = []

        for i in stqdm(range(0, len(tokens) - n + 1), desc=f"Gerando {n}-grams"):
            n_grams.append(tuple((tokens[k] for k in range(i, i+n))))

        return n_grams

    def __fit_bigram_model(self, tokens_freq, bigrams_freq):

        self.bigram_model = {}

        for bigram, bigram_freq in stqdm(bigrams_freq.items(), desc="Treinando modelo de bigramas"):
            if not self.bigram_model.get(bigram[:1]):
                self.bigram_model[bigram[:1]] = {}
            self.bigram_model[bigram[:1]][bigram[1]] = bigram_freq/tokens_freq.get(bigram[0])

    def fit(self, tokens):

        tokens_freq = Counter(tokens)
        bigrams_freq = Counter(self.generate_n_grams_freq(2, tokens))
        self.__fit_bigram_model(tokens_freq, bigrams_freq)

        self.n_minus_1_grams_freq = Counter(self.generate_n_grams_freq(self.n - 1, tokens))
        self.n_grams_freq = Counter(self.generate_n_grams_freq(self.n, tokens))


        self.model = {}

        for n_gram, n_gram_freq in stqdm(self.n_grams_freq.items(), desc=f"Treinando modelo de {self.n}-grams"):
            if not self.model.get(n_gram[:self.n-1]):
                self.model[n_gram[:self.n-1]] = {}
            self.model[n_gram[:self.n-1]][n_gram[-1]] = n_gram_freq/self.n_minus_1_grams_freq.get(n_gram[:self.n-1])

    def generate_next_word_from_bigram_model(self, current_word):
        if not self.bigram_model.get((current_word,)):
            current_word = random.choice(list(self.bigram_model.keys()))[0]

        categories = list(self.bigram_model[(current_word,)].keys())

        custom_dist = rv_discrete(name='custom', values=(np.arange(len(categories)),
                                                         list(self.bigram_model[(current_word,)].values())))

        # Amostragem da distribuição
        num_samples = 1000
        samples = custom_dist.rvs(size=num_samples)
        samples = [categories[i] for i in samples]

        samples_counter = Counter(samples)

        top_10_samples = samples_counter.most_common(self.top_k)

        next_word = random.choice(top_10_samples)[0]
        return next_word

    def generate_initial_prompt(self, prompt):
        output_size = len(prompt)
        output = " ".join(prompt)
        prompt = (prompt[-1],)

        while output_size < self.n - 1:
            next_word = self.generate_next_word_from_bigram_model(prompt[0])
            output = output + " " + next_word
            output_size += 1
            prompt = (output.split()[-1], )
        return output

    def generate_text(self, prompt, output_size=10):
        prompt = clean_text(prompt)
        output = " ".join(prompt)
        prompt = tuple(prompt)

        if len(prompt) == 0:
            print("prompt cannot be empty!")
            return None

        if len(prompt) < self.n - 1:
            prompt = self.generate_initial_prompt(prompt)
            output = prompt
            prompt = tuple(prompt.split())

        while output_size > 0:
            if not self.model.get(prompt):
                next_word = self.generate_next_word_from_bigram_model(prompt[-1])
            else:
                categories = list(self.model[prompt].keys())

                custom_dist = rv_discrete(name='custom', values=(np.arange(len(categories)), list(self.model[prompt].values())))

                # Amostragem da distribuição
                num_samples = 1000
                samples = custom_dist.rvs(size=num_samples)
                samples = [categories[i] for i in samples]

                samples_counter = Counter(samples)
                top_k_samples = samples_counter.most_common(self.top_k)

                next_word = random.choice(top_k_samples)[0]

            output = output + " " + next_word
            output_size -= 1
            prompt = tuple(output.split()[-(self.n-1):])

        return output

current_dir = os.path.dirname(__file__)

path = os.path.join(current_dir, 'data')
 

data_files = os.listdir(path)
st.sidebar.header("Configurações")
st.sidebar.subheader("Arquivos para treinamento do modelo")
selected_files = []
for file_ in data_files:
    if st.sidebar.checkbox(f"{file_.replace('_', ' ').replace('.txt', '')}", value=False):
        selected_files.append(file_)

st.title("N-Gram Model Dashboard")

n_value = st.sidebar.slider("Valor de n (N-gram):", min_value=2, max_value=5, value=3)
train_model = st.sidebar.button("Treinar modelo")

ngram_model = NGramModel(n=n_value)

if "model_is_trained" not in st.session_state:
    st.session_state.model_is_trained = False

if "ngram_model" not in st.session_state:
    st.session_state.ngram_model = None

if train_model:

    with st.spinner(f"Treinando modelo com {n_value}-grams..."):
        tokens = []
        for file_ in stqdm(selected_files, desc="Executando tokenizer..."):
            with open(os.path.join(path, file_), 'r') as f:
                text = f.read()
                tokens += clean_text(text)

        ngram_model.fit(tokens)
        st.session_state.ngram_model = ngram_model
        st.session_state.model_is_trained = True
        st.success("Modelo treinado com sucesso!")

if st.session_state.model_is_trained:
    st.sidebar.write("### Configurações aplicadas")
    st.sidebar.write(f"Arquivos selecionados: {', '.join(selected_files)}")
    st.sidebar.write(f"Valor de N-gram (n): {n_value}")

    top_k = st.slider("Valor de k (top-K tokens):", min_value=1, max_value=10, value=5)
    st.session_state.ngram_model.top_k = top_k

    user_input = st.text_input("Entre com um texto a ser completado:", "the")
    output_size = st.number_input("Tamanho da saída", min_value=10, max_value=30, value=20)

    if st.button("Gerar saída"):
        generated_text = st.session_state.ngram_model.generate_text(user_input, output_size)
        st.write("Texto gerado:")
        st.write(generated_text)
else:
    st.info("Por favor, treine o modelo para habilitar a geração de texto.")
