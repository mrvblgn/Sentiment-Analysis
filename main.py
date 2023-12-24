import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
import numpy as np
import requests
from bs4 import BeautifulSoup
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from keras.utils import plot_model

# Veri setini yükleme
data = pd.read_csv('magaza_yorumlari_duygu_analizi.csv', encoding='utf-16')

# Veri setindeki eksik değerleri kontrol etme ve doldurma
data["Görüş"].fillna("", inplace=True)

# Veri setini eğitim ve test verisi olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(data["Görüş"], data["Durum"], test_size=0.2, random_state=42)

# Etiketleri sayısallaştırma
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# Metin verilerini kelimelere ayırma
X_train_words = [text.split() for text in X_train]
X_test_words = [text.split() for text in X_test]

# Word2Vec modelini eğitme
word2vec = Word2Vec(X_train_words, vector_size=100, window=5, min_count=1, workers=4)

# Her bir belgeyi, belgedeki kelimelerin vektörlerinin ortalaması olarak temsil eden bir vektörle dönüştürme
X_train_word2vec = np.array([np.mean([word2vec.wv[word] for word in words if word in word2vec.wv.index_to_key], axis=0) if np.sum([word in word2vec.wv.index_to_key for word in words]) > 0 else np.zeros(100) for words in X_train_words])
X_test_word2vec = np.array([np.mean([word2vec.wv[word] for word in words if word in word2vec.wv.index_to_key], axis=0) if np.sum([word in word2vec.wv.index_to_key for word in words]) > 0 else np.zeros(100) for words in X_test_words])

# Modeli oluşturma ve eğitme
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_word2vec, y_train, epochs=10, batch_size=32)

# Tahminleri yapma ve modelin doğruluğunu hesaplama
y_pred = model.predict(X_test_word2vec)
y_pred = [np.argmax(prediction) for prediction in y_pred]
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")


header = {
"user-agent":"Mozilla/5.0(Macintosh;IntelMacOSX10_15_7)AppleWebKit/605.1.15(KHTML,likeGecko)Version/17.1.2Safari/605.1.15"
}
comments = []
for i in range(2, 10):
    url = f"https://izleryazar.com/category/sinema/film-yorumlari/page/{i}/"
    response = requests.get(url, headers=header)
    soup = BeautifulSoup(response.text, 'html.parser')
    data = soup.find_all('div', {'class': 'cb-excerpt'})
    for div in data:
        comments.append(div.text)

# Yorumları ve karşılık gelen tahminleri saklamak için liste
yorum_metinleri = []
yorum_tahminleri = []


# Tahmin etiketlerini oluşturma
tahmin_etiketleri = ['olumsuz', 'tarafsız', 'olumlu']

# Yorumları model ile tahmin etme
for comment in comments:
    comment_words = comment.split()
    comment_word2vec = np.mean([word2vec.wv[word] for word in comment_words if word in word2vec.wv.index_to_key], axis=0)
    prediction = model.predict(np.array([comment_word2vec]))
    yorum_metinleri.append(comment)
    yorum_tahminleri.append(tahmin_etiketleri[np.argmax(prediction)])


df = pd.DataFrame({
    'Yorum': yorum_metinleri,
    'Tahmin': yorum_tahminleri
})

df.to_csv('yorum_tahminleri.csv', index=False)

# Modeli görselleştirme
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)