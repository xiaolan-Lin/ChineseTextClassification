# 1. 导入数据
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv('./data/public_comment_review_data.csv', encoding='utf-8', low_memory=False)
len(data)  # 原始数据量:467455
data.dtypes  # 查看原始数据列值类型
# 2. 数据预处理
# （1）评分相加取平均
data = data[~(data['Score_taste'].isin(['|']) | data['Score_environment'].isin(['|']) | data['Score_service'].isin(
    ['|']) | data['Score_taste'].isin(['场']) | data['Score_taste'].isin(['产']) | data['Score_taste'].isin(
    ['房']))]  # 去掉三个评分列中含有非数值的记录
data[['Score_taste', 'Score_environment', 'Score_service']] = data[
    ['Score_taste', 'Score_environment', 'Score_service']].apply(pd.to_numeric)  # 将三列评分由str类型转换为int类型
data['score'] = data.apply(lambda x: int(round((x[2] + x[3] + x[4] + x[5]) / 4, 0)), axis=1)  # 添加评分均值列
# （2）删除多余的列
data = data.drop(
    ['Review_ID', 'Merchant', 'Rating', 'Score_taste', 'Score_environment', 'Score_service', 'Price_per_person', 'Time',
     'Num_thumbs_up', 'Num_ response', 'Reviewer', 'Reviewer_value', 'Reviewer_rank', 'Favorite_foods'],
    axis=1)  # 删除多余的列，剩下评论内容、评分均值两列
# （3）打标签——情绪
data['score'].unique()  # 查看评分有多少种（0~4，5种），此处直接用于标签分类
# # SNOWNLP——利用中文分类库SnowNLP对情绪进行评估
# from snownlp import SnowNLP
#
#
# def snow_result(Content_review):
#     s = SnowNLP(Content_review)
#     if s.sentiments < 0.2:
#         return 0
#     elif (s.sentiments >= 0.2) & (s.sentiments < 0.3):
#         return 1
#     elif (s.sentiments >= 0.3) & (s.sentiments < 0.7):
#         return 2
#     elif (s.sentiments >= 0.7) & (s.sentiments < 0.8):
#         return 3
#     elif (s.sentiments >= 0.8) & (s.sentiments <= 1.0):
#         return 4
#
#
# data['Content_review'] = data['Content_review'].astype(str)  # 将Content_review列种参杂了其他类型的列值全部转换为str类型，才可进行下一步操作
# data['snlp_result'] = data.Content_review.apply(snow_result)
# # 评价分均值与调库出来情绪的得分比较后的准确率
# counts = 0
# for i in range(len(data)):
#     if data.iloc[i, 2] == data.iloc[i, 3]:
#         counts += 1
#
# print(counts / len(data))  # 0.41632196255582493

# jieba分词
import jieba


# 增加分词列cut_comment


def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))


data['cut_comment'] = data.Content_review.astype(str).apply(chinese_word_cut)

# 将Content_review列存放在列表中
cut_list = []
for index in data.index:
    cut_list.append(data.loc[index].astype(str).values[0])


# 加载停用词表


def get_custom_stopwords(stop_words_file):
    with open(stop_words_file, encoding='utf8') as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list


stop_words_file = '哈工大停用词表.txt'
stopwords = get_custom_stopwords(stop_words_file)

# 去掉停词，存储于列表list
sentences_cut = []
for ele in cut_list:
    cuts = jieba.cut(ele, cut_all=False)
    new_cuts = []
    for cut in cuts:
        if cut not in stopwords:
            new_cuts.append(cut)
    res = ' '.join(new_cuts)
    sentences_cut.append(res)
# print(sentences_cut)

# 分词后的文本保存在filter_data.txt中
with open('filter_data.txt', 'w', encoding='utf8') as f:
    for ele in sentences_cut:
        ele = ele + '\n'
        f.write(ele)

# 词向量
from gensim.models import word2vec
import time

start = time.process_time()
sentences = word2vec.LineSentence('filter_data.txt')
model = word2vec.Word2Vec(sentences, size=300, workers=6, sg=1)
end = time.process_time()
print('Running time: %s Seconds' % (end - start))
# 模型保存加载方式
# 方法一
model.save('word2vec.model')
w2v_model = word2vec.Word2Vec.load('word2vec.model')
# 方法二（可直接通过txt打开可视，占用内存少，加载时间长）
model.wv.save_word2vec_format('word2vec.vector')
# t1 = time.time()
# model = word2vec.Word2Vec.load('word2vec.vector')
# t2 = time.time()
print(model)
# print('.model load time %.4f')
# model.wv.save_word2vec_format('word2vec.bin')

# 测试词向量模型
# y2 = model.wv.similarity(u"棒", u"好")
# print(y2)
#
# for i in model.wv.most_similar(u"酒吧"):
#     print(i[0], i[1])

# 基于Keras深度学习框架
import logging
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
# 隐含层所需要用到的函数，其中Convolution2D是卷积层；Activation是激活函数；MaxPooling2D作为池化层；
# Flatten是起到将多维输入易卫华的函数；Dense是全连接层
from tensorflow.keras.layers import MaxPooling1D, Flatten, Dense, Input, Dropout, Embedding, Conv1D
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import concatenate

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
tokenizer = Tokenizer()  # 创建一个Tokenizer对象
tokenizer.fit_on_texts(data['cut_comment'])  # fit_on_texts函数可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小
vocab = tokenizer.word_index

# 划分数据集
X = data['cut_comment']
y = data['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# 将每个样本中的每个词转换为数字列表，使用每个词的编号进行编号
x_train_word_ids = tokenizer.texts_to_sequences(X_train)
x_test_word_ids = tokenizer.texts_to_sequences(X_test)
# 序列模式
# 每条样本长度不唯一，将每条样本的长度设置一个固定值
x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=300)  # 将超过固定值的部分截掉，不足的在最前面用0填充,(373847, 300)
x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=300)

# 预训练的词向量中没有出现的词用0向量表示
embedding_matrix = np.zeros((len(vocab) + 1, 300), dtype='float32')
for word, i in vocab.items():
    try:
        embedding_vector = w2v_model[str(word)]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        continue


# 构建Text-CNN模型
# 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
main_input = Input(shape=(300,), dtype='float32')
# 词嵌入（使用预训练的词向量）
embedder = Embedding(len(vocab) + 1, 300, input_length=100, weights=[embedding_matrix], trainable=False)
embed = embedder(main_input)
# 词窗大小分别为3,4,5
cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
cnn1 = MaxPooling1D(pool_size=38)(cnn1)
cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
cnn2 = MaxPooling1D(pool_size=37)(cnn2)
cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
cnn3 = MaxPooling1D(pool_size=36)(cnn3)
# 合并三个模型的输出向量
cnn = concatenate([cnn1, cnn2, cnn3], axis=-2)
flat = Flatten()(cnn)
drop = Dropout(0.2)(flat)
main_output = Dense(5, activation='softmax')(drop)  # Dense此处units参数必须是标签数
model = Model(inputs=main_input, outputs=main_output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

one_hot_labels = tf.keras.utils.to_categorical(y_train, num_classes=5)  # 将标签转换为one-hot编码
model.fit(x_train_padded_seqs, one_hot_labels, batch_size=900, epochs=5)
# y_test_onehot = keras.utils.to_categorical(y_test, num_classes=3)  # 将标签转换为one-hot编码
result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
# y_predict = list(map(str, result_labels))
y_predict = list(result_labels)
y_test = list(y_test)
print('准确率', accuracy_score(y_test, y_predict))
print('平均f1-score:', f1_score(y_test, y_predict, average='weighted'))

# LSTM 模型
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping


# 定义LSTM模型
lstm_inputs = Input(name='inputs', shape=[400])
# Embedding(词汇表大小, batch大小,每条评论的词长)
layer = Embedding(len(vocab) + 1, 100, input_length=400)(lstm_inputs)
layer = LSTM(100)(layer)
layer = Dense(100, activation='relu', name='FC1')(layer)
layer = Dropout(0.5)(layer)
layer = Dense(2, activation='softmax', name='FC2')(layer)
lstm_model = Model(inputs=lstm_inputs, outputs=layer)
lstm_model.summary()
lstm_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
lstm_model_fit = lstm_model.fit(x_train_padded_seqs, one_hot_labels, batch_size=800, epochs=10,
                                callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])
test_pre = lstm_model.predict(x_test_padded_seqs)
confm = accuracy_score(list(np.argmax(test_pre, axis=1)), y_test)
print('================LSTM算法=================')
print("LSTM模型准确率:", confm)


# 机器学习
vect = CountVectorizer(max_df=0.8, min_df=3, token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b', stop_words=frozenset(stopwords))
test = pd.DataFrame(vect.fit_transform(X_train).toarray(), columns=vect.get_feature_names())
test.head()

# import os
#
# os.environ["PATH"] += os.pathsep + 'D:\\Program Files (x86)\\Graphviz2.38\\bin'
# 绘制模型图
from tensorflow.keras.utils import plot_model

# vect = CountVectorizer(max_df=0.8, min_df=3, token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b', stop_words=frozenset(stopwords))
# # test = pd.DataFrame(vect.fit_transform(X_train).toarray(), columns=vect.get_feature_names())
# # test.head()
#
# # 朴素贝叶斯
# # 训练模型
# from sklearn.naive_bayes import MultinomialNB
# nb = MultinomialNB()
#
# X_train_vect = vect.fit_transform(X_train)
# nb.fit(X_train_vect, y_train)
# train_score = nb.score(X_train_vect, y_train)
# print(train_score)  # 0.6476339251084
# # 测试模型
# X_test_vect = vect.transform(X_test)
# print(nb.score(X_test_vect, y_test))  # 0.6044167683122552


# text1 = '这个东西很不错'
# text2 = '这个东西很垃圾'
#
# s1 = SnowNLP(text1)
# s2 = SnowNLP(text2)
#
# print(s1.sentiments, s2.sentiments)



