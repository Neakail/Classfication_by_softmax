#coding: utf-8
import os
import json
import pickle
from collections import defaultdict
from sklearn.svm import SVC
from sklearn import svm
import nltk
import nltk.data
from nltk.tokenize import WordPunctTokenizer
import re
import random
import numpy as np
import math

'''
制作印尼分类器算法:
1)做词典列表(210个词)
2)打开每篇文章
3)构建特征向量
4)构建目标向量
5)构建模型
6)测试
'''


class Softmax(object):

    def __init__(self):
        self.learning_step = 0.000001           # 学习速率
        self.max_iteration = 100000             # 最大迭代次数
        self.weight_lambda = 0.01               # 衰退权重

    def cal_e(self,x,l):

        theta_l = self.w[l]
        product = np.dot(theta_l,x)

        return math.exp(product)

    def cal_probability(self,x,j):


        molecule = self.cal_e(x,j)
        denominator = sum([self.cal_e(x,i) for i in range(self.k)])

        return molecule/denominator


    def cal_partial_derivative(self,x,y,j):


        first = int(y==j)                           # 计算示性函数
        second = self.cal_probability(x,j)          # 计算后面那个概率

        return -x*(first-second) + self.weight_lambda*self.w[j]

    def predict_(self, x):
        result = np.dot(self.w,x)
        row, column = result.shape

        # 找最大值所在的列
        _positon = np.argmax(result)
        m, n = divmod(_positon, column)

        return m

    def train(self, features, labels):
        self.k = len(set(labels))

        self.w = np.zeros((self.k,len(features[0])+1))
        time = 0

        while time < self.max_iteration:
            print('loop %d' % time)
            time += 1
            index = random.randint(0, len(labels) - 1)

            x = features[index]
            y = labels[index]

            x = list(x)
            x.append(1.0)
            x = np.array(x)

            derivatives = [self.cal_partial_derivative(x,y,j) for j in range(self.k)]

            for j in range(self.k):
                self.w[j] -= self.learning_step * derivatives[j]

    def predict(self,features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)

            x = np.matrix(x)
            x = np.transpose(x)

            labels.append(self.predict_(x))
        return labels


#定义一个对印尼语的分词函数
def SpiltWords(paragraph):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tokenizer.tokenize(paragraph)
        return sentences

def wordtokenizer(sentence):
        words = WordPunctTokenizer().tokenize(sentence)
        return words


#制作一个印尼语词典列表
with open('dictionary') as f:
    lines = [line.strip().decode('ascii') for line in f.readlines()]


#对每篇文章构建特征向量
CharaterList=[]
TagList=[]
for i in range(1,4):  # json1-4用来做训练集
        # print "tag%d.json" % i
        f = open("corpus/ind-corpus.part%d.json" % i)
        respond = json.load(f)  # 读入每个json文件
        for j in respond: # j是每条新闻
            temp = [0] * 241
            for sentences in SpiltWords(j[u"body"]):
                for word in wordtokenizer(sentences):
                    if word in lines:
                        temp[lines.index(word)] += 1
            for sentences in SpiltWords(j[u"abstract"]):
                for word in wordtokenizer(sentences):
                    if word in lines:
                        temp[lines.index(word)] += 1
            for sentences in SpiltWords(j[u"title"]):
                for word in wordtokenizer(sentences):
                    if word in lines:
                        temp[lines.index(word)] += 1
            CharaterList.append(temp)
            TagList.append (j[u"tag"])
        f.close()

for i in CharaterList:
    print i
# # 将训练集的特征向量和目标向量存成pickle文件
# with open("CharaterList",'w')as f1:
#         f1.write(pickle.dumps(CharaterList))
# with open("TagList",'w')as f2:
#         f2.write(pickle.dumps(TagList))

#训练模型
clf=svm.SVC()
model=clf.fit(CharaterList,TagList)
# p = Softmax()
# p.train(CharaterList,TagList)

#对测试集做预处理
CharaterList1=[]
TagList1=[]
with open("corpus/ind-corpus.part5.json","r") as f1:
        respond1 = json.load(f1)
        for j in respond1:#j是每条新闻
            temp = [0] * 241
            for sentences in SpiltWords(j[u"body"]):
                for word in wordtokenizer(sentences):
                    if word in lines:
                        temp[lines.index(word)] += 1
            for sentences in SpiltWords(j[u"abstract"]):
                for word in wordtokenizer(sentences):
                    if word in lines:
                        temp[lines.index(word)] += 1
            for sentences in SpiltWords(j[u"title"]):
                for word in wordtokenizer(sentences):
                    if word in lines:
                        temp[lines.index(word)] += 1
            CharaterList1.append(temp)
            TagList1.append(j[u"tag"])
        f.close()

#预测与检验
result1 = model.predict(CharaterList1)
n = 0
for i in range(len(TagList1)):
        if result1[i] == TagList1[i]:
                n = n + 1

rate = n / (len(TagList) * 1.0)
print rate




