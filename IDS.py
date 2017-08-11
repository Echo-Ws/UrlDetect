
# coding: utf-8

# In[1]:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.cluster import KMeans
import numpy as np

import pickle
import urllib
import time


# 各十条数据作为例子分析
# 输入参数部分
good = 'data/good_fromE.txt'

bad = 'data/badqueries.txt'

k = 80
# ngram 系数
n = 2

# wether use kmean
# use_k = False
use_k = True




def printT(word):
    a = time.strftime('%Y-%m-%d %H:%M:%S: ', time.localtime(time.time()))
    print(a+str(word))


# return[good,bad]
def getdata():

    with open(good,'r') as f:
        good_query_list = [i.strip('\n') for i in f.readlines()[:]]
    with open(bad,'r') as f:
        bad_query_list = [i.strip('\n') for i in f.readlines()[:]]
    return [good_query_list, bad_query_list]


class IDS(object):
    pass


# In[8]:

#     训练模型基类
class Baseframe(object):

    def __init__(self):
        pass

    def getname(self):
        return 'baseframe'
    
    def Train(self):

        printT('读入数据，good：'+good+' bad:'+bad)
        data = getdata()
        printT('done, good numbers:'+str(len(data[0]))+' bad numbers:'+str(len(data[1])))
        # 打标记
        good_y = [0 for i in range(len(data[0]))]
        bad_y = [1 for i in range(len(data[1]))]
        
        y = good_y + bad_y

        #     向量化
        # converting data to vectors  定义矢量化实例
        self.vectorizer = TfidfVectorizer(tokenizer=self.get_ngrams)
        # 把不规律的文本字符串列表转换成规律的 ( [i,j],weight) 的矩阵X [url条数，分词总类的总数，理论上少于256^n] 
        # i表示第几条url，j对应于term编号（或者说是词片编号）
        # 用于下一步训练分类器 lgs
        X = self.vectorizer.fit_transform(data[0]+data[1])
        printT('向量化后维度：'+str(X.shape))
        # 通过kmeans降维 返回降维后的矩阵
        if use_k:
            X = self.transform(self.kmeans(X))

            printT('降维完成')

        printT('划分测试集训练集')
        # 使用 train_test_split 分割 X y 列表 testsize表示测试占的比例 random为种子
        # X_train矩阵的数目对应 y_train列表的数目(一一对应)  -->> 用来训练模型
        # X_test矩阵的数目对应 	 (一一对应) -->> 用来测试模型的准确性
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        printT('划分完成，开始训练分类器')
        printT(self.classifier)
        self.classifier.fit(X_train, y_train)

        # 使用测试值 对 模型的准确度进行计算
        printT(self.getname()+'模型的准确度:{}'.format(self.classifier.score(X_test, y_test)))
        
        #         保存训练结果
        with open('model/'+self.getname()+'.pickle', 'wb') as output:
            pickle.dump(self, output)

    # 对新的请求列表进行预测
    def predict(self, new_queries):
        try:
            with open('model/'+self.getname()+'.pickle', 'rb') as input:
                self = pickle.load(input)
            printT('loading '+self.getname()+'model success')
        except FileNotFoundError:
            printT('start to train the '+self.getname()+' model')
            self.Train()
        printT('start predict:')
        #         解码
        new_queries = [urllib.parse.unquote(url) for url in new_queries]
        X_predict = self.vectorizer.transform(new_queries)
        if use_k:
            printT('将输入转换')
            X_predict = self.transform(X_predict.tolil().transpose())

        printT('转换完成,开始预测')
        res = self.classifier.predict(X_predict)
        printT('预测完成,总数：' + str(len(res)))
        result = {}

        result[0] = []
        result[1] = []
        
        #         两个列表并入一个元组列表
        for q, r in zip(new_queries, res):
            result[r].append(q)

        printT('good query: '+str(len(result[0])))
        printT('bad query: '+str(len(result[1])))
        # printT("预测的结果列表:{}".format(str(result)))
        
        return result
    
    
# tokenizer function, this will make 3 grams of each query
    def get_ngrams(self, query):
        tempQuery = str(query)
        ngrams = []
        for i in range(0, len(tempQuery)-n):
            ngrams.append(tempQuery[i:i+n])
        return ngrams

    def kmeans(self, weight):

        printT('kmeans之前矩阵大小： ' + str(weight.shape))
        weight = weight.tolil().transpose()
        # 同一组数据 同一个k值的聚类结果是一样的。保存结果避免重复运算
        try:

            with open('model/k' + str(k) + '.label', 'r') as input:

                printT('loading kmeans success')
                a = input.read().split(' ')

                self.label = [int(i) for i in a[:-1]]

        except FileNotFoundError:

            printT('Start Kmeans ')

            clf = KMeans(n_clusters=k, precompute_distances=False )

            s = clf.fit(weight)
            printT(s)

            # 保存聚类的结果
            self.label = clf.labels_

            # with open('model/' + self.getname() + '.kmean', 'wb') as output:
            #     pickle.dump(clf, output)
            with open('model/k' + str(k) + '.label', 'w') as output:
                for i in self.label:
                    output.write(str(i) + ' ')
        printT('kmeans 完成,聚成 ' + str(k) + '类')
        return weight

    #     转换成聚类后结果 输入转置后的矩阵 返回转置好的矩阵
    def transform(self, weight):

        from scipy.sparse import coo_matrix

        a = set()
        # 用coo存 可以存储重复位置的元素
        row = []
        col = []
        data = []
        # i代表旧矩阵行号 label[i]代表新矩阵的行号
        for i in range(len(self.label)):
            if self.label[i] in a:
                continue
            a.add(self.label[i])
            for j in range(i, len(self.label)):
                if self.label[j] == self.label[i]:
                    temp = weight[j].rows[0]
                    col += temp
                    temp = [self.label[i] for t in range(len(temp))]
                    row += temp
                    data += weight[j].data[0]

        # print(row)
        # print(col)
        # print(data)
        newWeight = coo_matrix((data, (row, col)), shape=(k,weight.shape[1]))
        return newWeight.transpose()


class LG(Baseframe):
    def getname(self):
        if use_k:
            return 'LG__n'+str(n)+'_k'+str(k)
        return 'LG_n'+str(n)

    def __init__(self):
        # 定理逻辑回归方法模型
        self.classifier = LogisticRegression()


class SVM(Baseframe):

    def getname(self):
        if use_k:
            return 'SVM__n'+str(n)+'_k'+str(k)
        return 'SVM_n'+str(n)

    def __init__(self):
        # 定理逻辑回归方法模型
        self.classifier = svm.SVC()


