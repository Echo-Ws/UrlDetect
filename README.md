# 环境

> Python 环境： 3.5


# 说明
该项目是参考[用机器学习玩转恶意URL检测](http://www.freebuf.com/articles/network/131279.html) 后的改进版本，加入了svm分类器，依赖的库可以参见那篇文章。
参见test.py即可知道本系统如何运行。
详细的设计思路及分析可见
[通过机器学习识别恶意url](http://blog.csdn.net/solo_ws/article/details/77095341)
## 数据集
good_fromE 某系统的某天的正常访问url，已去重
good_fromE2 同上
bad_fromE 利用sql注入某系统产生的url记录
badqueries 来源于网上数据
goodqueries 来源于网上数据


## 参数设定
在IDS.py中可以设定是否使用kmeans降维，k的大小，n-gram中n的值。 比较懒，参数全放在py文件中前面设定。后期可以改为读取文件之类
通过test.py选择训练的模型。这里提供逻辑分类器和svm分类器。具体参数都是默认，如果想修改可以从IDS.py中修改，注释很多保证能看懂。

## 模型
.label 文件保存的是分词结果，只与数据集有关
.pickle是训练后的模型，各位按需自取

