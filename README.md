# Mercari-Price

参加kaggle比赛记录。


##比赛的链接地址为：
https://www.kaggle.com/c/mercari-price-suggestion-challenge

##首先是特征工程，需要对原始数据进行处理。

对数据要有充分的了解，所以需要画图统计各种信息。例如：商品描述信息有很多值得挖掘的点，比如分词，关键字提取等。sentence2vec,更多的涉及自然语言处理的内容，可以尝试LDA等。

##数据处理好之后就需要选择合适的模型。

这是一个预测回归的问题，需要回归预测模型。

常用的有线性回归，局部线性回归等，加入正则项后有类似的岭回归，lasso回归等。

工业上比较常用的有分类回归决策树，gbdt算法（xgboost,还有微软的lightgbm）。

当然，深度学习（神经网络，深度决策树还没有测试）也可以拿来用。

最后要想更多的提高精度，防止过拟合等等，需要调参，还有stacking.相关网址：http://blog.csdn.net/u014356002/article/details/54376138


