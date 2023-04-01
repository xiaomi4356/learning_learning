import numpy as np
import functools

from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder



#处理预测结果，使得n*class的矩阵中，概率最大的类别为True
def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret

#@的用法：是一个装饰器，针对函数，起调用传参的作用。

def label_classification(embeddings, y, ratio, logger):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    #
    # #创建一个编码器(model)：将特征进行独热编码，即为每个类别创建一个二进制的列，并返回一个稀疏矩阵或密集数组
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(bool)
    # #transform(X):	Transform X using one-hot encoding.
    # #将Y变成一个shape=n*class的数组，类型是bool型

    X = normalize(X, norm='l2')

    #把矩阵或者数组切割成训练集和测集
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1 - ratio )

    #逻辑回归分类器，这是一个model
    logreg = LogisticRegression(penalty='l2', solver='liblinear')
    # logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    #创建网格搜索model
    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,verbose=0)
    # 网格搜索和交叉验证
    # OneVsRestClassifier：用一个分类器对应一个类别， 每个分类器都把其他全部的类别作为相反类别看待
    #param_grid:需要最优化的参数的取值，值为字典或者列表
    #n_jobs: 并行数，int：个数,-1：跟CPU核数一致, 1:默认值
    #cv:交叉验证参数，默认None.
    #verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出。

    #运行网格搜索
    clf.fit(X_train, y_train)
    logger.info('Best: best_score:{}, best_params:{}'.format(clf.best_score_, clf.best_params_))

    #predict_proba是OneVsRestClassifier的返回函数
    #predict_proba返回的是一个n行k列的数组,第i行第j列上的数值是模型预测第i个预测样本为某个标签的概率,并且每一行的概率和为1。
    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    # y_pred = np.argmax(y_pred, axis=1)

    #F1-score可以看作是模型精确率和召回率的一种加权平均，它的最大值是1，最小值是0。F1-score越大自然说明模型质量更高。
    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    return acc



