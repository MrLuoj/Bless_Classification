# Bless-Classification-XGBoost

针对包厢祝福内容做分类，有12个标签分别为**生日聚会**、**同学聚会**、**商务聚会**、**爱情聚会**、**朋友聚会**、**同事聚会**、**家庭聚会**、**单身聚会**、
**夕阳红**、**新婚聚会**、**喜庆类聚会**、**其它(不属于上述的任何一种)**。

## 环境
- python 3.7
- pandas
- jieba
- sklearn
- xgboost
- flask

## 使用说明
1）在terminal用清华源安装相应包如下：
- pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas
- pip install -i https://pypi.tuna.tsinghua.edu.cn/simple jieba
- pip install -i https://pypi.tuna.tsinghua.edu.cn/simple sklearn
- pip install -i https://pypi.tuna.tsinghua.edu.cn/simple xgboost
- pip install -i https://pypi.tuna.tsinghua.edu.cn/simple flask

2）训练模型：python train.py

3）开启Flask服务：python app.py
- Postman测试样例： http://127.0.0.1:5000/predict?sentence=恭喜小两口喜结良缘，祝福白头偕老，心想事成，如胶似漆，一生幸福美满。

注：predict.py仅为开发人员测试用