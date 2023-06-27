from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd
app = Flask(__name__)

# 加载训练好的模型
model = joblib.load('iris_model.joblib')
#
# data = pd.read_excel(r'C:\Users\Sunny.y\Desktop\data_test.xlsx')
#
# data.fillna('0', inplace=True)#空填充为0
#
# data.drop(['技术稳定性','技术先进性','保护范围','文献种类代码','预估到期日'],axis=1,inplace=True)
# new_data = data[['权利要求数量','独立权利要求数量','从属权利要求数量','文献页数','首权字数','IPC','申请人数量','简单同族个数','扩展同族个数','转让次数','许可次数','质押次数','诉讼次数','海关备案','专利寿命（月）']]
#
# Y_value = model.predict(new_data)
# print(Y_value)

# 定义API的路由和请求处理函数
@app.route('/api/v1/predict', methods=['POST'])
def predict():
    # 从请求中获取输入数据
    data = request.get_json(force=True)
    X = np.array(data['features'])

    # 对输入数据进行预测
    y_pred = model.predict(X)

    # 将预测结果作为JSON响应返回
    response = {'prediction': y_pred.tolist()}
    return jsonify(response)

if __name__ == '__main__':
    # 启动Web服务器
    app.run(debug=True)