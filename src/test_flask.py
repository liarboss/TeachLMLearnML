# -*- encoding: utf-8 -*-
"""
@File    : test_flask.py
@Time    : 2022/5/7 0007 15:24
@Author  : L
@Software: PyCharm
"""
# 云你好服务源码
from flask import Flask, request, render_template

app = Flask(__name__)


# 云你好服务 API 接口
@app.route("/api/hello")
def hello():
    # 看用户是否传递了参数，参数为打招呼的目标
    name = request.args.get("name", "")
    # 如果传了参数就向目标对象打招呼，输出 Hello XXX，否则输出 Hello World
    return f"Hello {name}" if name else "Hello World"


# 启动云你好服务
if __name__ == '__main__':
    app.run()
