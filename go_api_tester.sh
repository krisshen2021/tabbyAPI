#!/bin/bash

# 进入虚拟环境目录并激活虚拟环境
source venv/bin/activate

# 运行 Python 脚本
python remote_api_tester.py

# 如果需要，您可以在运行完 Python 脚本后停用虚拟环境
deactivate