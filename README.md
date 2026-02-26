## 1. 核心文件夹
### datasets/
- 训练和测试数据集目录
  - train/ ：训练集图像，包含 Negative/ 和 Positive/ 两个子目录
  - test/ ：测试集图像，包含 test_negative/ 和 test_postive/ 两个子目录
### logs/
- 训练日志和模型权重目录
  - 保存每个训练过程的日志文件（损失曲线、指标记录）
  - 存储不同训练阶段的模型权重文件（.pth）
  - 包含 TensorBoard 事件文件，用于可视化训练过程
### model_data/
- 模型配置和预训练权重目录
  - cls_classes.txt ：定义分类类别（positive 和 negative）
  - 预训练权重文件（如 MobileNetV2 的预训练权重）
### nets/
- 网络模型实现目录
  - mobilenetv2.py ：MobileNetV2 网络模型实现
  - resnet.py ：ResNet 网络模型实现
  - vgg.py ：VGG 网络模型实现
### utils/
- 工具函数目录
  - callbacks.py ：回调函数实现，包含 LossHistory 类用于保存训练日志
  - dataloader.py ：数据加载器实现，用于加载和预处理训练数据
  - utils_aug.py ：数据增强工具函数
  - utils_fit.py ：训练工具函数，包含 fit_one_epoch 函数
  - utils.py ：通用工具函数，如图像预处理、类别加载等
## 2. 主要脚本
### 训练相关
- train.py ：核心训练脚本，实现完整的模型训练流程
- val.py ：模型验证脚本，用于评估模型在验证集上的性能
### 预测相关
- predict.py ：单张图像分类预测脚本，支持命令行参数输入图像路径
- classification.py ：分类推理脚本，另一种实现图像分类的方式
### 数据处理
- generate_train_file.py ：生成训练集标签文件（cls_train.txt）
- generate_test_file.py ：生成测试集标签文件（cls_test.txt）
- check_data_distribution.py ：分析数据集分布，检查类别平衡和图像一致性
- verify_val_data.py ：验证验证集数据加载是否正确
### 工具脚本
- clean_logs.py ：清理训练日志，删除不完整的训练日志文件夹
- txt_annotation.py ：生成数据集标注文件
## 3. 配置文件
- requirements.txt ：项目依赖文件，列出所需的 Python 包及其版本
- .gitignore ：Git 忽略规则文件，指定哪些文件或目录不纳入版本控制
- LICENSE ：项目许可证文件，定义项目的使用条款和条件
## 4. 工作流程
1. 数据准备 ：使用 generate_train_file.py 和 generate_test_file.py 生成标签文件
2. 模型训练 ：运行 train.py 进行模型训练，自动保存训练日志和模型权重
3. 模型验证 ：使用 val.py 评估训练好的模型性能
4. 模型预测 ：通过 predict.py 对单张图像进行分类预测
这个项目结构清晰，功能完整，涵盖了从数据准备到模型训练、验证和预测的全流程，是一个标准的深度学习图像分类项目架构。
