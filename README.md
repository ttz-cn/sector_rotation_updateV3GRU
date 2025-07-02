# sector_rotation_updateV3GRU

# 背景

考虑到多因子选择标的，低胜率高赔率可能在跑业绩过程中效果并不明显:

**【2025-06-23TO2025-07-01】对ETF策略进行如下优化**：

- 特征：

  - 由于基本面因子有效性一般，去除基本面因子，对技术指标、拥挤度指标全面复原，不考虑相关性
  - 特征融合上，参考华泰证券2023-12研报，使用GRU模型融合特征

- 仓位管理上：

  - 根据固定窗口，对夏普比率进行凸优化，过**最优化夏普比率**优化器
  - 直接将**行业择时**嵌入仓位管理（仓位限制根据行业整体收益预测进行动态调整）
 

> 当前待完善：
1.特征参数优化
2.稳健性检验
3.其他

# Readme

## Project Introduction

本项目基于多因子与深度学习（GRU）模型，结合技术、拥挤度、基本面等多维因子，自动化实现行业ETF轮动策略的信号生成与组合权重优化。  
核心亮点：  

- **GRU因子融合**：用深度学习捕捉因子间非线性与时序关系
- **行业专属模型**：每个行业单独建模，提升预测精度
- **智能组合优化**：多约束下自动配置最优权重

## Main Process

1. **数据准备**  
   - 运行 `app_update.py` 自动下载/更新所需数据

2. **因子计算**  
   - 运行 `app_get_factors.py` 计算各类因子

3. **因子融合与模型训练**  
   - 运行 `run_gru_factor_fusion.py --train` 训练各行业GRU模型

4. **预测与组合优化**  
   - 运行 `run_gru_factor_fusion.py --predict --optimize`  
     或单独运行 `run_optimizer.py` 基于预测结果优化权重

5. **结果分析与可视化**  
   - 查看 `output/` 目录下的预测、权重、绩效等结果文件

## File Structure

```
├── app_get_factors.py           # 计算各类因子值
├── app_inter_factor_composite_tfc.py  # 技术+基本面+拥挤度因子合成
├── app_inter_factor_composite_tcf.py  # 技术+拥挤度+基本面因子合成
├── app_update.py                # 数据更新和模型运行主程序
├── config.py                    # 配置文件
├── src/                         # 源代码目录
│   ├── src_get_indicators.py    # 因子计算函数
│   ├── src_get_composit_indicators.py  # 因子合成函数
│   ├── src_indicator_processing.py     # 因子处理和组合优化函数
│   └── src_EtfRotationStrategy.py      # ETF轮动策略实现
├── db/                          # 数据存储目录
└── output/                      # 输出结果目录
```

## Usage 

### 1. 安装依赖

```bash
pip install pandas numpy matplotlib seaborn torch scikit-learn cvxpy h5py tqdm openpyxl xlrd pyyaml joblib scipy
```

### 2. 数据准备

运行数据更新程序：

```bash
python app_update.py
```

### 3. 因子计算

计算各类因子：

```bash
python app_get_factors.py
```

### 4. 因子合成与组合优化

运行因子合成和组合优化：

```bash
python app_inter_factor_composite_tfc.py
```

或者：

```bash
python app_inter_factor_composite_tcf.py
```

## 模型流程

1. **数据获取**：获取行业ETF价格、成交量等数据
2. **因子计算**：计算技术、拥挤度和基本面因子
3. **GRU因子合成**：使用GRU模型合成因子信号
4. **组合优化**：基于优化算法计算权重配置
5. **拥挤度调整**：根据拥挤度阈值调整最终组合

## 参数配置

主要参数在`config.py`中配置：

- `rebalance_period`：调仓周期
- `dic_Industry2Etf`：行业与ETF对应关系
- `selected_indicators`：选择的指标列表
- `optimization_params`：组合优化参数

## 注意事项

- GRU模型训练需要较长时间，建议使用GPU加速
- 组合优化需要安装cvxpy库
- 模型使用5天收益率作为训练目标 

# 【20250110-20250625】回测展示（Ablation Study **Verified** ）：

#### 概括：

r^2和MSE能够证明模型具有较强预测能力。

**周度调仓**的情况下，舍弃20250407的回撤控制超额，模型能跟上1-3和4-6月份的涨幅的，【0418-0628】保持对wind全A的跟踪（8%+）。
其中，4/7那波回撤，我调整很多参数，都是可以预测到的，具有一定的稳健性。

剩下的跑不出来超额我**猜测**是
1.频率不高+持仓少，我们策略出现大回撤时，都仅持仓3只ETF标的，在过少的标持有的前提下获得超额，对周度预测能力要求极高，
然而
2.对于周频调仓，我还没来得及对参数进行优化，需要更适应的参数提取特征

**日度调仓** 我是weight_tolerance=5%，（>5%调仓），手续费0.1%

#### 预测能力（forward5）

```
R² 平均值: 0.9106
R² 中位数: 0.9186
R² 最大值: 0.9703 (801880.SI - 汽车)
R² 最小值: 0.8044 (801780.SI - 银行)

MSE 平均值: 0.000115
MSE 中位数: 0.000104
MSE 最小值: 0.000040 (801880.SI - 汽车)
MSE 最大值: 0.000220 (801750.SI - 计算机)
 ```


#### 周频调仓

![image](https://github.com/user-attachments/assets/1fb7bd16-5003-4f67-b064-c8ad1e3e7c16)

![image](https://github.com/user-attachments/assets/28b84e96-3b74-46e3-890b-8ea239284e3d)

![image](https://github.com/user-attachments/assets/40320981-46fe-421e-8b24-d0890d7c5f90)



#### 日频调仓
![image](https://github.com/user-attachments/assets/a9f809ed-abc6-4980-af3c-33527dcc2891)

![image](https://github.com/user-attachments/assets/5f5b0e51-891f-4a64-9285-ad41de0f1adf)

![image](https://github.com/user-attachments/assets/0bd76c11-db02-4f08-970b-9749f07ca695)

# GRU-FUSION-MODEL GRU融合模块介绍

## Introduction

本模块采用门控循环单元（GRU）深度学习模型，将多种因子（技术、拥挤度、基本面等）融合，专为行业收益率预测设计。每个行业单独训练专属GRU模型，充分挖掘行业特性与因子间复杂关系。

## Architecture

- **输入层**：多因子时序数据
- **GRU层**：提取时序特征
- **全连接层**：输出行业未来收益率预测

## Training and Prediction Process

1. **训练**  

   ```bash
   python run_gru_factor_fusion.py --train
   ```

   - 每个行业独立训练，自动保存最优模型至`models/`

2. **预测**  

   ```bash
   python run_gru_factor_fusion.py --predict
   ```

   - 加载各行业模型，输出预测结果至`output/`

3. **一键训练+预测+优化**  

   ```bash
   python run_gru_factor_fusion.py --train --predict --optimize
   ```

## Hyperparameter Optimization

- 隐藏层大小（hidden_size）：[20, 30, 40]
- 学习率（learning_rate）：[0.001, 0.005, 0.0001]
- 采用时间序列交叉验证与早停策略，自动为每个行业选择最优参数

## Output File Explanation

- `models/gru_model_{industry}.pkl`：各行业GRU模型
- `output/gru_predictions_{date}.csv`：预测结果
- `output/gru_metrics_{date}.csv`：模型评估与参数
- `output/historical_weights_*.csv`：历史最优权重
- 其他可视化与日志文件详见`output/`

## Data Normalization

- 训练前对每个时间点的因子数据做标准化，提升模型稳定性

## 优势总结

- **行业专属建模**：提升预测精度
- **自动超参搜索**：无需手动调参
- **多因子融合**：捕捉复杂关系
- **全流程自动化**：一键训练、预测、优化

## 常见问题与建议

- 确保`db/`、`output/`、`models/`目录存在
- 数据量大时建议使用GPU
- 若模型训练缓慢，可适当减少行业数量或调整参数
- 预测结果异常时，建议检查因子数据和训练日志

## 依赖库

- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- cvxpy
等 ，待更新
---

如需主项目说明与完整流程，请参见 [README.md](./README.md) 





> 





