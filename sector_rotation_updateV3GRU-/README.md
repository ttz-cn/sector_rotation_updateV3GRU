# 行业轮动策略优化项目

## 项目简介

本项目基于多因子与深度学习（GRU）模型，结合技术、拥挤度、基本面等多维因子，自动化实现行业ETF轮动策略的信号生成与组合权重优化。  
核心亮点：  
- **GRU因子融合**：用深度学习捕捉因子间非线性与时序关系
- **行业专属模型**：每个行业单独建模，提升预测精度
- **智能组合优化**：多约束下自动配置最优权重

## 主要流程

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

## 主要文件结构

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

## 使用方法

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