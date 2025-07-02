# GRU因子融合模型说明

## 模型简介

本模块采用门控循环单元（GRU）深度学习模型，将多种因子（技术、拥挤度、基本面等）融合，专为行业收益率预测设计。每个行业单独训练专属GRU模型，充分挖掘行业特性与因子间复杂关系。

## 模型结构

- **输入层**：多因子时序数据
- **GRU层**：提取时序特征
- **全连接层**：输出行业未来收益率预测

## 训练与预测流程

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

## 超参数优化

- 隐藏层大小（hidden_size）：[20, 30, 40]
- 学习率（learning_rate）：[0.001, 0.005, 0.0001]
- 采用时间序列交叉验证与早停策略，自动为每个行业选择最优参数

## 输出文件说明

- `models/gru_model_{industry}.pkl`：各行业GRU模型
- `output/gru_predictions_{date}.csv`：预测结果
- `output/gru_metrics_{date}.csv`：模型评估与参数
- `output/historical_weights_*.csv`：历史最优权重
- 其他可视化与日志文件详见`output/`

## 数据标准化

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

---

如需主项目说明与完整流程，请参见 [README.md](./README.md) 