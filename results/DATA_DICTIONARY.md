# 数据字典与版本关系

本仓库保留原 CSV 字段名，不为不同协议强行统一指标。缺失/不适用值沿用源表的空串或 `NA`；不能解释为零。`method=ft/lp` 为目标模型的实际更新方式，`rule=source/adapted/direct/dynamic` 为预测规则，两者是不同维度。

|组别|统计单位|主要字段和指标|使用时注意|
|---|---|---|---|
|`primary/five_settings_msgt_main.csv`|任务 × FT/LP 的预定预算等权汇总，共 10 行|`task,method,metric,adapted,dynamic,fixed_half,dynamic_gain,relative_error_reduction_percent`|回归 RMSE、分类 Brier 均越低越好；有 K2U proxy 负结果|
|`primary/five_settings_all_models.csv`|任务 × 实际受测骨干 × FT/LP × 指标 × 规则，共 550 行|`task,model,method,metric,rule,value,source`|只有有实测的组合才有行；`source` 已转换为原项目相对路径|
|`primary/five_settings_budget_curves.csv`|任务/骨干/适配/预算/规则/指标|`budget_n,budget_kind,n_participants,n_repeats,protocol` 等|预算人数不是训练记录数；U2K/K2U 重复分层与分类唯一人分层不同|
|`primary/five_settings_paired_effects.csv`|已报告的预算级规则对比|`comparison,candidate,reference,benefit,ci_low` 等|`benefit` 已按指标方向统一；区间是其原协议的描述性区间，缺失不能补零|
|`primary/fair_u2k_irds_summary.csv`|U2K 三 repeat 或 IRDS 五 fold 后的唯一参与者聚合|`dataset,model,budget,method,rule,metric,value,valid_unique_participants,aggregation`|两个模型、两适配方式、三个预算；分类 AUROC/BA 可定义的分母小于 Brier|
|`primary/fair_u2k_irds_primary_intervals.csv`|同一公平矩阵的预定配对|`candidate,reference,mean_benefit,ci_low,ci_high,ci_scope`|2000 次唯一参与者 cluster 描述性 bootstrap；不覆盖训练集或方法选择不确定性|
|`primary/recon_keraal_*.csv`|KERAAL 六位患者聚合及配对|`candidate,reference,metric,value/mean_delta`|ReCon 用作者模型与损失、按本研究分类接口适配；另计 2000 epoch 源预训练|
|`diagnostics/*`|FT/LP、受益人数、动作、风险矩或参数融合的聚合|各表原字段|没有逐人 ID；机制矩只支持定义的平方风险和条件解释|
|`exploratory/function_sweep_across_budget.csv`|15 个已完成函数阶段的跨预算汇总|`stage,method,subset,rater,variant,mse,rmse,mae,ccc,...`|80 个具名规则包含有意重复的端点；测试后函数扫描不用于选新正式主规则|
|`exploratory/new_weight_*.csv`|U2K/IRDS 两模型 24 个主指标设置及配对|`risk_gradient_shrinkage` 规则|同一批已观察结果上的一次事后再分析；不能继承五设置旧规则的覆盖|
|`legacy_tables/*.tex`|旧投稿稿件的具体表格|原 LaTeX 表格文字|历史端点/原分类协议，与新增公平适配矩阵分开引用|

主要身份定义：`U2K` 是 UI-PRMD→KIMORE 连续评分；`K2U_proxy` 是 KIMORE→UI-PRMD 代理评分；`KERAAL` 为健康人→患者动作错误分类；`irds` 和 `rehab24_6` 为各自库内参与者隔离分类。`dynamic` 在已完成的正式五设置和公平矩阵中指 `alpha=n/(n+0.5)` 的源/目标适配输出融合；它不是逐测试样本门控。`risk_gradient_shrinkage` 是另一个探索版本。

需要溯源时，先看 `MANIFEST.json` 对应 `source_path`、源 SHA 和 `audit_references`，再查原项目报告。不要把数据仓库中的行数当作独立患者数，也不要用表格中绝对值相近替代同协议配对检验。
