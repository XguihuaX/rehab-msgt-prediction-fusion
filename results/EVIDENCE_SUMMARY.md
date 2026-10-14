# 已完成结果的阅读顺序

本页是仓库自带的简短索引，避免只同步 CSV 后失去实验版本和结论边界。数值均可在 `data/` 下的相应表格及 `CLAIMS_INDEX.json` 中核验。

1. **完整 Dynamic MSGT 的主对照。** 五设置的 MSGT 预算平均主指标中，原 `alpha=n/(n+0.5)` 相对同模型直接适配在 8/10 个任务×FT/LP 行更低；U2K、KERAAL、IRDS、REHAB24-6 为改善方向，K2U proxy 两行退步。读 `primary/five_settings_msgt_main.csv`；比较必须按每任务指标与原参与者/重复层级进行，不把 RMSE 和 Brier 混合成一个总分。
2. **跨受测模型接入。** 五设置的 28 个任务×骨干×FT/LP 预算汇总行中，相同人数规则相对各自 direct 为 25 胜 3 负；排除 MSGT 后，其他受测基线 17 胜 1 负。三处非改善是 K2U proxy 的 MSGT FT/LP，以及 REHAB24-6 的 STGCN-Rehab LP。REHAB24-6 的 CTR-GCN LP 仅约 `0.000009` Brier 的点估计改善，不宜写成大幅收益。读 `primary/five_settings_all_models.csv`。
3. **最新同协议领域近邻。** 新 U2K/IRDS 矩阵分别在 MSGT、RI-GCN 上给出 12/12 个同模型 direct→Dynamic 主指标改善，合计 24/24。相同适配/规则下 MSGT 相对 RI-GCN 的 direct、固定融合、原 Dynamic 分别在 9/12、11/12、10/12 个设置主指标更低；这些也是相关设置计数，不是独立患者检验。读 `primary/fair_u2k_irds_summary.csv` 及其描述性配对区间。新矩阵的训练选择和 U2K repeat 数与较早五设置不同，两套数值不得直接合并排名。
4. **结构与强基线。** 原端点对齐消融中，Full MSGT 在 16 个 RMSE/MAE 配对对照里有 14 个获支持、2 个未分开；U2K 最大预算 FT 的旧架构表给 MSGT RMSE 0.2132、ST-GCN 0.2252、2s-AGCN 0.2196、CTR-GCN 0.2154，MSGT 与 CTR 的配对区间未分开。原单分支 MSGT 为约 0.488M 参数、0.412G MAC；完整输出融合为 975,746 参数与双前向。旧表原文在 `legacy_tables/`，不要将其视作最新公平适配的预算结果。KERAAL 的 ReCon LP 在 Brier/BA/AUROC 均值上强于 Dynamic MSGT LP，详见 `primary/recon_keraal_*.csv`。
5. **适配策略和个体差异。** U2K 低人数预算 LP、高预算 FT 的排序变化，以及分类 AUROC 与 Brier/log-loss 的取舍在 `diagnostics/ftlp_budget_comparison.csv`。受益/退步人数和 KERAAL 动作层聚合在 `diagnostics/participant_benefit_summary.csv`、`action_brier_summary.csv`；没有逐人身份或轨迹。
6. **规则、机制和后续探索。** `diagnostics/risk_*.csv` 保存源/适配误差及交叉矩条件；`exploratory/function_sweep_across_budget.csv` 保存 15 阶段各已比较函数，未从测试结果中挑选新的正式主规则。新风险梯度权重在已看过的 U2K/IRDS 上做过一次事后探索：相对旧动态 21/24 更低、相对 direct 22/24 更低；它没有独立验证，也不能继承旧规则的五设置覆盖。读 `exploratory/new_weight_*.csv`。
7. **成本。** `primary/older_training_cost.csv` 与 `older_inference_cost.csv` 对应原完整工作配置；新公平矩阵的六次内拟合搜索、最终重拟合和 32 项设备内推理在 `primary/fair_*cost.csv`。源训练可复用，但完整输出融合仍需两个模型与两次前向；设备内计时不包含姿态提取或端到端临床延迟。

所有“胜/负”都是当前协议下的描述性比较。PASS 表示结果来源和计算通过独立核验，不等于临床验证、跨设备外推或新权重的盲测成功。负格、简单先验、强外部方法和计算成本都保留在仓库数据中。
