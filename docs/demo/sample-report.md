# 推荐系统离线评估应如何选择数据切分、负采样和排名指标以避免偏差？

## 问题拆解

- 研究推荐系统离线评估中数据切分方法对偏差的影响
- 分析负采样策略对评估指标偏倚的纠正作用
- 设计适合推荐场景的去偏排名指标体系
- 验证组合策略对整体评估偏差的改善效果
- 建立评估偏差的检测与量化模型

## 技术路线

采用离线检索夹具梳理协同过滤、序列建模与评估流程。 [S1] [S2]

## 代表工作

代表性方向包括潜因子方法和基于注意力的序列推荐。 [S1] [S2]

## 实验与指标

建议同时报告 Recall、NDCG、覆盖率，并固定数据划分。 [S1] [S2]

## 论文与代码对照

- 基于时序分解和随机森林的时间序列多步预测算法：任务类型未知；代码：未确认公开代码。 [S1] [S2]
- 基于不确定性的多元时间序列分类算法研究：召回；代码：未确认公开代码。 [S1] [S2]
- 10 时间序列预测 | 金融时间序列分析讲义：任务类型未知；代码：未确认公开代码。 [S1] [S2]
- 什么是时间序列模型？| IBM：任务类型未知；代码：未确认公开代码。 [S1] [S2]
- 时间序列分析：步骤、类型和示例 - MATLAB & Simulink：任务类型未知；代码：未确认公开代码。 [S1] [S2]
- CN118657591A - 一种基于个性化推荐系统的负样本采样方法 - Google Patents：任务类型未知；代码：未确认公开代码。 [S1] [S2]
- 推荐系统中负采样策略及采样偏差的校正方法 - 石头开会 - 博客园：召回, 排序；代码：未确认公开代码。 [S1] [S2]
- [论文评述] Negative Sampling in Recommendation: A Survey and Future Directions：任务类型未知；代码：未确认公开代码。 [S1] [S2]
- 基于社交扩散和自适应负采样的推荐算法：任务类型未知；代码：未确认公开代码。 [S1] [S2]
- 基于隐式用户反馈数据流的实时个性化推荐：排序；代码：未确认公开代码。 [S1] [S2]
- 长文！推荐‑搜索‑广告系统评估指标与损失函数技术报告 - 石头开会 - 博客园：排序, 多任务学习；代码：未确认公开代码。 [S1] [S2]
- [论文评述] Offline Preference-Based Trajectory Evaluation：任务类型未知；代码：未确认公开代码。 [S1] [S2]
- 基于因果推断的推荐系统去偏研究综述：排序；代码：未确认公开代码。 [S1] [S2]
- 推荐系统评价指标综述 - 山东大学信息检索实验室：排序；代码：未确认公开代码。 [S1] [S2]
- 推荐算法的离线评价指标综述：召回；代码：未确认公开代码。 [S1] [S2]
- 推荐系统系列之排序任务的样本工程 | 亚马逊AWS官方博客：排序；代码：未确认公开代码。 [S1] [S2]
- [论文评述] Does Negative Sampling Matter? A Review with Insights into its Theory and Applications：任务类型未知；代码：未确认公开代码。 [S1] [S2]
- DSSM负采样版 — easy_rec 0.8.8 documentation：任务类型未知；代码：未确认公开代码。 [S1] [S2]
- 深度学习炼丹-不平衡样本的处理 - 文章 - 开发者社区 - 火山引擎：排序；代码：未确认公开代码。 [S1] [S2]
- CN110781340B - 一种推荐系统召回策略的离线评估方法、系统、装置及存储介质 - Google Patents：召回, 排序；代码：未确认公开代码。 [S1] [S2]
- 推荐系统及相关算法简介 - Iawen's Blog - 风无形，水无势，互联网没有昼夜。趁这些许的闲暇时光，随手采摘或记录着这知识海洋的点点滴滴......：召回；代码：未确认公开代码。 [S1] [S2]
- 推荐系统评估指标 — PaddleEdu documentation：召回, 排序；代码：未确认公开代码。 [S1] [S2]
- 【推荐算法的评估与调试】离线评估+在线A/B Test：召回, 排序；代码：未确认公开代码。 [S1] [S2]

## 数据集与指标

- 基于时序分解和随机森林的时间序列多步预测算法：数据集 unknown；指标 unknown。 [S1] [S2]
- 基于不确定性的多元时间序列分类算法研究：数据集 unknown；指标 unknown。 [S1] [S2]
- 10 时间序列预测 | 金融时间序列分析讲义：数据集 unknown；指标 unknown。 [S1] [S2]
- 什么是时间序列模型？| IBM：数据集 unknown；指标 unknown。 [S1] [S2]
- 时间序列分析：步骤、类型和示例 - MATLAB & Simulink：数据集 unknown；指标 unknown。 [S1] [S2]
- CN118657591A - 一种基于个性化推荐系统的负样本采样方法 - Google Patents：数据集 unknown；指标 unknown。 [S1] [S2]
- 推荐系统中负采样策略及采样偏差的校正方法 - 石头开会 - 博客园：数据集 unknown；指标 unknown。 [S1] [S2]
- [论文评述] Negative Sampling in Recommendation: A Survey and Future Directions：数据集 unknown；指标 unknown。 [S1] [S2]
- 基于社交扩散和自适应负采样的推荐算法：数据集 unknown；指标 unknown。 [S1] [S2]
- 基于隐式用户反馈数据流的实时个性化推荐：数据集 unknown；指标 unknown。 [S1] [S2]
- 长文！推荐‑搜索‑广告系统评估指标与损失函数技术报告 - 石头开会 - 博客园：数据集 unknown；指标 Recall, NDCG, MRR。 [S1] [S2]
- [论文评述] Offline Preference-Based Trajectory Evaluation：数据集 unknown；指标 unknown。 [S1] [S2]
- 基于因果推断的推荐系统去偏研究综述：数据集 unknown；指标 unknown。 [S1] [S2]
- 推荐系统评价指标综述 - 山东大学信息检索实验室：数据集 unknown；指标 AUC。 [S1] [S2]
- 推荐算法的离线评价指标综述：数据集 unknown；指标 unknown。 [S1] [S2]
- 推荐系统系列之排序任务的样本工程 | 亚马逊AWS官方博客：数据集 unknown；指标 unknown。 [S1] [S2]
- [论文评述] Does Negative Sampling Matter? A Review with Insights into its Theory and Applications：数据集 unknown；指标 unknown。 [S1] [S2]
- DSSM负采样版 — easy_rec 0.8.8 documentation：数据集 unknown；指标 unknown。 [S1] [S2]
- 深度学习炼丹-不平衡样本的处理 - 文章 - 开发者社区 - 火山引擎：数据集 unknown；指标 unknown。 [S1] [S2]
- CN110781340B - 一种推荐系统召回策略的离线评估方法、系统、装置及存储介质 - Google Patents：数据集 unknown；指标 unknown。 [S1] [S2]
- 推荐系统及相关算法简介 - Iawen's Blog - 风无形，水无势，互联网没有昼夜。趁这些许的闲暇时光，随手采摘或记录着这知识海洋的点点滴滴......：数据集 unknown；指标 AUC。 [S1] [S2]
- 推荐系统评估指标 — PaddleEdu documentation：数据集 unknown；指标 LogLoss。 [S1] [S2]
- 【推荐算法的评估与调试】离线评估+在线A/B Test：数据集 unknown；指标 AUC。 [S1] [S2]

## 复现难度分析

- 基于时序分解和随机森林的时间序列多步预测算法：high (score=4)；+2：未识别到公开代码；+2：未识别到公开数据集。 [S1] [S2]
- 基于不确定性的多元时间序列分类算法研究：high (score=4)；+2：未识别到公开代码；+2：未识别到公开数据集。 [S1] [S2]
- 10 时间序列预测 | 金融时间序列分析讲义：high (score=4)；+2：未识别到公开代码；+2：未识别到公开数据集。 [S1] [S2]
- 什么是时间序列模型？| IBM：high (score=4)；+2：未识别到公开代码；+2：未识别到公开数据集。 [S1] [S2]
- 时间序列分析：步骤、类型和示例 - MATLAB & Simulink：high (score=4)；+2：未识别到公开代码；+2：未识别到公开数据集。 [S1] [S2]
- CN118657591A - 一种基于个性化推荐系统的负样本采样方法 - Google Patents：high (score=4)；+2：未识别到公开代码；+2：未识别到公开数据集。 [S1] [S2]
- 推荐系统中负采样策略及采样偏差的校正方法 - 石头开会 - 博客园：high (score=4)；+2：未识别到公开代码；+2：未识别到公开数据集。 [S1] [S2]
- [论文评述] Negative Sampling in Recommendation: A Survey and Future Directions：high (score=4)；+2：未识别到公开代码；+2：未识别到公开数据集。 [S1] [S2]
- 基于社交扩散和自适应负采样的推荐算法：high (score=4)；+2：未识别到公开代码；+2：未识别到公开数据集。 [S1] [S2]
- 基于隐式用户反馈数据流的实时个性化推荐：high (score=4)；+2：未识别到公开代码；+2：未识别到公开数据集。 [S1] [S2]
- 长文！推荐‑搜索‑广告系统评估指标与损失函数技术报告 - 石头开会 - 博客园：high (score=4)；+2：未识别到公开代码；+2：未识别到公开数据集。 [S1] [S2]
- [论文评述] Offline Preference-Based Trajectory Evaluation：high (score=4)；+2：未识别到公开代码；+2：未识别到公开数据集。 [S1] [S2]
- 基于因果推断的推荐系统去偏研究综述：high (score=4)；+2：未识别到公开代码；+2：未识别到公开数据集。 [S1] [S2]
- 推荐系统评价指标综述 - 山东大学信息检索实验室：high (score=4)；+2：未识别到公开代码；+2：未识别到公开数据集。 [S1] [S2]
- 推荐算法的离线评价指标综述：high (score=4)；+2：未识别到公开代码；+2：未识别到公开数据集。 [S1] [S2]
- 推荐系统系列之排序任务的样本工程 | 亚马逊AWS官方博客：high (score=6)；+2：未识别到公开代码；+2：未识别到公开数据集；+2：需要多卡或分布式运行。 [S1] [S2]
- [论文评述] Does Negative Sampling Matter? A Review with Insights into its Theory and Applications：high (score=4)；+2：未识别到公开代码；+2：未识别到公开数据集。 [S1] [S2]
- DSSM负采样版 — easy_rec 0.8.8 documentation：high (score=4)；+2：未识别到公开代码；+2：未识别到公开数据集。 [S1] [S2]
- 深度学习炼丹-不平衡样本的处理 - 文章 - 开发者社区 - 火山引擎：high (score=4)；+2：未识别到公开代码；+2：未识别到公开数据集。 [S1] [S2]
- CN110781340B - 一种推荐系统召回策略的离线评估方法、系统、装置及存储介质 - Google Patents：high (score=4)；+2：未识别到公开代码；+2：未识别到公开数据集。 [S1] [S2]
- 推荐系统及相关算法简介 - Iawen's Blog - 风无形，水无势，互联网没有昼夜。趁这些许的闲暇时光，随手采摘或记录着这知识海洋的点点滴滴......：high (score=4)；+2：未识别到公开代码；+2：未识别到公开数据集。 [S1] [S2]
- 推荐系统评估指标 — PaddleEdu documentation：high (score=4)；+2：未识别到公开代码；+2：未识别到公开数据集。 [S1] [S2]
- 【推荐算法的评估与调试】离线评估+在线A/B Test：high (score=4)；+2：未识别到公开代码；+2：未识别到公开数据集。 [S1] [S2]

## 三天复现建议

- 第一天：核对论文、代码、数据许可及评估口径。
- 第二天：运行最小基线，固定随机种子与配置。
- 第三天：复现主指标，记录差异、风险和未决证据。

## 局限性

本报告完全由虚构的离线测试来源生成，不能替代真实文献检索。

## References

[S1] 基于时序分解和随机森林的时间序列多步预测算法 — https://journal.ecust.edu.cn/cn/article/pdf/preview/10.14135/j.cnki.1006-3080.20220810001.pdf
[S2] 基于不确定性的多元时间序列分类算法研究 — https://www.aas.net.cn/cn/article/doi/10.16383/j.aas.c210302?viewType=HTML
[S3] 10 时间序列预测 | 金融时间序列分析讲义 — https://math.pku.edu.cn/teachers/lidf/course/fts/ftsnotes/html/_ftsnotes/forecasting.html
[S4] 什么是时间序列模型？| IBM — https://www.ibm.com/cn-zh/think/topics/time-series-model
[S5] 时间序列分析：步骤、类型和示例 - MATLAB & Simulink — https://ww2.mathworks.cn/discovery/time-series-analysis.html
[S6] CN118657591A - 一种基于个性化推荐系统的负样本采样方法 - Google Patents — https://patents.google.com/patent/CN118657591A/zh
[S7] 推荐系统中负采样策略及采样偏差的校正方法 - 石头开会 - 博客园 — https://www.cnblogs.com/GlenTt/p/19091367
[S8] [论文评述] Negative Sampling in Recommendation: A Survey and Future Directions — https://www.themoonlight.io/zh/review/negative-sampling-in-recommendation-a-survey-and-future-directions
[S9] 基于社交扩散和自适应负采样的推荐算法 — https://www.sciopen.com/local/article_pdf/10.12141/j.issn.1000-565X.250179.pdf
[S10] 基于隐式用户反馈数据流的实时个性化推荐 — http://cjc.ict.ac.cn/online/onlinepaper/wzs-20151228130502.pdf
[S11] 长文！推荐‑搜索‑广告系统评估指标与损失函数技术报告 - 石头开会 - 博客园 — https://www.cnblogs.com/GlenTt/p/19009559
[S12] [论文评述] Offline Preference-Based Trajectory Evaluation — https://www.themoonlight.io/zh/review/offline-preference-based-trajectory-evaluation
[S13] 基于因果推断的推荐系统去偏研究综述 — http://cjc.ict.ac.cn/online/onlinepaper/yxx-2024928105817.pdf
[S14] 推荐系统评价指标综述 - 山东大学信息检索实验室 — https://ir.sdu.edu.cn/~zhuminchen/RS/evaluation.pdf
[S15] 推荐算法的离线评价指标综述 — https://zhuanlan.zhihu.com/p/584923052
[S16] 推荐系统系列之排序任务的样本工程 | 亚马逊AWS官方博客 — https://aws.amazon.com/cn/blogs/china/sample-project-for-sorting-tasks-of-recommendation-system-series
[S17] [论文评述] Does Negative Sampling Matter? A Review with Insights into its Theory and Applications — https://www.themoonlight.io/zh/review/does-negative-sampling-matter-a-review-with-insights-into-its-theory-and-applications
[S18] DSSM负采样版 — easy_rec 0.8.8 documentation — https://easyrec.readthedocs.io/en/latest/models/dssm_neg_sampler.html
[S19] 深度学习炼丹-不平衡样本的处理 - 文章 - 开发者社区 - 火山引擎 — https://developer.volcengine.com/articles/7382358117046026249
[S20] CN110781340B - 一种推荐系统召回策略的离线评估方法、系统、装置及存储介质 - Google Patents — https://patents.google.com/patent/CN110781340B/zh
[S21] 推荐系统及相关算法简介 - Iawen's Blog - 风无形，水无势，互联网没有昼夜。趁这些许的闲暇时光，随手采摘或记录着这知识海洋的点点滴滴...... — http://note.iawen.com/note/ds/recommend
[S22] 推荐系统评估指标 — PaddleEdu documentation — https://paddlepedia.readthedocs.io/en/latest/tutorials/recommendation_system/evaluation_metric.html
[S23] 【推荐算法的评估与调试】离线评估+在线A/B Test — https://blog.csdn.net/m0_48086806/article/details/139655162
