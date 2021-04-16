# 防水堡演示案例

## 案例目的
**本案例演示的目的是，在分析师只能看到样本数据的情况下，可以获得与全量数据情况下同样好的效果。**


## 案例任务背景

**过去的人寿保险申请流程平均需要30天才能完成风险评估与资格确定。整个流程繁琐且战线长，大大降低了人民购买保险的效率。本案例即根据申请人的基本信息，医疗相关信息、保险相关信息、工作、家庭等等信息，实现对申请人申请险种的资格通过响应值预测。**


## 目录结构描述
├── Readme.md                   // 项目说明<p>
├── dataset                     // 数据集                 
│   ├── sample.csv              //抽样训练数据集<p>
│   ├── test.csv                // 全量测试数据集<p>
│   ├── train.csv               // 全量训练数据集<p>
   
├── 0.612_v1_baseline.ipynb     //v1模型——baseline模型，在全量数据上训练的Quadratic_Weighted_Kappa指标值为0.612<p>
├── 0.605_v2_modeling.ipynb     	//v2模型——在全量数据上训练的Quadratic_Weighted_Kappa指标值为0.605<p>
├── 0.614_v3_modeling.ipynb    //v3模型——在全量数据上训练的Quadratic_Weighted_Kappa指标值为0.614<p>
├── 0.659_v4_modeling.ipynb  // v4模型——在全量数据上训练的Quadratic_Weighted_Kappa指标值为0.659<p>
├──EDA.ipynb                 //在抽样训练集上的数据探索过程<p>
├──demo.txt                 //申请结果导出结果<p>
├──requirements.txt              //依赖包<p>


## run

根据requirements.txt  安装依赖包
<p>

**请修改训练集和测试集加载数据的路径后，执行以下操作**

<p>
全量运行0.612_v1_baseline.ipynb，即可得到v1模型——baseline模型的结果，Quadratic_Weighted_Kappa指标值为0.612
<p>
全量运行0.605_v2_modeling.ipynb，即可得到v2模型结果，Quadratic_Weighted_Kappa指标值为0.605
<p>
全量运行0.614_v1_baseline.ipynb，即可得到v3模型结果，Quadratic_Weighted_Kappa指标值为0.614
<p>
全量运行0.659_v1_baseline.ipynb，即可得到v4模型结果，Quadratic_Weighted_Kappa指标值为0.659
<p>
调试环境运行EDA.ipynb，即可在抽样数据上进行数据探索
<p>
！！！有问题，请联系孙慧玲，电话（同微信）18055755893


