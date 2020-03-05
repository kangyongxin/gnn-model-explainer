# install
pywidgets==7.5.1 没装上

tensorflow-gpu 1.14.0 与tensorboard 1.15不配套

dopamine3.0.1与pillow 4.1.1不配套

# 代码初步

python train.py --dataset=EXPERIMENT_NAME 是用来 训练GCN的，可以看看GCN的训练过程

python explainer_main.py --dataset=EXPERIMENT_NAME 是这篇文章的重点，是用来得到一个解释网络的。

tensorboard --logdir log 用来显示结果。

## GCN的训练

train.py中的入口函数是main(), 每一个不同的任务会有一个相应的task入口，应该是由dataset 参数指定的。

先看看不指定dataset的时候用的benchmark_task是什么？
默认的数据集在configs中，syn1; 

以下1-5是benchmark_task中的任务，a-d是syn1中的:

1.先从io_utils中读出一个图结构graphs

2.feat == 各种，但是这个feat后来干嘛用了；feat 的含义应该是nodes的feature，或者node的label

3.    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = prepare_data(
        graphs, args, max_nodes=args.max_nodes
    )
    用graphs来准备数据，准备数据之后得到的是训练集验证集和测试集，以及最大节点数目，输入维度，assign_input_dim 是什么意思？

4.base方法的时候模型构造是model = models.GcnEncoderGraph（）

5.训练    train(
        train_dataset,
        model,
        args,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        writer=writer,
    )

6.评估 evaluate(test_dataset, model, args, "Validation")

syn1, 中的基本任务，这应该是一个节点分类的任务，从最后一句可以看出：
A.
    G, labels, name = gengraph.gen_syn1(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    )
B.
    num_classes
C.
            model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )

D.
    train_node_classifier(G, labels, model, args, writer=writer)
前面三步是准备工作，把数据集构造成基本的图，然后最后一步是训练，先看构图的过程，再看训练的过程

A. 
    图的生成（构造数据集的过程）：任务好像是这个样子的，它原先有一个图（Barabasi-Albert graph），在这个图中附加一些房子形状的子图（house-shaped subgraphs）。这个BA是一个常用的用来生成随机图的数据集或者叫生成集，另外一个常用的叫做Erdos-Rényi 图。
    
    不同点在于，ER中在 Erdos-Rényi 模型中，我们构建一个带有 n 个节点的随机图模型。这个图是通过以概率 p 独立地在节点 (i,j) 对之间画边来生成的。因此，我们有两个参数：节点数量 n 和概率 p。
    + n = 50
    + p = 0.2
    + G_erdos = nx.erdos_renyi_graph(n,p, seed =100)
    在BA 在 Barabasi-Albert 模型中，我们构建一个有 n 个节点的随机图模型，其有一个优先连接（preferential attachment）分量。这种图可通过以下算法生成：
    步骤 1：以概率 p 执行步骤 2，否则执行步骤 3
    步骤 2：将一个新节点连接到随机均匀选取的已有节点
    步骤 3：以与 n 个已有节点成比例的概率将这个新节点连接到这 n 个已有节点
    这个图的目标是建模优先连接（preferential attachment），真实世界网络中常会观察到这一点。（注：优先连接是指根据各个个体或对象已有的量来分配某个量，这通常会进一步加大优势个体的优势。）
    + n = 150
    + m = 3
    + G_barabasi = nx.barabasi_albert_graph(n,m)  
    参考https://www.jiqizhixin.com/articles/2019-07-30-10?utm_source=tuicool&utm_medium=referral

这里建图用的是G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, m=5
    ) % Utilities for generating certain graph shapes.先通过ba 来建立一个basis 图（graph = nx.barabasi_albert_graph(width, m)）， 在固定的节点上加边来实现特有的结构
有个比较有趣的函数eval 他把相应的字符串形式的名称转换成函数名，然后把相应的变量传进去。所以这里先用ba 构造基本图basis，然后用house 构造子图graph_s, 最后把子图嵌入到基本图中，并返回给主函数。

B. 根据构建过程中的标签数量来定义类别数（这个感觉是个trick）

C. Model, 这里编码的是node , 而在model 文件夹中还有别的编码比如，编码graph，要注意区分。具体细节：gcnencodernode是从GcnEncoderGraph中继承来的，GraphConv负责构建基本的GCN网络，


更想关注的是如何在networkx的基础上使用GCN
