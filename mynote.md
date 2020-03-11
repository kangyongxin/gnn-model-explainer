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

C. Model, 这里编码的是node , 而在model 文件夹中还有别的编码比如，编码graph，要注意区分。具体细节：gcnencodernode是从GcnEncoderGraph中继承来的，GraphConv负责构建基本的GCN网络，三个模块之间是继承的关系； 

cNN网络的定义方法：(详见cnnclassifier.py)
1.定义一个 net,其_init_部分定义基本的网络单元，比如，卷积,池化，全连接等等
2.然后有一个forward 函数，定义整个前向网络的结构，其中，会涉及到一些基本的网络架构的搭建，比如relu , view(类似于flatten)，最后得到一个输出（这个输出是网络结构而不是网络输出）Net(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 64, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=1600, out_features=240, bias=True)
  (fc2): Linear(in_features=240, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
同时定义loss的计算模式，以及优化方法
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
3.训练过程：
3.1 先得到输入和标签inputs, labels = data
3.2 清零优化器，也就是之前存储的梯度
3.3 通过网络得到输出
3.4 通过criterion 计算loss
3.5 误差反传
3.6 更新参数
4.每个epoch包含多个batch

GCN 的定义方法：参考的是（models.py）
GCN 的基本操作是在GraphConv()函数中的，应该同样是在init 中构建基本的操作，在forward 中构建基本的网络结构。
1.在init中，定义了 att, add_self, dropout, normalize_embedding, input_dim, output_dim 等参数，定义了 基本的参数结构， weight, self_weight 和 att_weight, 和bias.
2.forward 中利用上面定义参数结构，
2.1 进入之后经过dropout；
2.2 然后和att_weight 相乘得到x_att,  之后的操作att = x_att @ x_att.permute(0, 2, 1)没看懂是干啥的，最后得到了一个adj应该是邻接矩阵
2.3 y的计算，adj*x*weight + x*self_weight
2.4 对F进行normalize

最后这个函数输出是一个y(y应该是一个模型) 和一个邻接矩阵（这个邻接矩阵不一定会被改变）

GcnEncoderGraph()
与基本的pytorch构建网络的套路是相近的，先定义一些网络单元，然后再把所有的模块组装起来；
模块有：build_conv_layers, build_pred_layers,construct_mask, apply_bn,gcn_forward
然后是 forward
最后构建loss 

1.build_conv_layers 
输出三个conv_first, conv_block, conv_last
1.1 conv_first 是用之前graphConv构造的；（构建的是输入层）
1.2 conv_block 构造多个模块的连接,也是用GraphConv 来完成的，但是有一个问题没想清楚，就是为啥都是同等维度的？
1.3 conv_last 最后一层的输出是embedding dim

2.build_prd_layer
输入中的 pred 是什么意思，预测吗/要看它用在哪儿； 另外这里的num_aggs是啥？
pred_model 是把一串linear和ReLU的组合连接在一起nn.Sequential（），这也是pytorch构造网络常用的一种方法

3.construct_mask
batch中的每一个条目中都会有一个掩码。
Dimension of mask: [batch_size x max_nodes x 1]
把一个用列表表示的形式通过for 循环的方式转换成张量掩码的形式

4.gcnforward
以上面三个模块作为输入，构建前向网络， 返回x_tensor, adj_att_tensor

输入是x和邻接矩阵
用conv_first 构造第一层，主要处理 x, 邻接矩阵不变；
用conv_block 构造中间层, 同样处理x, 不处理adj；
用conv_last 构造输出层；
然后把所有层的x进行批结，得到x_tensor,# x_tensor: [batch_size x num_nodes x embedding]； 得到adj_att_tensor，# adj_att_tensor: [batch_size x num_nodes x num_nodes x num_gc_layers]

init的时候负责把这几个模块定义出来，并构造一些内部变量。

********我们这里的forward 是平行于gcnforward的，应该是自己建立的；
这里比上面多的是每一层都多出一个max操作，不知道是为啥？
另外会有一个掩码层，和输出预测层的存在（ypred）

定义loss, 有两种可能


现在回到最开始的GcnEncoderNode()他继承了GcnEncodergraph(),又重新加入了forward和loss.这里的forward 核心是gcn_forward 上面的4,加上pred 模块，loss 是用pred 和节点标签构造的。


D.
train_node_classifier(G, labels, model, args, writer=writer)
（这个整体是符合torch训练的框架的，先读入数据，然后定义优化器，定义loss， 然后开始循环训练）
G是前面构造的图，跟我们在强化学习中构造的图是一样的。

划分数据集，难道只是把同一个图（自始至终就只有一个图？）中的部分节点拿出来作为训练集？而不是多个不同的图吗？
这样倒是有个好处，就是我们每进行一次探索就可以回来做一次训练，尽管在强化学习过程中图是动态的，但是在单一的一次任务中是静态的。

对图进行预处理，data = gengraph.preprocess_input_graph(G, labels) 得到labels， adj， feature

定义优化器


model.train()是啥？
model.train()
启用 BatchNormalization 和 Dropout
model.eval()
不启用 BatchNormalization 和 Dropout

循环：

    清零梯度

    得到输出（ypred, adj_att = model(x, adj)）

    计算loss

    loss.backward

    更新参数 optimize.step()

这是一个完整的训练过程，其中如何把一副图提取出特征，然后用到gcn中值得借鉴（data = gengraph.preprocess_input_graph(G, labels) 得到labels， adj， feature），至于是否使用gcn进行处理有待进一步探讨。



更想关注的是如何在networkx的基础上使用GCN


