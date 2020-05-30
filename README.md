# bert_text_classification



### 文本分类流程

+ ![1590812453(1)](D:\bert_text_cnn\bert_text_classification\images\1590812453(1).png)

### CNN

+ CNN中Convolution和Max pooling的作用
  + Convolution目的是检测出图像中的主要特征
  + Max pooling是下采样，不改变图片原本面貌

### TextCNN

+ Yoon Kim在论文(2014 EMNLP) Convolutional Neural Networks for Sentence Classification

  提出TextCNN。 将卷积神经网络CNN应用到文本分类任务，利用多个不同size的kernel来提取句子中的关键信息

  （类似于多窗口大小的ngram），从而能够更好地捕捉局部相关性。

  ![1590812626](D:\bert_text_cnn\bert_text_classification\images\1590812626.png)

+ TextCNN模型结构

  + ![1590812792(1)](D:\bert_text_cnn\bert_text_classification\images\1590812792(1).png)

+ TextCnn与image-CNN的差别：

  +  最大的不同便是在**输入数据**的不同；

  + 图像是二维数据, 图像的卷积核是从左到右, 从上到下进行滑动来进行特征抽取；

  + 自然语言是一维数据, 虽然经过word-embedding 生成了二维向量，但是对词向量只能做从上到下，做从左到右滑动来进行卷积没有意义；

  + 文本卷积宽度的固定的，宽度的就embedding的维度。

     

+ TextCNN的成功, 不是网络结构的成功, 而是通过引入已经训练好的词向量，来在多个数据集上达到了超越benchmark 的表现，进一步证明了构造更好的embedding, 是提升nlp 各项任务的关键能力。

  

+ TextCNN训练详细过程

  + ![1590813717(1)](D:\bert_text_cnn\bert_text_classification\images\1590813717(1).png)
    +  Embedding：第一层是图中最左边的7乘5的句子矩阵，每行是词向量，维度=5，这个可以类比为图像中的原始像素点。
    + Convolution：然后经过 kernel_sizes=(2,3,4) 的一维卷积层，每个kernel_size 有两个输出 channel。
    + MaxPolling：第三层是一个1-max pooling层，这样不同长度句子经过pooling层之后都能变成定长的表示。
    +  FullConnection and Softmax：最后接一层全连接的 softmax 层，输出每个类别的概率。

  + 通道：
    +   图像中可以利用 (R, G, B) 作为不同channel；
    + 文本的输入的channel通常是不同方式的embedding方式（比如 word2vec或Glove），实践中也有利用静态词向量和fine-tunning词向量作为不同channel的做法。
  + 一维卷积（conv-1d）：
    +  图像是二维数据；
    + 文本是一维数据，因此在TextCNN卷积用的是一维卷积（在word-level上是一维卷积；虽然文本经过词向量表达后是二维数据，但是在embedding-level上的二维卷积没有意义）。一维卷积带来的问题是需要通过   设计不同 kernel_size 的 filter 获取不同宽度的视野。

###  RCNN

+   RCNN模型来源于论文Recurrent Convolutional Neural Networks for Text Classification。 简单的说就是 双向lstm+poolig层构成

+ 模型结构：

  + ![1590814167(1)](D:\bert_text_cnn\bert_text_classification\images\1590814167(1).png)

+ RCNN 整体的模型构建流程如下：

  +  利用Bi-LSTM获得上下文的信息，类似于语言模型；

  +  将Bi-LSTM获得的隐层输出和词向量拼接[fwOutput, wordEmbedding, bwOutput]；

  + 将拼接后的向量非线性映射到低维；

  +  向量中的每一个位置的值都取所有时序上的最大值，得到最终的特征向

    量，该过程类似于max-pool；

  + softmax分类。