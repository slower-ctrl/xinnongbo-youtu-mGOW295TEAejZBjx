
#### 简介


在本篇文章中，我们采用逻辑回归作为案例，探索神经网络的构建方式。文章详细阐述了神经网络中层结构的实现过程，并提供了线性层、激活函数以及损失函数的定义（实现方法）。


#### 目录


1. 背景介绍
2. 网络框架构建


* 层的定义
* 线性层
* 激活函数
* 损失函数


## 背景介绍




---


在网络的实现过程中，往往设计大量层的计算，对于简单的网络（算法），其实现相对较容易，例如线性回归，但对于逻辑回归，从输入到激活值再到损失估计的过程整体已经较冗长，实现复杂，并且难以维护，因此，我们需要采用系统性的框架来实现网络（算法），以达到更好的性能、可维护性等。


### 以逻辑回归为例初步探究




---


逻辑回归的决策过程可以分为三步


1. 对采样数据X进行评估


z\=Xw\+b∂z∂w\=X2. 激活函数变换


y^\=σ(z)\=11\+e−z∂y^∂z\=σ∘(1−σ)3. 损失函数计算


LOSS\=y∘log⁡y^\+(1−y)∘log⁡(1−y^)∂LOSS∂z\=yy^\+1−y1−y^
> 其中∘表述逐元素相乘或称哈达玛积
> 上述过程中反应出了这些层的一些共性：


1. 计算函数值的输入与计算梯度时的输入相同。
2. 计算函数值时，每一步的输入都是上一步的输出。
3. 计算梯度时，由链式法则，每一步的梯度累乘即为损失关于参数的梯度。



> 函数值的计算过程从输入开始直至输出结果，故称为**前向传播(forward)**
> 导数值的计算过程从输出开始反向直至第一层，故称为**反向传播(backward)**


由此，可以总结，并设计出层的基本代码如下：



```


|  | class Layer { |
| --- | --- |
|  | private: |
|  | MatrixXd para;  // 参数 |
|  | MatrixXd cache; // 缓存 |
|  | public: |
|  | MatrixXd forward(MatrixXd input);     // 前向传播 |
|  | MatrixXd backward(MatrixXd prevGrad); // 反向传播 |
|  | void update(double learning_rate);    // 参数更新 |
|  | } |


```


> 部分层是没有参数的，例如激活函数、损失函数。这里只有线性层是有参数的。


### 框架功能探索




---


基于上一小节提出的框架，我们可以构建三个层次，伪代码如下：



```


|  | Layer linear; |
| --- | --- |
|  | Layer sigmoid; |
|  | Layer logit; |


```

下面我们来详细讨论如何基于该框架进行计算。


## **前向传播**



```


|  | MatrixXd Layer::forward(MatrixXd input) { |
| --- | --- |
|  | cache = input; // 保存输入 |
|  | return input * para; // 对输入做计算并返回；这里给了一个示例 |
|  | } |


```

1. `z=linear.forward(X)`：线性层对输入X进行评估，并在其内存`cache`中保存X
2. `hat_y=sigmoid.forward(z)`：激活函数对输入z进行变换，得到类别概率（预测）y^，同时在其内存`cache`中保存z
3. `loss = logit.forward(y, hat_y)`：损失函数依据真实值y和预测值y^估计模型误差.



> 可以注意到，loss具有两个输入，一个是前向传播过程中的计算值，一个是真实结果。



> 从**多态性**的角度，虽然行为上有大量的一致性，但是由于输入参数数量不一致，其很难和普通层以及激活函数一样从基本层类派生，而是需要另外定义基类。


### 反向传播




---


反向传播的示例代码如下：



```


|  | MatrixXd Layer::backward(MatrixXd prevGrad) { |
| --- | --- |
|  | MatrixXd grad = ...;      // 采用cache中内容计算梯度 |
|  | cache = grad * prevGrad;  // 梯度累乘，同时保存进内存 |
|  | return cache;             // 返回梯度，以供下一层进行计算 |
|  | } |


```

1. `grad = logit.backward(y, hat_y)`：依据公式计算损失的梯度∂LOSS/∂y^。
2. `grad = sigmoid.backward(grad)`：首先计算梯度∂y^/∂z，其中，计算所需的z从缓存中读取。最后，将梯度与输入的梯度相乘，得到梯度∂LOSS/∂z。
3. `grad = linear.backward(grad)`：首先计算梯度∂z/∂w，计算所需的X从缓存中读取。最后，将梯度与输入梯度相乘，得到梯度∂LOSS/∂w。


### 参数更新




---


参数更新的示例代码如下：



```


|  | MatrixXd Layer::update(double learning_rate) { |
| --- | --- |
|  | w -= learning_rate * cache; // 采用缓存中存储的梯度进行参数更新 |
|  | } |


```

在本例中，只有`linear`是具有参数的层，因此在更新的时候只用调用`linear`的`update()`方法即可。


### 总结




---


在本小节中，我们以**逻辑回归**为例，初步探索了神经网络的构建方法，即：定义层类型，用以表示网络的每一层（或组件），在计算过程中，分为**前向传播**（推理及评估模型误差），**反向传播**（计算梯度）以及**参数更新**三个步骤。


该实现方法很好的将复杂的计算拆分为了多个独立的单元，便于较大体量的算法的实现、并且提供了极好可维护性。同时，对内存`cache`的合理使用，也极大的简化了调用过程，并提升了算法效率。


## 网络框架构建




---


神经网络是由多个组件一层层组件起来的，或者说组建神经网络的基本单元是**层**`Layer`，在本文中，将详细讲述怎么设计并实现**层**单元。


### 层的定义




---


层通常包含三种基本方法（行为）：**前向传播（forward）**、**反向传播（backward）**和**参数更新（update）**。


对于每种行为，不同类型的层所对同一操作所执行的行为不同，例如，在前向传播过程中，线性层对输入进行线性映射，激活函数则将每一元素映射至\[0,1]。


C\+\+中，允许在基类中定义方法，在派生类中重载这些方法，并在运行时根据实际类型来调整调用的函数。这种方法称之为**多态性（Polymorphism）**。


下面给出了层的定义（基类）。



```


|  | class Layer { |
| --- | --- |
|  | public: |
|  | virtual MatrixXd forward(const MatrixXd& input) = 0; |
|  | virtual MatrixXd backward(const MatrixXd& prevGrad) = 0; |
|  | virtual void update(double learning_rate) = 0; |
|  | virtual ~Layer() {} |
|  | }; |


```

在层的声明中，定义了层的三种基本方法。


### 线性层




---


对于线性层，其包含两个**参数**：权重矩阵`W`，和偏置`b`。其包含三个内存，一个用于存储输入`inputCache`，另外两个用于存储参数的变化量`dW`和`db`（一般是损失函数关于参数的梯度）。下面给出代码：



```


|  | class Linear : public Layer { |
| --- | --- |
|  | private: |
|  | // para |
|  | MatrixXd W; |
|  | MatrixXd b; |
|  | // cache |
|  | MatrixXd inputCache; |
|  | MatrixXd dW; |
|  | MatrixXd db; |
|  |  |
|  | public: |
|  | Linear(size_t input_D, size_t output_D); |
|  |  |
|  | MatrixXd forward(const MatrixXd& input) override; |
|  | MatrixXd backward(const MatrixXd& prevGrad) override; |
|  | void update(double learning_rate) override; |
|  | }; |


```

**构造函数**
构造函数用于初始化线性层的参数和缓存变量，代码如下：



```


|  | Linear::Linear(size_t input_D, size_t output_D) |
| --- | --- |
|  | : W(MatrixE::Random(input_D, output_D)), b(MatrixE::Random(1, output_D)) |
|  | { |
|  | inputCache = MatrixE::Constant(1, input_D, 0); |
|  | dW = MatrixE::Constant(input_D, output_D, 0); |
|  | db = MatrixE::Constant(1, output_D, 0); |
|  | }; |


```

1. **构造函数的参数**：


	* `input_D`：输入特征的维度，即输入数据的特征数量。
	* `output_D`：输出特征的维度，即线性层输出的特征数量。
2. **成员初始化列表**：
将参数`W`和`b`初始化为指定尺寸的随机矩阵。随机矩阵中的元素通常从标准正态分布中采样。
3. **缓存变量的初始化**：
将缓存变量初始化为与其尺寸匹配的全零矩阵。对于输入缓存，由于输入尺寸未知，仅知其特征维度，故初始化时设定其尺寸为1。


**前向传播**
下述为线性层前向传播计算的代码实现。



```


|  | MatrixXd ML::Linear::forward(const MatrixXd& input) { |
| --- | --- |
|  | // 缓存输入 |
|  | this->inputCache = input; |
|  | // 返回计算结果 |
|  | return (input * W) + b.replicate(input.rows(), 1); |
|  | } |


```

在计算过程中，`input * W`表示矩阵乘法，其中`input`是`m x n`矩阵，`W`是`n x o`矩阵，结果为`m x o`矩阵，表示线性变换的结果。


`b`是一个 `o x 1` 的偏置向量，通过`b.replicate(input.rows(), 1)`将其沿行方向复制`m`次，生成一个`m x o`的矩阵，为每个样本添加相同的偏置项。


最终，将线性变换结果与偏置矩阵相加，得到输出矩阵。


**反向传播**
下述为线性层方向传播计算的代码实现。



```


|  | MatrixXd ML::Linear::backward(const MatrixXd& prevGrad) { |
| --- | --- |
|  | // 计算关于权重的梯度 |
|  | this->dW = inputCache.transpose() * prevGrad; |
|  | // 计算关于偏置的梯度 |
|  | this->db = prevGrad.colwise().sum(); |
|  |  |
|  | // 计算并返回关于输入的梯度 |
|  | return prevGrad * W.transpose(); |
|  | } |


```

该函数接受前一层的梯度`prevGrad`，并根据缓存的输入矩阵`inputCache`计算损失关于本层参数的偏导（分别计算`dW`和`db`）并保存以用于参数更新。最后，返回关于输入的梯度矩阵。


### 激活函数




---


激活函数是比较特殊的层函数，其不包含任何的参数信息，因此，其无需进行参数更新，或者说参数更新函数不做任何操作；进一步的，激活函数也不需要在缓存中存储梯度信息，因为它无需更新参数。


下面给出较经典的三种激活函数的定义，它们都是从`Layer`中派生的：


**ReLU**激活函数



```


|  | class ReLU : public Layer { |
| --- | --- |
|  | private: |
|  | MatrixXd inputCache; |
|  |  |
|  | public: |
|  | MatrixXd forward(const MatrixXd& input) override; |
|  | MatrixXd backward(const MatrixXd& prevGrad) override; |
|  | void update(double learning_rate) override {} |
|  | }; |
|  |  |
|  | MatrixXd ReLU::forward(const MatrixXd& input) { |
|  | inputCache = input; |
|  | return input.unaryExpr([](double x) { return x > 0 ? x : 0; }); |
|  | } |
|  |  |
|  | MatrixXd ReLU::backward(const MatrixXd& prevGrad) { |
|  | MatrixXd derivative = inputCache.unaryExpr([](double x) { return x > 0 ? 1 : 0; }).cast<double>().matrix(); |
|  | return prevGrad.cwiseProduct(derivative); |
|  | } |


```

**Sigmoid激活函数**



```


|  | class Sigmoid : public Layer { |
| --- | --- |
|  | private: |
|  | MatrixXd inputCache; |
|  |  |
|  | public: |
|  | MatrixXd forward(const MatrixXd& input) override; |
|  | MatrixXd backward(const MatrixXd& prevGrad) override; |
|  | void update(double learning_rate) override {} |
|  | }; |
|  |  |
|  | MatrixXd Sigmoid::forward(const MatrixXd& input) { |
|  | inputCache = input; |
|  | return input.unaryExpr([](double x) { return 1.0 / (1.0 + exp(-x)); }); |
|  | } |
|  |  |
|  | MatrixXd Sigmoid::backward(const MatrixXd& prevGrad) { |
|  | MatrixXd sigmoidOutput = inputCache.unaryExpr([](double x) { return 1.0 / (1.0 + exp(-x)); }); |
|  | return prevGrad.cwiseProduct(sigmoidOutput.cwiseProduct((1 - sigmoidOutput.array()).matrix())); |
|  | } |


```

**tanh激活函数**



```


|  | class Tanh : public Layer { |
| --- | --- |
|  | private: |
|  | MatrixXd inputCache; |
|  |  |
|  | public: |
|  | MatrixXd forward(const MatrixXd& input) override; |
|  | MatrixXd backward(const MatrixXd& prevGrad) override; |
|  | void update(double learning_rate) override {} |
|  | }; |
|  |  |
|  | MatrixXd Tanh::forward(const MatrixXd& input) { |
|  | inputCache = input; |
|  | return input.unaryExpr([](double x) { return tanh(x); }); |
|  | } |
|  |  |
|  | MatrixXd Tanh::backward(const MatrixXd& prevGrad) { |
|  | MatrixXd tanhOutput = inputCache.unaryExpr([](double x) { return tanh(x); }); |
|  | return prevGrad.cwiseProduct((1 - tanhOutput.cwiseProduct(tanhOutput).array()).matrix()); |
|  | } |


```

### 损失函数


损失函数是一种特殊的层，其行为与普通的层相似，但在前向传播和反向传播过程中与普通的层存在较大差异：


* **前向传播**：从使用来看，如果仅需使用网络，而无需评价，或者训练网络，前向传播过程中，无需调用损失函数。从输入个数来看，普通层函数只需要传入上一层的输出即可，而损失函数需要上一层的输出（也即：网络的预测结果）和真实结果两个参数。
* **反向传播**：损失函数是反向传播的起点，其接受预测结果和真实结果作为其输入，返回梯度矩阵，而其它矩阵则是输入上一层的梯度矩阵，返回本层的梯度矩阵。


由于损失函数与普通层的差异，一般单独定义损失函数的调用接口（基类），代码如下：



```


|  | class LossFunction { |
| --- | --- |
|  | public: |
|  | virtual double computeLoss(const MatrixXd& predicted, const MatrixXd& actual) = 0; |
|  | virtual MatrixXd computeGradient(const MatrixXd& predicted, const MatrixXd& actual) = 0; |
|  | virtual ~LossFunction() {} |
|  | }; |


```

下文分别以均方根误差和对数损失为例，给出代码，供读者参考学习


**MSE均方根误差**



```


|  | class MSELoss : public LossFunction { |
| --- | --- |
|  | public: |
|  | double computeLoss(const MatrixXd& predicted, const MatrixXd& actual) override; |
|  | MatrixXd computeGradient(const MatrixXd& predicted, const MatrixXd& actual) override; |
|  | }; |
|  |  |
|  | double MSELoss::computeLoss(const MatrixXd& predicted, const MatrixXd& actual) { |
|  | MatrixXd diff = predicted - actual; |
|  | return diff.squaredNorm() / (2.0 * predicted.rows()); |
|  | } |
|  |  |
|  | MatrixXd MSELoss::computeGradient(const MatrixXd& predicted, const MatrixXd& actual) { |
|  | MatrixXd diff = predicted - actual; |
|  | return diff / predicted.rows(); |
|  | } |


```

**对数损失函数**



```


|  | class LogisticLoss : public LossFunction { |
| --- | --- |
|  | public: |
|  | double computeLoss(const MatrixXd& predicted, const MatrixXd& actual) override; |
|  | MatrixXd computeGradient(const MatrixXd& predicted, const MatrixXd& actual) override; |
|  | }; |
|  |  |
|  | double LogisticLoss::computeLoss(const MatrixXd& predicted, const MatrixXd& actual) { |
|  | MatrixXd log_predicted = predicted.unaryExpr([](double p) { return log(p); }); |
|  | MatrixXd log_1_minus_predicted = predicted.unaryExpr([](double p) { return log(1 - p); }); |
|  |  |
|  | MatrixXd term1 = actual.cwiseProduct(log_predicted); |
|  | // MatrixXd term2 = (1 - actual).cwiseProduct(log_1_minus_predicted); |
|  | MatrixXd term2 = (1 - actual.array()).matrix().cwiseProduct(log_1_minus_predicted); |
|  |  |
|  | double loss = -(term1 + term2).mean(); |
|  |  |
|  | return loss; |
|  | } |
|  |  |
|  | MatrixXd LogisticLoss::computeGradient(const MatrixXd& predicted, const MatrixXd& actual) { |
|  | MatrixXd temp1 = predicted - actual; |
|  | MatrixXd temp2 = predicted.cwiseProduct((1 - predicted.array()).matrix()); |
|  |  |
|  | return (temp1).cwiseQuotient(temp2); |
|  | } |


```

 本博客参考[樱花宇宙官网](https://yzygzn.com)。转载请注明出处！
