+++
title = '知识追踪任务的公平性比较与数据泄露分析'
date = 2025-05-08T12:23:47-07:00
draft = false
+++

## RouterKT 的数据泄露

最近完成了一份工作，借鉴 MoE 的思路去正交化 Knowledge Tracing 里的并行化模块，期望是不同的 Attention Head 应当关注到不同的特征，比如有的 Attention Head 应当关注到近期的内容，有的应当关注到远期的内容。

然而现有的很多模型都会给所有的 Attention Head 加同样的 forgetting decay，也就是给注意力机制以各种方式加上距离相关的decay，迫使模型更多关注最近的交互。然而，这可能会损坏模型对其他模式的关注，比如 spacing effects，也就是系统其实是会周期性安排复习的，如果我们更多关注最近的交互，会很容易忽略掉这些模式。

借鉴 MoE 的思路，我们正交化并行的 Multi Head Attention，也就是加上一个路由损失：

如果用 $p_{i,e}$ 表示第 i 个样本被路由到专家 e 的概率，那么负载均衡损失可以表示为：

$$L_{balance} = \alpha \cdot N \sum_{e=1}^E (\frac{1}{N}\sum_{i=1}^N p_{i,e} - \frac{1}{E})^2$$

其中 N 是批次大小，E 是专家数量，α 是权重系数。

听起来很 work 对不对？

我的实现如下：

```python
def forward(self, q, k, v, mask, zero_pad, question_difficulty_emb, q4router=None):
    model_config = self.params["models_config"]["RouterKT"]
    dim_model = model_config["dim_model"]
    num_head = model_config["num_head"]
    dim_head = dim_model // num_head
    
    batch_size = q.size(0)
    
    # Linear projections
    k = self.key_linear(k).view(batch_size, -1, num_head, dim_head)
    if self.key_query_same:
        q = self.key_linear(q).view(batch_size, -1, num_head, dim_head)
    else:
        q = self.query_linear(q).view(batch_size, -1, num_head, dim_head)
    v = self.value_linear(v).view(batch_size, -1, num_head, dim_head)
    
    # Transpose for attention computation
    k = k.transpose(1, 2)
    q = q.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # Calculate routing scores for dynamic heads
    # Always use question information for routing
    q4router = q4router.view(batch_size, q4router.size(1), num_head, dim_head)
    q_for_routing = q4router.permute(0, 2, 1, 3).reshape(batch_size * q4router.size(1), num_head * dim_head)
    logits = self.wg(q_for_routing)  # [bs*seq_len, n_dynamic_heads]
    gates = F.softmax(logits, dim=1)  # [bs*seq_len, n_dynamic_heads]
    
    # Select top-k heads
    _, indices = torch.topk(gates, k=self.n_selected_heads, dim=1)
    dynamic_mask = torch.zeros_like(gates).scatter_(1, indices, 1.0)
    
    # Update routing statistics
    self.head_routing_probs = gates.mean(dim=0)
    self.head_selections = dynamic_mask.sum(dim=0)
    
    # Create routing mask
    dynamic_scores_reshaped = (gates * dynamic_mask).view(batch_size, q4router.size(1), -1)
    routing_mask = torch.zeros(batch_size, q4router.size(1), num_head).to(q4router.device)
    routing_mask[:, :, :self.n_shared_heads] = 1.0  # Shared heads always active
    routing_mask[:, :, self.n_shared_heads:] = dynamic_scores_reshaped  # Add dynamic head weights
    
    # Reshape routing mask to match attention dimensions
    routing_mask = routing_mask.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
    
    # Calculate attention using the attention function
    scores = attention4router_kt(q, k, v, dim_head, mask, self.dropout, zero_pad, routing_mask, device=self.params["device"])
    
    # Combine heads
    concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, dim_model)
    
    return self.out_proj(concat)
```

基本的逻辑就是用一个线性层去学习如何Drop掉一部分注意力头，然后用软聚合的方式，这个思想来自于 MHA。之前很多其他领域的实验证明多头注意力机制里的大部分头都是冗余的。这件事情也的确在知识追踪任务里成立，我尝试把 AKT 和 SimpleKT 的注意力头数量设置为 1，实验证明没有任何性能下降。

```python
# Knowledge encoder
for block in self.blocks_1:
    y, _ = block(mask=1, query=y, key=y, values=y, diff=diff, response=r, q4router=x)
    
# Question encoder
flag_first = True
for block in self.blocks_2:
    if flag_first:
        # x can see both current and past information
        x, _ = block(mask=1, query=x, key=x, values=x, diff=diff, response=r, apply_pos=False, q4router=x)
        flag_first = False
    else:# dont peek current response
        # knoweldge retriever
        # h can see past only
        x, attn = block(mask=0, query=x, key=x, values=y, diff=diff, response=r, apply_pos=True, q4router=x)
        flag_first = True
        
return x, attn
```

然后取得了非常好的效果
| Model            | algebra05 AUC | algebra05 RMSE | bridge06 AUC | bridge06 RMSE | assistments09 AUC | assistments09 RMSE | slepemapy AUC | slepemapy RMSE | spanish AUC | spanish RMSE | statics AUC | statics RMSE |
|------------------|---------------|----------------|---------------|----------------|--------------------|---------------------|----------------|----------------|--------------|---------------|--------------|---------------|
| IRT              | 0.7141        | 0.4005         | 0.6559        | 0.4025         | 0.6708             | 0.4631              | 0.6210         | 0.4068         | 0.6956       | 0.4596        | 0.7404       | 0.4303        |
| PFA              | 0.7481        | 0.3932         | 0.7460        | 0.3848         | 0.7284             | 0.4444              | 0.6583         | 0.4020         | 0.7467       | 0.4428        | 0.7489       | 0.4096        |
| DKT              | 0.7636        | 0.3921         | 0.7589        | 0.3820         | 0.7504             | 0.4371              | 0.6986         | 0.3978         | 0.8066       | 0.4139        | 0.7674       | 0.4111        |
| DKVMN            | 0.7562        | 0.3907         | 0.7463        | 0.3864         | 0.7475             | 0.4375              | 0.7064         | 0.3962         | 0.8027       | 0.4156        | 0.7736       | 0.3975        |
| SAKT             | 0.7636        | 0.3899         | 0.7512        | 0.3862         | 0.7491             | 0.4381              | 0.6846         | 0.4062         | 0.8065       | 0.4179        | 0.7492       | 0.4105        |
| SparseKT         | 0.7806        | 0.3875         | 0.7694        | 0.3804         | 0.7670             | 0.4396              | 0.7255         | 0.3903         | 0.8395       | 0.3959        | 0.7887       | 0.3909        |
| ATKT             | 0.7624        | 0.3899         | 0.7426        | 0.3891         | 0.7543             | 0.4348              | 0.6952         | 0.3983         | 0.8047       | 0.4186        | 0.7421       | 0.4130        |
| CL4KT            | 0.7891        | 0.3815         | 0.7733        | 0.3791         | 0.7624             | 0.4333              | 0.7218         | 0.3926         | 0.8289       | 0.4049        | 0.7943       | 0.3945        |
| DTransformer     | 0.7694        | 0.3906         | 0.7391        | 0.3892         | 0.7508             | 0.4505              | 0.7217         | 0.3892         | 0.8170       | 0.4108        | 0.7690       | 0.3966        |
| MIKT             | 0.7912        | 0.3824         | 0.7721        | 0.3784         | 0.7693             | 0.4339              | 0.7293         | 0.3874         | 0.8374       | 0.3972        | 0.7812       | 0.3903        |
| SimpleKT         | 0.7763        | 0.3846         | 0.7656        | 0.3809         | 0.7566             | 0.4346              | 0.7150         | 0.3916         | 0.8353       | 0.3985        | 0.7692       | 0.3980        |
| FoLiBi(SimpleKT) | 0.7827        | 0.3832         | 0.7714        | 0.3799         | 0.7615             | 0.4322              | 0.7126         | 0.3344         | 0.8366       | 0.3979        | 0.7850       | 0.3922        |
| RouterSimpleKT   | 0.7844        | 0.3820         | 0.7710        | 0.3790         | 0.7716             | 0.4288              | 0.7452         | 0.3843         | 0.8429       | 0.3933        | 0.7848       | 0.3923        |
| AKT              | 0.7860        | 0.3827         | 0.7664        | 0.3819         | 0.7660             | 0.4379              | 0.7243         | 0.3894         | 0.8373       | 0.3964        | 0.7999       | 0.3889        |
| FoLiBi(AKT)      | 0.7960        | 0.3800         | 0.7805        | 0.3765         | 0.7716             | 0.4329              | 0.7284         | 0.3893         | 0.8379       | 0.3974        | 0.7988       | 0.3894        |
| RouterAKT        | 0.8011        | 0.3775         | 0.7842        | 0.3749         | 0.7816             | 0.4287              | 0.7562         | 0.3838         | 0.8519       | 0.3870        | 0.8123       | 0.3819        |


可以看到效果的确是非常不错，实现上看起来也没有太大的问题，直到这里我都没有发现问题，甚至以为性能就该好，这不就是软正交学习嘛！而且性能提升也不是说剧增，除了在部分数据集上（database, prob）等，但是这个提升幅度和 DisKT 类似，因此我欣然接受了这个结果。

直到我试图把我的模型复现到其他框架，发现了一件非常诡异的事情，在短序列上模型性能居然远强于长序列。这是非常不符合直觉的，短序列我们能够得到的信息非常少，按理说应该和瞎猜差不多。


这提示我是不是发生了数据泄露问题。于是我试着构造一个随机的 response 序列去测试，看起来一切正常，模型并不能捕捉到随机构造的标签，输出一直在 0.5 的 AUC 上下波动（still don't know why）。读者可以停下来想想，发生数据泄露是在哪一行代码？

下面是 AKT 的结构：

```python
def forward(self, batch):
    x = batch["question_emb"]
    y = batch["interaction_emb"]
    question_difficulty_emb = batch["question_difficulty_emb"]
    response = None

    # Knowledge encoder
    for block in self.knowledge_encoder:
        # 对0～t-1时刻前的qa信息进行编码, \hat{y_t}
        y = block(query=y, key=y, values=y, diff=question_difficulty_emb, apply_pos=True, mask_flag=True, q4router=y)


    flag_first = True
    for block in self.question_encoder:
        if flag_first:
            # peek current question
            # False: 没有FFN, 第一层只有self attention, \hat{x_t}
            x = block(query=x, key=x, values=x, diff=question_difficulty_emb, apply_pos=False, mask_flag=True, q4router=x)
            flag_first = False
        else:
            # don't peek current response
            # True: +FFN+残差+layer norm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
            # mask=0，不能看到当前的response, 在Knowledge Retriever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
            x = block(query=x, key=x, values=y, diff=question_difficulty_emb, apply_pos=True, mask_flag=False, q4router=x)
            flag_first = True


    return x
```

这里的 x 经过了因果掩码，所以不会泄露未来的信息。

但果真如此吗？

```python
# TransformerLayer
def forward(self, q, k, v, mask, zero_pad):
    bs = q.size(0)
    # perform linear operation and split into h heads
    k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
    if self.kq_same is False:
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
    else:
        q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
    v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
    # transpose to get dimensions bs * h * sl * embedding_size
    k = k.transpose(1, 2)
    q = q.transpose(1, 2)
    v = v.transpose(1, 2)
    # calculate attention using function we will define next
    # 经过因果掩码的 scores
    scores = attention(q, k, v, self.d_k,
                        mask, self.dropout, zero_pad)
    # concatenate heads and put through final linear layer
    concat = scores.transpose(1, 2).contiguous()\
        .view(bs, -1, self.embedding_size)

    output = self.out_proj(concat)

    return output

def attention(q, k, v, d_k, mask, dropout, zero_pad):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    device = q.device

    scores.masked_fill_(mask == 0, -1e32) # 在这里做掩码，保证 scores 只保留不带对角线的下三角的信息
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score置0
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output
```

这里的数据链路是 attention 里经过因果掩码，返回 output 作为 TransformerLayer 的 scores，然后经过一个 FFN 层得到预测结果。

```python
# Knowledge Retriver
x = block(query=x, key=x, values=y, diff=question_difficulty_emb, apply_pos=True, mask_flag=False, q4router=x)
            flag_first = True
```

然后 AKT 层叠地使用了历史的答题信息（x）。这里，我想当然地认为，x 一定是只保留历史的信息，一定不会发生数据泄露。果真如此吗？让我们仔细看看。

$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

```python
# Mixture of Attention Head
def forward(self, q, k, v, mask, zero_pad, question_difficulty_emb, q4router=None):

    # Linear projections
    k = self.key_linear(k).view(batch_size, -1, num_head, dim_head)
    if self.key_query_same:
        q = self.key_linear(q).view(batch_size, -1, num_head, dim_head)
    else:
        q = self.query_linear(q).view(batch_size, -1, num_head, dim_head)
    v = self.value_linear(v).view(batch_size, -1, num_head, dim_head)
    
    
    # Calculate routing scores for dynamic heads
    # Always use question information for routing
    q4router = q4router.view(batch_size, q4router.size(1), num_head, dim_head)
    q_for_routing = q4router.permute(0, 2, 1, 3).reshape(batch_size * q4router.size(1), num_head * dim_head)
    logits = self.wg(q_for_routing)  # [bs*seq_len, n_dynamic_heads]
    gates = F.softmax(logits, dim=1)  # [bs*seq_len, n_dynamic_heads]
    
    # Select top-k heads
    _, indices = torch.topk(gates, k=self.n_selected_heads, dim=1)
    dynamic_mask = torch.zeros_like(gates).scatter_(1, indices, 1.0)
    
    # Create routing mask
    dynamic_scores_reshaped = (gates * dynamic_mask).view(batch_size, q4router.size(1), -1)
    routing_mask = torch.zeros(batch_size, q4router.size(1), num_head).to(q4router.device)
    routing_mask[:, :, :self.n_shared_heads] = 1.0  # Shared heads always active
    routing_mask[:, :, self.n_shared_heads:] = dynamic_scores_reshaped  # Add dynamic head weights
    
    # Reshape routing mask to match attention dimensions
    routing_mask = routing_mask.mean(dim=1).unsqueeze(-1).unsqueeze(-1)

    # Calculate attention using the attention function
    scores = attention(q, k, v, dim_head, mask, self.dropout, zero_pad, routing_mask, device=self.params["device"])
    
    # Combine heads
    concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, dim_model)
    
    return self.out_proj(concat)
```
让我们再回顾一下这里的计算链路：
首先根据 query 去计算路由得分，然后把路由得分分配给注意力头，这样我们就能得到这个序列最适合的注意力头，因为我们给router的是x，x已经被掩码过了，所以不会泄露历史信息。

但真的不会泄露吗？我们再来想想，即使我们传入的是 x 也就是被掩码过的信息（这是一个下三角矩阵），但是我们对路由策略做了平均：
```python
routing_mask = routing_mask.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
```
因为我们期望做 task-level routing[1]。 如果做 token-wise routing，整个序列会被拆散分发到不同的注意力头，注意力头之间就只能通过 FFN 层去交换信息，做 task-level routing， 就可以根据序列全局的特性去路由到合适的注意力头，比如有的头可能更关注重要的题目，有的头可能更关注近期的信息等等。这听起来是不是 motivation 还挺好的。

坏就坏在这个全局路由。我们当前的 Token 是不可以知道未来 Token 的任何信息的，包括它可以被路由到哪个注意力头！所以包含 label 信息的 task-level routing 是不可以被用在自回归任务上的！这里需要改为：

```
routing_mask = routing_mask.permute(0, 2, 1).unsqueeze(-1)
```

当我们把这个部分修正，就只能得到和 AKT 差不多的性能。这里也提示我，一个非常值得注意的点是，自回归任务里平均池化、最大最小池化是非常需要注意的。事实上，我在开始想办法把 MoE 的思想融入知识追踪任务的时候，就在考虑是否存在数据泄露，因为性能提升很大，显得非常异常。自查过几次代码，也询问过 Chadiskt-result-v2PT、Claude、Gemini 等 ChatBot 他们都说没问题...

![Data Leakage](/rethinking-kt/data-leakage-kt.png)


## DisKT 的数据泄露[2] [Github Issue](https://github.com/zyy-2001/DisKT/issues/2)
上文提到， RouterKT 的性能提升和 DisKT 类似，要知道，RouterKT 本身是发生数据泄露了，性能提升还和 DisKT类似，这不是很诡异吗？于是我尝试在 DisKT 的框架下进行数据泄露的检验，果然发现了问题！

我首先构造了随机序列：

![Data Leakage](/rethinking-kt/diskt-dl.png)

这里显然是不合理的，对于一个随机序列本应 AUC 是 0.5 上下。 对于 DisKT，我们发现他能学到随机序列上的信息，那就一定有问题，于是我对他们的每一个组件进行了消融实验，最后发现是在这里：

```python
def contradictory_attention(query, key, value1, value2, mask=None, dropout=None, counter_attention_mask=None):
    "Compute 'Scaled Dot Product Attention'"
    '''
    query: [batch_size, head, seq_len, feature]
    '''
    bs, head, seqlen, d_k = query.size(0), query.size(1), query.size(2), query.size(-1)
    device = query.device
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e32)
    p_attn = F.softmax(scores, dim = -1) # [batch_size, head, seq_len, seq_len]

    # Reallocate attention weights, making the positions of mistakes and guesses receive less attention weight
    attn_reshape = p_attn.reshape(bs*head*seqlen, -1)
    counter_attention_mask = counter_attention_mask.unsqueeze(1)
    counter_attention_mask = counter_attention_mask.expand(-1, head*seqlen, -1)
    counter_attention_mask = counter_attention_mask.reshape(-1, seqlen)

    # attn_reshape = attn_reshape * counter_attention_mask
    p_attn = torch.where((counter_attention_mask == 1), torch.zeros_like(attn_reshape), attn_reshape)

    p_attn = p_attn.reshape(bs, head, seqlen, -1)

    # 此处对 p_attn 重新分配了 softmax，归一化后，原先被 mask 掉的元素会重新获得注意力
    p_attn = F.softmax(p_attn, dim = -1)

    pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)

    p_attn = torch.cat([pad_zero, p_attn[:, :, 1:, :]], dim=2) # 第一行score置0
    if dropout is not None:
        p_attn = dropout(p_attn)

    output_v1 = torch.matmul(p_attn, value1)
    output_v2 = torch.matmul(p_attn, value2)
    return output_v1, output_v2, p_attn
```

这个问题 sparsekt 也同样发生过：[sparseKT-ktop算法存在的问题](https://github.com/pykt-team/pykt-toolkit/issues/165)

解决方案是，要么去掉第二次的softmax，如果担心遮住一部分元素之后，概率相加不为 1了，可以修复为：

```python
def contradictory_attention(query, key, value1, value2, mask=None, dropout=None, counter_attention_mask=None):
    "Compute 'Scaled Dot Product Attention'"
    bs, head, seqlen, d_k = query.size(0), query.size(1), query.size(2), query.size(-1)
    device = query.device
    
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e32)
    
    p_attn = F.softmax(scores, dim=-1)  # [batch_size, head, seq_len, seq_len]
    
    counter_attention_mask = counter_attention_mask.unsqueeze(1)
    counter_attention_mask = counter_attention_mask.expand(-1, head*seqlen, -1)
    counter_attention_mask = counter_attention_mask.reshape(-1, seqlen)
    
    attn_reshape = p_attn.reshape(bs*head*seqlen, -1)
    
    masked_attn = torch.where((counter_attention_mask == 1), 
                             torch.zeros_like(attn_reshape), 
                             attn_reshape)
    
    masked_attn = masked_attn.reshape(bs, head, seqlen, -1)
    
    row_sums = masked_attn.sum(dim=-1, keepdim=True)
    
    valid_rows = (row_sums > 0).float()
    safe_row_sums = row_sums + (1 - valid_rows)
    
    normalized_attn = masked_attn / safe_row_sums
    
    pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
    normalized_attn = torch.cat([pad_zero, normalized_attn[:, :, 1:, :]], dim=2)
    
    if dropout is not None:
        normalized_attn = dropout(normalized_attn)
    
    output_v1 = torch.matmul(normalized_attn, value1)
    output_v2 = torch.matmul(normalized_attn, value2)
    
    return output_v1, output_v2, normalized_attn
```

也就是手动屏蔽一下不应该获得注意力的元素。我在修复后的版本上重跑了 DisKT 的实验：

<table class="diskt-result">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>DisKT✓</th>
      <th>DisKT-Fixed</th>
      <th>simpleKT</th>
      <th>数据泄露的收益</th>
      <th>与SimpleKT 相比提升</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>assist09</td>
      <td>0.7923</td>
      <td>0.77351</td>
      <td>0.7709</td>
      <td>1.88%</td>
      <td>0.26%</td>
    </tr>
    <tr>
      <td>algebra05</td>
      <td>0.8033</td>
      <td>0.7896</td>
      <td>0.7874</td>
      <td>1.37%</td>
      <td>0.22%</td>
    </tr>
    <tr>
      <td>algebra06</td>
      <td>0.7846</td>
      <td>0.77015</td>
      <td>0.7695</td>
      <td>1.45%</td>
      <td>0.07%</td>
    </tr>
    <tr>
      <td>ednet</td>
      <td>0.7384</td>
      <td>0.70105</td>
      <td>0.7048</td>
      <td>3.74%</td>
      <td>-0.38%</td>
    </tr>
    <tr>
      <td>prob</td>
      <td>0.7731</td>
      <td>0.73794</td>
      <td>0.7265</td>
      <td>3.52%</td>
      <td>1.14%</td>
    </tr>
    <tr>
      <td>linux</td>
      <td>0.8622</td>
      <td>0.82169</td>
      <td>0.8221</td>
      <td>4.05%</td>
      <td>-0.04%</td>
    </tr>
    <tr>
      <td>comp</td>
      <td>0.8324</td>
      <td>0.80098</td>
      <td>0.8000</td>
      <td>3.14%</td>
      <td>0.10%</td>
    </tr>
    <tr>
      <td>database</td>
      <td>0.8769</td>
      <td>0.82688</td>
      <td>0.8272</td>
      <td>5.00%</td>
      <td>-0.03%</td>
    </tr>
    <tr>
      <td>slepemapy</td>
      <td>0.7632</td>
      <td>0.72501</td>
      <td>0.7269</td>
      <td>3.82%</td>
      <td>-0.19%</td>
    </tr>
  </tbody>
</table>

作者提醒，上面的实现并没考虑到第二次 Softmax 的非线性能力，这的确是一个问题，因此我提出了新的修复方案：
```python
def contradictory_attention(query, key, value1, value2, mask=None, dropout=None, counter_attention_mask=None):
    bs, head, seqlen, d_k = query.size(0), query.size(1), query.size(2), query.size(-1)
    device = query.device
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e32)
    
    p_attn = F.softmax(scores, dim = -1)  # [batch_size, head, seq_len, seq_len]
    
    # 准备counter_attention_mask，使其形状匹配p_attn
    expanded_mask = counter_attention_mask.unsqueeze(1).unsqueeze(1)  # [bs, 1, 1, seqlen]
    expanded_mask = expanded_mask.expand(-1, head, seqlen, -1)  # [bs, head, seqlen, seqlen]
    
    # 直接在原始维度应用mask
    LOG_MIN = -1e32
    masked_attn = torch.where(expanded_mask == 1, 
                             torch.ones_like(p_attn) * LOG_MIN, 
                             torch.log(p_attn + 1e-10))
    
    p_attn = F.softmax(masked_attn, dim = -1)
    
    pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
    p_attn = torch.cat([pad_zero, p_attn[:, :, 1:, :]], dim=2)
    
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    output_v1 = torch.matmul(p_attn, value1)
    output_v2 = torch.matmul(p_attn, value2)
    return output_v1, output_v2, p_attn
```
这里还是保留了第二次 Softmax 的非线性能力，不过避免了因果掩码的丢失。

<table class="diskt-result-v2"><thead>
  <tr>
    <th class="diskt-result-v2-0lax">Dataset</th>
    <th class="diskt-result-v2-0lax">DisKT</th>
    <th class="diskt-result-v2-0lax">DisKT-第一版Fixed</th>
    <th class="diskt-result-v2-0lax">DisKT-Fixed</th>
    <th class="diskt-result-v2-0lax">simpleKT</th>
    <th class="diskt-result-v2-0lax">AKT</th>
    <th class="diskt-result-v2-0lax">数据泄露的收益</th>
    <th class="diskt-result-v2-0lax">与SimpleKT 相比提升</th>
    <th class="diskt-result-v2-0lax">与AKT 相比提升</th>
    <th class="diskt-result-v2-0lax">两版 Fix 差距</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="diskt-result-v2-0lax">assist09</td>
    <td class="diskt-result-v2-0lax">0.7923</td>
    <td class="diskt-result-v2-0lax">0.7735</td>
    <td class="diskt-result-v2-0lax">0.7725</td>
    <td class="diskt-result-v2-0lax">0.7709</td>
    <td class="diskt-result-v2-0lax">0.7705</td>
    <td class="diskt-result-v2-0lax">1.98%</td>
    <td class="diskt-result-v2-0lax">0.16%</td>
    <td class="diskt-result-v2-0lax">0.20%</td>
    <td class="diskt-result-v2-0lax">-0.10%</td>
  </tr>
  <tr>
    <td class="diskt-result-v2-0lax">algebra05</td>
    <td class="diskt-result-v2-0lax">0.8033</td>
    <td class="diskt-result-v2-0lax">0.7896</td>
    <td class="diskt-result-v2-0lax">0.7904</td>
    <td class="diskt-result-v2-0lax">0.7874</td>
    <td class="diskt-result-v2-0lax">0.7932</td>
    <td class="diskt-result-v2-0lax">1.29%</td>
    <td class="diskt-result-v2-0lax">0.30%</td>
    <td class="diskt-result-v2-0lax">-0.28%</td>
    <td class="diskt-result-v2-0lax">0.08%</td>
  </tr>
  <tr>
    <td class="diskt-result-v2-0lax">prob</td>
    <td class="diskt-result-v2-0lax">0.7731</td>
    <td class="diskt-result-v2-0lax">0.7379</td>
    <td class="diskt-result-v2-0lax">0.7357</td>
    <td class="diskt-result-v2-0lax">0.7265</td>
    <td class="diskt-result-v2-0lax">0.7376</td>
    <td class="diskt-result-v2-0lax">3.74%</td>
    <td class="diskt-result-v2-0lax">0.92%</td>
    <td class="diskt-result-v2-0lax">-0.19%</td>
    <td class="diskt-result-v2-0lax">-0.23%</td>
  </tr>
  <tr>
    <td class="diskt-result-v2-0lax">slepemapy</td>
    <td class="diskt-result-v2-0lax">0.7632</td>
    <td class="diskt-result-v2-0lax">0.7250</td>
    <td class="diskt-result-v2-0lax">0.7249</td>
    <td class="diskt-result-v2-0lax">0.7269</td>
    <td class="diskt-result-v2-0lax">0.7258</td>
    <td class="diskt-result-v2-0lax">3.83%</td>
    <td class="diskt-result-v2-0lax">-0.20%</td>
    <td class="diskt-result-v2-0lax">-0.09%</td>
    <td class="diskt-result-v2-0lax">-0.01%</td>
  </tr>
</tbody></table>

此外，我还做了验证性实验：
<table class="diskt-result-v3"><thead>
  <tr>
    <th class="diskt-result-v3-0pky">seq len</th>
    <th class="diskt-result-v3-0pky">SimpleKT</th>
    <th class="diskt-result-v3-0pky">SparseKT</th>
    <th class="diskt-result-v3-0pky">DisKT</th>
    <th class="diskt-result-v3-0pky">DisKT-fixed</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="diskt-result-v3-0pky">2</td>
    <td class="diskt-result-v3-0pky">0.6622</td>
    <td class="diskt-result-v3-0pky">0.6682</td>
    <td class="diskt-result-v3-0pky">0.9950</td>
    <td class="diskt-result-v3-0pky">0.7678</td>
  </tr>
  <tr>
    <td class="diskt-result-v3-0pky">10</td>
    <td class="diskt-result-v3-0pky">0.6878</td>
    <td class="diskt-result-v3-0pky">0.6878</td>
    <td class="diskt-result-v3-0pky">0.8201</td>
    <td class="diskt-result-v3-0pky">0.6834</td>
  </tr>
  <tr>
    <td class="diskt-result-v3-0pky">50</td>
    <td class="diskt-result-v3-0pky">0.7382</td>
    <td class="diskt-result-v3-0pky">0.7457</td>
    <td class="diskt-result-v3-0pky">0.7933</td>
    <td class="diskt-result-v3-0pky">0.7409</td>
  </tr>
  <tr>
    <td class="diskt-result-v3-0pky">100</td>
    <td class="diskt-result-v3-0pky">0.7350</td>
    <td class="diskt-result-v3-0pky">0.7469</td>
    <td class="diskt-result-v3-0pky">0.7725</td>
    <td class="diskt-result-v3-0pky">0.7357</td>
  </tr>
  <tr>
    <td class="diskt-result-v3-0pky">200</td>
    <td class="diskt-result-v3-0pky">0.7295</td>
    <td class="diskt-result-v3-0pky">0.7301</td>
    <td class="diskt-result-v3-0pky">0.7474</td>
    <td class="diskt-result-v3-0pky">0.7324</td>
  </tr>
</tbody></table>
可以看到，随着在很少量的 seq length 的时候， DisKT 能达到接近 100% 的 TEST AUC，这是明显不合理的。

| setting            | 随机初始化标签 |
|--------------------|----------------|
| with early stop    | 56.19%         |
| without early stop | 100.00%        |

同时测试了随机初始化下带早停和不带早停的情况。

# PyKT的问题准确率[3]
我在 pykt 做随机标签检测，发现他们的指标也有异常。[PyKT - 关于在随机数据上的训练问题](https://github.com/pykt-team/pykt-toolkit/issues/245)

![PyKT](/rethinking-kt/pykt-balance.png)

这里稍微解释一下，这里的

```
{'testauc': 0.6438930956398151, 'testacc': 0.5795748255363143, 'window_testauc': 0.644660752721687, 'window_testacc': 0.5804480980012895, 'oriaucconcepts': 0.5033722608280454, 'oriauclate_mean': 0.5028598981901652, 'oriauclate_vote': 0.5031274181458916, 'oriauclate_all': 0.5032359821913478, 'oriaucearly_preds': 0.502985136956939, 'oriaccconcepts': 0.5033216669907318, 'oriacclate_mean': 0.5021625138745359, 'oriacclate_vote': 0.5030619665480155, 'oriacclate_all': 0.5027174953113637, 'oriaccearly_preds': 0.502181651165461, 'windowaucconcepts': 0.5035561220606548, 'windowauclate_mean': 0.502587681942156, 'windowauclate_vote': 0.503010394236416, 'windowauclate_all': 0.5036517308862161, 'windowaucearly_preds': 0.5030404483146422, 'windowaccconcepts': 0.5037813903881581, 'windowacclate_mean': 0.5019855664592004, 'windowacclate_vote': 0.502806521822139, 'windowacclate_all': 0.5029974416739853, 'windowaccearly_preds': 0.5021192103554928}
```

testauc是pykt提出的在题目级别进行预测（如果我没猜错的话），所有需要聚合的（比如oriauclate_mean, windowaucearly_preds），是在知识点级别进行预测。

这里也是非常诡异，按理说即使是在题目级别进行预测，我们的结果也应当是 0.5，而不应该是 0.66附近（其实就是2/3），这可能是 PyKT 框架的一个结构性偏差，可能也是为什么他们报的题目级别的指标异常的高。

## conclusion
我们发现，RouterKT 和 DisKT 在有数据泄露的情况下，在 algebra2005 这个数据集上的性能提升都非常有限，如果后面看到明显高于 AKT 在 algebra2005 上的结果，就得仔细看看了...

## reference
[1] Kudugunta S, Huang Y, Bapna A, et al. Beyond distillation: Task-level mixture-of-experts for efficient inference[J]. arXiv preprint arXiv:2110.03742, 2021.

[2] Zhou Y, Lv Z, Zhang S, et al. Disentangled Knowledge Tracing for Alleviating Cognitive Bias[C]//Proceedings of the ACM on Web Conference 2025. 2025: 2633-2645.

[3] Liu Z, Liu Q, Chen J, et al. pyKT: a python library to benchmark deep learning based knowledge tracing models[J]. Advances in Neural Information Processing Systems, 2022, 35: 18542-18555.

<style>
.diskt-result td:nth-child(2),
.diskt-result th:nth-child(2) {
    background-color: var(--red);
}

.diskt-result td:nth-child(3),
.diskt-result th:nth-child(3) {
    background-color: var(--green);
}

.diskt-result-v2 td:nth-child(2),
.diskt-result-v2 th:nth-child(2) {
    background-color: var(--red);
}

.diskt-result-v2 td:nth-child(3),
.diskt-result-v2 th:nth-child(3) {
    background-color: var(--green);
}

.diskt-result-v2 td:nth-child(4),
.diskt-result-v2 th:nth-child(4) {
    background-color: var(--green);
}

.diskt-result-v2 td:nth-child(7),
.diskt-result-v2 th:nth-child(7) {
    background-color: var(--red);
}

.diskt-result-v3 td:nth-child(4),
.diskt-result-v3 th:nth-child(4) {
    background-color: var(--red);
}

.diskt-result-v3 td:nth-child(5),
.diskt-result-v3 th:nth-child(5) {
    background-color: var(--green);
}
</style>