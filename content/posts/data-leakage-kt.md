+++
title = '知识追踪任务的公平性比较与数据泄露分析'
date = 2025-05-08T12:23:47-07:00
draft = true
+++

## RouterKT 的数据泄露

在最近的一项工作中，我们借鉴了 MoE (Mixture of Experts) 的思路来优化 Knowledge Tracing 任务中的注意力机制。主要思路是让不同的 Attention Head 专注于不同的特征，例如近期和远期的学习行为。

目前的主流模型普遍会为所有 Attention Head 添加相同的遗忘衰减(forgetting decay)，这种做法可能会限制模型捕捉其他重要模式的能力，比如间隔效应(spacing effects)。为了解决这个问题，我们引入了路由损失来正交化并行的多头注意力机制：

$$L_{balance} = \alpha \cdot N \sum_{e=1}^E (\frac{1}{N}\sum_{i=1}^N p_{i,e} - \frac{1}{E})^2$$

其中:
- N 是批次大小
- E 是专家数量
- α 是权重系数
- p_{i,e} 表示第 i 个样本被路由到专家 e 的概率

听起来很合理，对吧？

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
<table class="routerkt-result"><thead><tr><th>Models</th><th>algebra05</th><th>assist09</th><th>bridge06</th><th>slepemapy</th><th>spanish</th><th>ednet</th><th>database</th><th>computer</th><th>prob</th><th>linux</th></tr></thead><tbody><tr><td>DKT</td><td>0.783</td><td>0.7577</td><td>0.7616</td><td>0.7049</td><td>0.8136</td><td>0.6533</td><td>0.754</td><td>0.7243</td><td>0.7153</td><td>0.7501</td></tr><tr><td>SAKT</td><td>0.7503</td><td>0.7366</td><td>0.7337</td><td>0.6706</td><td>0.8008</td><td>0.6466</td><td>0.7416</td><td>0.7074</td><td>0.7142</td><td>0.7374</td></tr><tr><td>DKVMN</td><td>0.771</td><td>0.7561</td><td>0.7668</td><td>0.6996</td><td>0.8119</td><td>0.6587</td><td>0.753</td><td>0.7168</td><td>0.7196</td><td>0.7459</td></tr><tr><td>CoreKT</td><td>0.7592</td><td>0.7438</td><td>0.747</td><td>0.7143</td><td>0.8163</td><td>0.6622</td><td>0.7836</td><td>0.7394</td><td>0.7311</td><td>0.781</td></tr>
<tr><td>ATKT</td><td>0.7595</td><td>0.7545</td><td>0.741</td><td>0.6947</td><td>0.8028</td><td>0.6444</td><td>0.7548</td><td>0.7253</td><td>0.7029</td><td>0.753</td></tr><tr><td>DeepIRT</td><td>0.7705</td><td>0.7564</td><td>0.7677</td><td>0.6946</td><td>0.806</td><td>0.6568</td><td>0.7494</td><td>0.7156</td><td>0.7212</td><td>0.7437</td></tr><tr><td>SparseKT/relative</td><td>0.7839</td><td>0.7677</td><td>0.7687</td><td>0.7249</td><td>0.8382</td><td>0.6953</td><td>0.8348</td><td>0.7968</td><td>0.7398</td><td>0.8300</td></tr><tr><td>SimpleKT/relative</td><td>0.7781</td><td>0.7552</td><td>0.7625</td><td>0.7149</td><td>0.8365</td><td>0.6671</td><td>0.7589</td><td>0.7253</td><td>0.7291</td><td>0.7527</td></tr><tr><td>SimpleKT/monotonic</td><td>0.7787</td><td>0.7411</td><td>0.7561</td><td>0.7106</td><td>0.8353</td><td>0.6659</td><td>0.7499</td><td>0.722</td><td>0.7286</td><td>0.7496</td></tr>
<tr><td>SimpleKT/ALiBi</td><td>0.7827</td><td>0.7615</td><td>0.7714</td><td>0.7126</td><td>0.8366</td><td>0.66</td><td>0.7571</td><td>0.7571</td><td>0.7308</td><td>0.7543</td></tr><tr><td>SimpleKT/RouterKT</td><td>0.7844</td><td>0.7716</td><td>0.7757</td><td>0.7452</td><td>0.8429</td><td>0.6789</td><td>0.8041</td><td>0.7456</td><td>0.7423</td><td>0.7731</td></tr><tr><td>CL4KT/relative</td><td>0.7736</td><td>0.7555</td><td>0.7652</td><td>0.7116</td><td>0.8169</td><td>0.6632</td><td>0.7862</td><td>0.7239</td><td>0.7172</td><td>0.7596</td></tr><tr><td>CL4KT/monotonic</td><td>0.7765</td><td>0.7554</td><td>0.7642</td><td>0.7123</td><td>0.8147</td><td>0.6617</td><td>0.7551</td><td>0.7251</td><td>0.7218</td><td>0.7526</td></tr><tr><td>CL4KT/ALiBi</td><td>0.7899</td><td>0.7609</td><td>0.7765</td><td>0.7147</td><td>0.8241</td><td>0.6659</td><td>0.7593</td><td>0.7273</td><td>0.7219</td><td>0.7602</td></tr>
<tr><td>CL4KT/RouterKT</td><td>0.7878</td><td>0.7815</td><td>0.7792</td><td>0.7519</td><td>0.8268</td><td>0.6947</td><td>0.8372</td><td>0.7752</td><td>0.7436</td><td>0.822</td></tr><tr><td>AKT/relative</td><td>0.7879</td><td>0.7661</td><td>0.7711</td><td>0.7224</td><td>0.8427</td><td>0.6939</td><td>0.8236</td><td>0.7978</td><td>0.733</td><td>0.8217</td></tr><tr><td>AKT/monotonic</td><td>0.7854</td><td>0.7636</td><td>0.7671</td><td>0.7253</td><td>0.8384</td><td>0.6945</td><td>0.8242</td><td>0.7975</td><td>0.7508</td><td>0.8202</td></tr><tr><td>AKT/ALiBi</td><td>0.7974</td><td>0.7751</td><td>0.7792</td><td>0.7266</td><td>0.839</td><td>0.6864</td><td>0.8267</td><td>0.7974</td><td>0.7279</td><td>0.8208</td></tr><tr><td>AKT/RouterKT</td><td>0.8017</td><td>0.7815</td><td>0.7872</td><td>0.7545</td><td>0.853</td><td>0.7213</td><td>0.896</td><td>0.8399</td><td>0.7768</td><td>0.8737</td></tr></tbody></table>


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

坏就坏在这个全局路由。我们当前的 Token 是不可以知道未来 Token 的任何信息的，包括它可以被路由到哪个注意力头！如果是只用题目信息去路由还无所谓，但是其实这里的 x 已经是包含了答案信息的。

> x 虽然是一个被掩码的矩阵，但是这个掩码是说，在给定的 timestamp 上，我们无法关注关注到后续的信息
> 但是如果我们对 x 进行平均池化，会绕过这个掩码机制


所以包含 label 信息的 task-level routing 是不可以被用在自回归任务上的！这里需要改为：

```
routing_mask = routing_mask.permute(0, 2, 1).unsqueeze(-1)
```

或者说修正为只能用题目信息（q_embed_data）去路由。

当我们把这个部分修正，就只能得到和 AKT 差不多的性能。这里也提示我，一个非常值得注意的点是，自回归任务里平均池化、最大最小池化是非常需要注意的。事实上，我在开始想办法把 MoE 的思想融入知识追踪任务的时候，就在考虑是否存在数据泄露，因为性能提升很大，显得非常异常。

自查过几次代码，也询问过 Chadiskt-result-v2PT、Claude、Gemini 等 ChatBot 他们都说没问题...

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

不过值得一提的是，DisKT 使用了和 SimpleKT 类似的单层编码器结构，并且可以看到在修复后，DisKT在更长的序列长度上仍然保持了较好的性能，**这说明我们并不应该否认他们模型的贡献**，只是对于相关结果应当更加慎重。

# PyKT的问题准确率[3]
我在 pykt 做随机标签检测，发现他们的指标可能也有异常。[PyKT - 关于在随机数据上的训练问题](https://github.com/pykt-team/pykt-toolkit/issues/245)

![PyKT](/rethinking-kt/pykt-balance.png)

这里稍微解释一下，这里的

```
{'testauc': 0.6438930956398151, 'testacc': 0.5795748255363143, 'window_testauc': 0.644660752721687, 'window_testacc': 0.5804480980012895, 'oriaucconcepts': 0.5033722608280454, 'oriauclate_mean': 0.5028598981901652, 'oriauclate_vote': 0.5031274181458916, 'oriauclate_all': 0.5032359821913478, 'oriaucearly_preds': 0.502985136956939, 'oriaccconcepts': 0.5033216669907318, 'oriacclate_mean': 0.5021625138745359, 'oriacclate_vote': 0.5030619665480155, 'oriacclate_all': 0.5027174953113637, 'oriaccearly_preds': 0.502181651165461, 'windowaucconcepts': 0.5035561220606548, 'windowauclate_mean': 0.502587681942156, 'windowauclate_vote': 0.503010394236416, 'windowauclate_all': 0.5036517308862161, 'windowaucearly_preds': 0.5030404483146422, 'windowaccconcepts': 0.5037813903881581, 'windowacclate_mean': 0.5019855664592004, 'windowacclate_vote': 0.502806521822139, 'windowacclate_all': 0.5029974416739853, 'windowaccearly_preds': 0.5021192103554928}
```

testauc是pykt提出的在题目级别进行预测（如果我没猜错的话），所有需要聚合的（比如oriauclate_mean, windowaucearly_preds），是在知识点级别进行预测。

这里也是非常诡异，按理说即使是在题目级别进行预测，我们的结果也应当是 0.5，而不应该是 0.66附近（其实就是2/3），这可能是 PyKT 框架的一个结构性偏差，可能也是为什么他们报的题目级别的指标异常的高。不过大部分使用 PyKT 的文章，都会使用这套统一的评测框架对结果进行评测，这种偏差不至于影响到模型的公平性比较。

## conclusion
我们发现，RouterKT 和 DisKT 在有数据泄露的情况下，在 algebra2005 这个数据集上的性能提升都非常有限，如果后面看到明显高于 AKT 在 algebra2005 上的结果，就得仔细看看了...

在几个月前，发了篇关于 Knowledge Tracing 的牢骚文，当时认为 Knowledge Tracing 做无可做。后面意外看到 MoH: Multi-Head Attention as Mixture-of-Head Attention [4] 这篇文章，快速复现他们的注意力机制发现的确是有提升，而且在常见的几个数据集上，提升都在预期内（大部分的文章，相比于AKT，提升都在 1 个点上下），因为是在 AKT 的基础上改的，所以也并未意识到有何不妥。

可能真如我在牢骚文里所言，知识追踪的性能上限几乎在五年前的 AKT 甚至十年前的 DKT 就已经确定了。数据本身的稀疏性、学生行为的噪音，这些因素让进一步提升准确率变得很难。另一方面就是，这个问题本身是一个既简单又复杂的问题，简单在于，要达到「几乎是大家心里公认的SoTA」，AKT，的性能，几乎不需要额外的设计，比如 SimpleKT 告诉我们，去掉 AKT 的距离机制、去掉交叉编码器同样能得到大差不差的结果，RNN（也就是DKT）达到的模型性能也比较好。复杂在于我们刚刚提到的稀疏性、噪音，这决定了信息论角度的性能上限。

另一方面，在面试微信的时候，和面试官聊到了知识追踪这个任务。面试官灵魂发问，「你觉得真的会有平台需要知道学生的未来做题情况吗」？我被问得哑口无言。诚然，知识追踪宣称自己有捕捉学生知识状态的能力，但是我们要如何去应用这个能力呢？我看到，最近有一些论文开始使用知识追踪的知识状态去做 Tutor，Training Turn-by-Turn Verifiers for Dialogue Tutoring Agents: The Curious Case of LLMs as Your Coding Tutors [5]。

## reference
[1] Kudugunta S, Huang Y, Bapna A, et al. Beyond distillation: Task-level mixture-of-experts for efficient inference[J]. arXiv preprint arXiv:2110.03742, 2021.

[2] Zhou Y, Lv Z, Zhang S, et al. Disentangled Knowledge Tracing for Alleviating Cognitive Bias[C]//Proceedings of the ACM on Web Conference 2025. 2025: 2633-2645.

[3] Liu Z, Liu Q, Chen J, et al. pyKT: a python library to benchmark deep learning based knowledge tracing models[J]. Advances in Neural Information Processing Systems, 2022, 35: 18542-18555.

[4] Jin, Peng, et al. "MoH: Multi-head attention as mixture-of-head attention." arXiv preprint arXiv:2410.11842 (2024).

[5] Wang, Jian, et al. "Training Turn-by-Turn Verifiers for Dialogue Tutoring Agents: The Curious Case of LLMs as Your Coding Tutors." arXiv preprint arXiv:2502.13311 (2025).

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

.routerkt-result tr:nth-child(11),
.routerkt-result tr:nth-child(15),
.routerkt-result tr:nth-child(19) {
    background-color: var(--red);
}

</style>
