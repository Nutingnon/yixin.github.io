---
title: CLIP 系列学习 (CLIP, BLIP, Long-CLIP, GLIP, PyramidCLIP，MotionCLIP和SigLIP)
date: 2024-12-17
---

## 引言

### 什么是CLIP

[CLIP (Contrastive Language-Image Pre-training)](https://arxiv.org/abs/2103.00020) <font style="color:rgb(25, 26, 36);">模型由OpenAI在2021年提出，是一种用于图像和文本联合表示学习的多模态预训练模型。其核心思想在于，通过对比学习的方式，在大规模图像-文本对数据集上进行预训练，使模型能够学习到图像和文本之间的深层语义关联。这种学习方式不仅突破了传统视觉模型在泛化性和迁移能力上的局限，还为实现真正的zero-shot学习提供了可能。</font>

[https://miro.medium.com/v2/resize:fit:1400/format:webp/1*OVi8blLZw_wf2rrxdlfbdg.png](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*OVi8blLZw_wf2rrxdlfbdg.png)

### 为什么CLIP这么火？

CLIP模型的应用场景非常广泛。 在图像分类领域，CLIP可以实现零样本分类，即不需要任何训练数据即可对未知类别的图像进行分类。 在文本到图像检索领域，CLIP可以根据文本描述快速检索出与之匹配的图像。 此外，CLIP还可以应用于图像生成、视频理解等多个领域



## CLIP (Contrastive Language–Image Pre-training)

学习资料【Videos】：

+ [跟李沐学 AI - CLIP 论文逐段精读](https://www.youtube.com/watch?v=OZF1t_Hieq8&pp=ygUEY2xpcA%3D%3D)
+ [OpenAI CLIP: ConnectingText and Images (Paper Explained)](https://www.youtube.com/watch?v=T9XSU0pKX2E&t=1501s)



### 工作原理

#### 图像编码器和文本编码器的设计。

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733715080285-5cba4a3d-172f-456e-9461-fd4f894cf365.png)



将图像、文本配成 N 对正样本，则有$ N^2-N $对负样本。每一对的图片和文字输入 Image-Encoder 和 Text-Encoder，得到 $ I_f $图像特征和$ T_f $文本特征。后在通过一个 linear layer 去转换到同一个 embedding 空间 $ I_e $和 $ T_e $， 然后基于 cosine similarity 来得到 logits，后通过 cross-entropy 得到 loss。



**Bag of words Prediction**

![image](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733725163807-4559a43b-9e6a-4ece-a64f-5ca5d9485cb4.png)

根据文章中说，在最开始实验的时候，选了一个 CNN 网络和 text Transformer 网络来做（Text Transformer 用来 predict Image caption。但后来发现 Transformer 很难去做 scaling，而且要求的计算量又大又低效。

> <font style="color:rgb(0,0,0);">we show that a 63 million parameter transformer language model, which already uses twice the compute of its ResNet-50 image encoder, learns to recognize ImageNet classes three times slower than a much simpler baseline that predicts a bag-ofwords encoding of the same text.</font>

<font style="color:rgb(0,0,0);">不如一个 predict bag of words 的模型，但不管是 Transformer 还是另外的 predict bag of words 这样的 baseline 都有一个问题就是想去特别精确地输出图片对应的文字。和 对比模型 比起来，要达到同样的表现水平的话，需要一个数量级的计算量。为了避开精确预测文字这种坑，CLIP 用了从大量图像-文本对中找出最合适的 pair 作为目标。所以把 bag-of-words 的 model 的 训练 loss 从 predictive objective 换成了contrastive objective 然后观察到了 4 倍的提升。这种 loss 也可以叫做 multiclass N-pair loss</font>



图像和文本的 embedding 映射函数都采取的是简单的线性层，没有用非线性层，实验观察这两者没有什么太大区别。（但有一个 Normalization）。

#### 模型选取

对于图像这一块，用了 ResNet 的不同尺寸和 ViT（详见 open_clip)。我们图像的 scaling up 同时增加了模型的深度、宽度和分辨率。



文本这一块的 text-encoder 采用了 Transformer，选用了 63Million 参数的一个 12 层 512 宽，8 个 attention-heads 的模型， 通过 BPE 来 token，vocab size 有 49152 这么大。对于 Transformer 的 scale up，只增加了宽度，没动深度



有 `[SOS]`, `[EOS]` 两个特殊的 token, `[EOS]`被认为是整个句子的 representation



Masked self attention 被采用







### 训练细节

+ 从网上搜集的 400 million 个图像-文本对
+ 图片数据只用了 crop 这一种增强手段
+ softmax 中的温度是学来的
+ Adam Optimizer
+ 每个模型都训练了 32 个 epoch
+ 用了 weight decay optimization，用在所有层
+ 用 grid search 来找 hyper parameter（模型是用了一个参考模型 ResNet-50，训练 1 个 epoch）
  - 然后更大的模型是人工调了一下
+ batch-size 是 32768
+ 混合精度训练
+ gradient checkpointing
+ half-precision 模型和 adam statistics



最后论文里默认选用了 `ViT-L/14@336px` 这个模型



#### 对比学习 (Contrastive Learning) 目标函数。

```python
# image_encoder - ResNet or Vision Transformer
# text_encoder - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l] - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t - learned temperature parameter
# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]
# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)
# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)
# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t)/2
```

### 优势

#### 零样本学习能力（Zero-Shot Learning）。

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733727062234-f997fbbf-4d7c-40d0-8640-4d50d19590e8.png)



当做 zero-shot 的时候，发现得加一点 Prompt Engineering，增加了 prompt-engineering 的方法比起直接用类名，能够显著提升模型表现。

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733727253469-2c959236-fc68-4c25-b9b4-a3411fb7bcdf.png)

prompt engineering 样例：

+ 狗 可以被改写为 "一张狗的照片"





<font style="color:rgb(14, 14, 14);">CLIP模型对用于图像描述的单词很敏感。文本“a photo of a bird”、“a photo of a bird siting near bird feeder”或“an image of a bird”与相同的图像匹配产生的概率是不同的。</font>

![image](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733731333174-dc04f7b1-f5ed-43b4-b615-471e1a11a00d.png)

<font style="color:rgb(14, 14, 14);"></font>

<font style="color:rgb(14, 14, 14);"></font>

#### 可扩展性与迁移能力。

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733727084172-958d5206-d854-4e45-a517-28f23db0edae.png)



可见大部分基于 clip 特征的下游任务都表现不错



![image](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733727398304-d3388eff-8cc9-4328-9ec7-2d5de50ab11b.png)



#### 鲁棒性较强

![image](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733727486913-aa29375d-8616-4347-95e8-f6c78a492fcc.png)



为了证明他的 zero-shot 能力是真实的，文章做了**<font style="color:rgb(0,0,0);">Data Overlap Analysis </font>**<font style="color:rgb(0,0,0);">来证明他们用来测试 zero-shot 能力所用到的任务没有出现在他们的训练集中。</font>

<font style="color:rgb(0,0,0);"></font>

### <font style="color:rgb(0,0,0);">应用场景</font>

在图像生成、图像问答等多模态应用中已经离不开 clip 的存在。



### 劣势(Limitations)

文章中自己提到的劣势有以下方面

+ 在各个任务上不够 sota -- 如果要达到各方面 sota，估计要当前的训练资源翻 1000 倍
+ zero-shot 能力在一些任务上仍然不够理想，例如一些细粒度的任务：汽车的型号分类，花卉分类等。一些抽象任务表现也不太行，例如在图像中对某一目标进行计数。还有的因为任务相对比较新，CLIP 也没有涉及，所以表现起来就跟乱猜没啥区别（比如 对离你最近的汽车有多少距离距离进行分类）
+ 对于图像类型是 ood 的，也不太理想

### 样例代码

[https://huggingface.co/docs/transformers/model_doc/clip](https://huggingface.co/docs/transformers/model_doc/clip)

```python
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```





**实际上，后续工作中改进了其他的劣势，包括**

+ 原本的 clip 只能支持 77 个词的输入，`long-clip` 这个工作 拓展了这个限制
+ `BLIP` 增加了任务的难度和多样性
+ CLIP 的训练形式是对整张图片的描述是否对齐，缺少了对图片中更细粒度的对齐。 GLIP 解决了这点
+ CLIP 针对特有领域的应用效果（泛化能力）相对较差（如文章中提到的平时生活中见得比较少的但其实较简单的 MNIST）。医疗领域内，一些专门针对性的 CLIP 被提出，如 `BioMedCLIP`, `PubMedCLIP`



## BLIP & BLIP-2

[BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086) Jan 28, 2022



[BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) Jan 30, 2023



项目地址：[https://github.com/salesforce/BLIP](https://github.com/salesforce/BLIP)

学习资料【Video】

+ [BLIP -- yannic](https://www.youtube.com/watch?v=X2k7n4FuI7c)
+ [BLIP2 -- huggingface[Computer Vision Study Group Session on BLIP-2]](https://www.youtube.com/watch?v=k0DAtZCCl1w)



### 创新点

#### BLIP

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733734880226-b73bf630-c427-4a60-b0f8-5c40e3143672.png)

相较于 CLIP，BLIP 不仅采用 contrastive loss，还增加了 match loss 和 language loss。同样颜色的模块之间 share weights，可以看到这里有几个模块

+ Image Encoder
+ Text Encoder
+ Image-Grounded Text Encoder （其实就是之前的 Text-Encoder，只不过多了一个 CrossAttn 的对齐模块）
+ Image-Grounded Text decoder，为了生成任务，把 Image-grounded Text encoder 里的 `Bi Self-Att`替换成了 `Causal Self-Att`



#### BLIP2

[https://huggingface.co/docs/transformers/model_doc/blip-2](https://huggingface.co/docs/transformers/model_doc/blip-2)

> The BLIP-2 model was proposed in BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models by Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi. BLIP-2 **leverages frozen pre-trained image encoders** and **large language models (LLMs) by training a lightweight, 12-layer Transformer encoder in between them,** achieving state-of-the-art performance on various vision-language tasks. Most notably, BLIP-2 improves upon Flamingo, an 80 billion parameter model, by 8.7% on zero-shot VQAv2 with 54x fewer trainable parameters.



主要就是引入了 pretrain 的 model，中间揉了个 Transformer Encoder 来做中介去揉合两个 pretrain 的 models



### 技术细节

#### BLIP

**Image Encoder**：Visual Transformer

**MED**：Multimodal mixture of encoder-decoder

+ **Unimodal Encoder**: Text-encoder 是 Bert （`[CLS]`token 代表整个句子）。Image Encoder 是 Visual Transformer。
+ **Image-grounded text encoder**: 视觉特征通过插入额外的 `cross-attention` （在 `self-attention`和 `feed-forward-network`）之间），在每个 text-encoder 的 transformer block。
+ **Image-grounded text decoder**: A `[Decode]` token is used to signal the beginning of a sequence, and an end-of-sequence token is used to signal its end.



##### 预训练目标

**Image-Text Contrastive Loss** **（ITC）**

![Align before Fuse:](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733737511766-12504ebe-6f3e-4002-ad93-bce3c6cf35ef.png)

****

**Image-Text Matching Loss (ITM)**

+ Binary classificatio loss with hard sample mining

****

**Language Modeling Loss (LM)**

+ cross-entropy loss in auto-regression manner

****

**CapFilt：**



![CapFilt](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733736524328-905aacf1-e2f2-4594-9e22-01fc4b278df1.png)



先用前面的数据训练一个初版的 BLIP，然后基于 COCO 数据集去基于不同的训练目标来搞成两个模型，一个是 Captioner ，一个 Filter。

> a captioner to generate captions given web images, and a filter to remove noisy image-text pairs.

训练好后，用外部数据，来过 captioner，然后用 filter 过一遍，然后继续提升 MED 这个模型



**数据集**

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733745802704-91687557-9a68-46d9-b3f4-e1b55bb9ecae.png)



##### 应用

和 CLIP 不一样的是，这个模型因为有三个模块，其实自带挺多功能，比如 Image-Text Retrieval， Image Captioning，Visual Question Answering，Natural  Language Visual Reasoning (NLVR (Suhr et al., 2019) asks the model to predict whether a sentence describes a pair of images.)



#### BLIP2

数据集：和 BLIP 一样，但经过了一些处理

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733745876635-2f9b7dca-9ed9-4c2d-aa9f-60d4c4ce0671.png)



pretrain 模型结构选择

![都是Encoder-Decoder结构的](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733745922668-6976662f-c5ac-4cba-86b8-bc5e85526198.png)



BLIP2 中间的模型进行了两阶段的学习，第一个阶段是 vision-language representation learning，从冻结参数的 pretrain image encoder 中学，第二个阶段是 vision-to-language 任务，从 LLM 中学。



![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733746604781-10f1bf36-2ec4-4aed-86a7-fe694c350d9e.png)

在 BLIP2 中的 Q-former 的左边的模型主要是提取视觉特征的，会产出固定的 token 数量（无视输入的分辨率），右边是个 encoder-decoder. 对于 image transformer，有几个固定的 learnable query（实验设置 32 个 query，，每个 query 的 dimension 是 768），output ($ Z $)也是一样的维度  （32x768）。Text 部分的网络权重用 bert-base 来初始化（那说明架构也一样），但把 cross-attention 部分的权重给随机初始化掉了。

总的来说，Q-former 含 188 Million 个参数。



**第一阶段的学习（Bootstrap Vision-Language Representation Learning from a Frozen Image Encoder）**

Image-Text Contrastive Learning （ITC）

>  We align the output query representation Z from the image transformer with the text representation $ t $from the text transformer, where $ t $ is the output embedding of the `[CLS] `token.
>
>  用每个 query 和 那个 `cls`计算 pairwise similarity，然后选择最高的一个。（神奇，这个时候 label 从哪儿来，他这都是一张图啊。。。）
>
>  为了防止信息泄露，用了统一的 self-attention 层，也就是这里限制了 query 和 input text 看不到对方。

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733747408528-b124df8f-a172-455f-9dc6-01884d99239c.png)

这里应该是 query 产出的 token 被认为是 text 的某种表示，长度 32，和真正的 ground truth text 去得到 ITC，本质上是个多分类任务。问题是 label 从哪儿来啊。。。



**ITG Loss（Image-grounded Text Generation）**

目标是让这个 Q-former 具有文字生成的功能。由于该结构不允许 image encoder（frozen）直接和 text token 接触，所以输入的可学习 queries 必须得通过学习来拥有为了 text 输出而抓取 image 特征的能力。这里用了个 multimodal causal self-attention mask 来控制。这里每一个 query 都可以看到彼此但看不到 text tokens。text tokens 可以看到所有的 query 和其之前的 text tokens。这里还将 `[CLS]` 这个特殊的 token 换成了 `[DEC]`

感觉这里就是个 LM 的 loss



**<font style="color:rgb(0,0,0);">Image-Text Matching </font>**<font style="color:rgb(0,0,0);">(ITM)</font>

<font style="color:rgb(0,0,0);">是一个二元的 classifier，来预测 image-text pair 是否成对。这里用的是 bi-directional self-attention masks，也就是基本没有 masks</font>

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733748033546-24d41ad7-3743-4016-892f-6463820a9a68.png)



感觉这玩意重点在 learned queries。但确实没有放出训练脚本和预处理脚本。只有模型权重和结构。这篇 paper 这方面做得挺不好的，把事情说清楚都没做到，代码也不给看。

### 样例代码

BLIP

[https://huggingface.co/docs/transformers/model_doc/blip](https://huggingface.co/docs/transformers/model_doc/blip)



```python
from PIL import Image
import requests
from transformers import AutoProcessor, BlipModel

model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```



BLIP2

[https://huggingface.co/docs/transformers/model_doc/blip-2](https://huggingface.co/docs/transformers/model_doc/blip-2)

```python
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2Model
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
model.to(device)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "Question: how many cats are there? Answer:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

outputs = model(**inputs)
```



## Long Clip

Long-CLIP: Unlocking the Long-Text Capability of CLIP | <font style="color:rgb(31, 35, 40);"> </font>_**<font style="color:rgb(31, 35, 40);">ECCV2024</font>**_<font style="color:rgb(31, 35, 40);">.</font>

+ github：[https://github.com/beichenzbc/Long-CLIP](https://github.com/beichenzbc/Long-CLIP)



**创新点**：a significant limitation of CLIP lies in the inadequate length of text input. The length of the text token is restricted to 77, and an empirical study shows the actual effective length is even less than 20. 

总的来说，该文就是想突破 clip 原本 77 个 token 的限制，达到更长 token，能够让相关的下游应用（图像生成) 能够应对更加丰富的细节和对齐能力。但是在实验过程中，发现 clip 在位置 20 以后的 token 其实训练的都不太行。所以该文把前 20 个 token 的 embedding 保留，然后重新训练了 20 以后的token。



**挑战**：想当然的去直接训练一个长的 clip 并不合适。 Nevertheless, achieving this goal is far from straightforward, as simplistic fine-tuning can result in a significant degradation of CLIP’s performance. 



所以提出两个办法来做：

+ **Knowledge-preserved stretching of positional embedding 位置嵌入在保留知识的情况下拉长**
+ **primary component matching of CLIP Features 和 CLIP 特征的主成分匹配**

****

训练数据：1-million extra long text-image pairs。



效果：20%的 long-caption text-image Retrieval，6%的 traditional text-image Retrieval 的提升

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733800021691-44af379d-583c-41c6-928e-c8a14bf3eef3.png)

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733800034615-9c820097-6c39-49f1-bfb9-d6322d6dbacb.png)

### 扩展性

由 CLIP 的 77 变成了 256 的长度



### 技术方案

首先说了下常见的 image-text pair 的数据集：Visual Genome [10], Conceptual-12M [1], SBU [19], COCO [15], LAION-5B 这些基本都是短的 caption。然后他们 找了个叫 ShareGPT4V 的数据集，有长的 caption，大概 1.2 million 个数据对。



#### CLIP 的有效长度

作者认为 CLIP 虽然最大长度是 77，但有效长度更短。他们用了一个自己建的数据集叫 urban-200 （公开在了 huggingface）

> <font style="color:#000000;">After the first submission, we further scaled up Urban-200 into Urban-</font>
>
> <font style="color:#000000;">1k. The dataset has been released at https://huggingface.co/datasets/</font>
>
> <font style="color:#000000;">BeichenZhang/Urban1k. Urban-200 is used in the main paper. Detailed results</font>
>
> <font style="color:#000000;">about Urban-1k is shown in supplementary materials.</font>

他们认为理论上随着 text 的长度增加，信息越丰富，那么图片的 Retrieve 能力会增强（应为对该图片的描述增多了），然而他们通过实验发现，CLIP 的 R@1 准确率在文字长度超过 20 以后就增长的非常缓慢了，所以认为 CLIP 的有效长度实际上在 20.

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733800823636-f335855c-c4b2-489a-a08b-b04657e81fc7.png)



此外，他们还发现 clip 模型在对图片进行训练时，由于是整个图片和整个 caption 进行配对的，所以对于图片的细节内容容易混淆。这里给了个例子：一张图片里含有柠檬（左）和茄子（右）。然后写 4 个 prompt，分别物体的位置和颜色进行混淆。

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733800929137-e358d84b-ac17-44ea-acc2-7d93bd87b043.png)

理想的情况下是 A 第一，然而 CLIP 把 A 排最后。





#### Knowledge Preserving Stretching

> <font style="color:#000000;">Therefore, instead of performing full interpolation with a fixed value, we choose to retain the embedding of the top 20 positions, which aligns with the effective length identified in our experiment. As for the remaining 57 positions, we apply interpolation using a larger ratio denoted as λ2. This process can be mathematically formulated as follows</font>

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733801421267-399915f7-8865-4629-b16d-d82e20db5cc9.png)

也就是对 positional encoding 进行重新训练， 但保留位置在 20 及以内的。对于 20-77 的，用另外一个 function 去做插入，这里 $ \lambda_2 $设置为 4，两个符号不一样估计是向下取整和向上取整把。这里把 pos 最大限制为 248， 这里也就是 PE（x）中的 x 应该不超过 62。





具体代码如下

LongClip/model/longclip.py --> def load_from_clip

```python
    model = build_model(state_dict or model.state_dict(), load_from_clip = True).to(device)
        
    positional_embedding_pre = model.positional_embedding.type(model.dtype)
            
    length, dim = positional_embedding_pre.shape
    keep_len = 20
    posisitonal_embedding_new = torch.zeros([4*length-3*keep_len, dim], dtype=model.dtype) # why this magic value?

    for i in range(keep_len):
        posisitonal_embedding_new[i] = positional_embedding_pre[i] # 0 to 19
    for i in range(length-1-keep_len): # 0 to 248 - 1 - 20 = 227
        # i=0, new 20 = pre 20 
        # i=1, new 24 = pre 21
        # i=2, new 28 = pre 22
        posisitonal_embedding_new[4*i + keep_len] = positional_embedding_pre[i + keep_len]

        # i=0, new 21 = 3/4 pre 20 + 1/4 pre 21
        # i=1, new 25 = 3/4 pre 21 + 1/4 pre 22
        # i=2, new 29 = 3/4 pre 22 + 2/4 pre 23
        posisitonal_embedding_new[4*i + 1 + keep_len] = 3*positional_embedding_pre[i + keep_len]/4 + 1*positional_embedding_pre[i+1+keep_len]/4

        # i=0, new 22 = 2/4 pre 20 + 2/4 pre 21
        # i=1, new 26 = 2/4 pre 21 + 2/4 pre 22
        # i=2, new 30 = 2/4 pre 22 + 2/4 pre 23
        posisitonal_embedding_new[4*i + 2+keep_len] = 2*positional_embedding_pre[i+keep_len]/4 + 2*positional_embedding_pre[i+1+keep_len]/4

        # i=0, new 23 = 1/4 pre 20 + 3/4 pre 21
        # i=1, new 27 = 1/4 pre 21 + 3/4 pre 22
        # i=2, new 31 = 1/4 pre 22 + 3/4 pre 23
        posisitonal_embedding_new[4*i + 3+keep_len] = 1*positional_embedding_pre[i+keep_len]/4 + 3*positional_embedding_pre[i+1+keep_len]/4

    # 最后几个
    posisitonal_embedding_new[4*length -3*keep_len - 4] = positional_embedding_pre[length-1] + 0*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4

    posisitonal_embedding_new[4*length -3*keep_len - 3] = positional_embedding_pre[length-1] + 1*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4

    posisitonal_embedding_new[4*length -3*keep_len - 2] = positional_embedding_pre[length-1] + 2*(positional_embedding_pre[length-1] - 
    positional_embedding_pre[length-2])/4
    
    posisitonal_embedding_new[4*length -3*keep_len - 1] = positional_embedding_pre[length-1] + 3*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
            
    positional_embedding_res = posisitonal_embedding_new.clone()
            
    model.positional_embedding = nn.Parameter(posisitonal_embedding_new, requires_grad=False)
    model.positional_embedding_res = nn.Parameter(positional_embedding_res, requires_grad=True)

    if str(device) == "cpu":
        model.float()
    return model, _transform(model.visual.input_resolution)


# model_longclip.py里，对这两个embedding有mask：
...
        self.initialize_parameters()
        self.mask1 = torch.zeros([248, 1])
        self.mask1[:20, :] = 1
        self.mask2 = torch.zeros([248, 1])
        self.mask2[20:, :] = 1
...

```





他们还把这种改过的方法和直接插值的方法的最后微调效果做了比较，确实 direct fine-tuning 可能会下降，也可能会提升，但他们的这种一定会提升。

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733802067052-5c67b91b-a65f-46ee-b66e-f16ed36e0c6c.png)



在具体训练中，参数设置如下，

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733801537472-899166ed-a7fb-4d5a-9808-9c576eb0aa73.png)



#### Fine-tuning with Primary Component matching

文中说只把模型在长的 caption 上微调是不太行的，会把模型推向另一个极端，也就是不分重点的把图片的所有描述细节都融合到 embedding 里。

> Merely fine-tuning the model with long captions may push it to another extreme, where it attempts to encompass all attributes in a single image without distinguishing their respective importance.



所以文章中同样做了粗粒度的 shot summary caption 的图像 文本对的数据准备，然后训练

> <font style="color:#000000;">Apart from aligning the fine-grained feature of an image with its long caption, we extract a coarse-grained image feature that focuses on capturing key attributes. This coarse-grained feature is then aligned with a short summary caption.By doing so,we require the model not only to capture detailed attributes but also to discern and prioritize the importance of different attribute</font>



为此，做了三个 modules，(分解、过滤、重建

+ <font style="color:#DF2A3F;">component docomposition function </font>$ \mathcal{F} $, This function **decomposes the feature into several vectors that represent different attributes and also analyzes the importance of each attribute.**
+ **<font style="color:#DF2A3F;">component-filtration function </font>**$ \mathcal{E} $**，根据重要性来过滤掉不重要的 attributes**
+ **component-reconstruction function **$ \mathcal{F}^{-1} $**, reconstruct the image feature**





![The Numpy-like pseudo-code of our fine-tuning. We separately align the fine- grained and coarse-grained information in both modalities.](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733811422223-ef479fac-ae63-426a-8ad8-329cefffdd3b.png)



![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733813884223-5384677a-063b-47f8-bf70-994e8496ed9d.png)



这里的 F 用的是 PCA，然后用 32 个最大的 eigen value 来保证最大的重要性得以保留。然后再反过去重建得到这个 coarse 输出。



### 实际表现

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733814037170-f3668cb7-4b1d-43c9-899f-dc01cbd459ad.png)



### 样例代码

from github

```python
from model import longclip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = longclip.load("./checkpoints/longclip-B.pt", device=device)

text = longclip.tokenize(["A man is crossing the street with a red car parked nearby.", "A man is driving a car in an urban scene."]).to(device)
image = preprocess(Image.open("./img/demo.png")).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image = image_features @ text_features.T
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs) 
```





## GLIP and <font style="color:rgb(31, 35, 40);">GLIPv2 </font>

**GLIP**: [https://arxiv.org/abs/2112.03857](https://arxiv.org/abs/2112.03857)

**GLIPv2**: [https://arxiv.org/abs/2206.05836](https://arxiv.org/abs/2206.05836)

项目地址： [https://github.com/microsoft/GLIP](https://github.com/microsoft/GLIP)



### 概述创新点

#### GLIP

<font style="color:rgb(6, 6, 7);">GLIP相比于CLIP，主要拓展了细粒度的 semantic 能力：</font>

1. **<font style="color:rgb(6, 6, 7);">对象级别的视觉建模</font>**<font style="color:rgb(6, 6, 7);">：</font>
   - <font style="color:rgb(6, 6, 7);">GLIP引入了</font>**<font style="color:rgb(6, 6, 7);">对象级别的视觉建模</font>**<font style="color:rgb(6, 6, 7);">，将图像划分为多个局部区域，每个区域对应一个潜在的对象。这种划分通常基于对象检测模型中的区域建议方法（如RPN）或密集特征图上的Anchor框。</font>
   - <font style="color:rgb(6, 6, 7);">相比于CLIP的基于整图特征提取，GLIP通过将图像中的区域与文本提示中的短语进行匹配（短语定位），实现了对象级别的视觉建模。</font>
2. **<font style="color:rgb(6, 6, 7);">图像与文本特征的深度融合（Deep Fusion）</font>**<font style="color:rgb(6, 6, 7);">：</font>
   - <font style="color:rgb(6, 6, 7);">CLIP的跨模态融合仅限于最后的点积操作，即在生成图像嵌入和文本嵌入后，计算两者的余弦相似度以实现匹配。GLIP在此基础上增加了深度交互的机制，通过Cross-Modality Multi-Head Attention（X-MHA）模块，让视觉特征能够直接参考语言特征，反之亦然。</font>
3. **<font style="color:rgb(6, 6, 7);">扩展到实例级别的图文对比</font>**<font style="color:rgb(6, 6, 7);">：</font>
   - <font style="color:rgb(6, 6, 7);">GLIP系列专注于instance-level级的图文对比，开放世界目标检测，而CLIP系列主要关注image-level级的图文对比，开放图像分类。</font>
4. **<font style="color:rgb(6, 6, 7);">预训练任务的创新</font>**<font style="color:rgb(6, 6, 7);">：</font>
   - <font style="color:rgb(6, 6, 7);">GLIP引入了新的预训练任务，如区域到短语的对齐、视觉问答等，使得模型能够学习图像区域和文本描述之间的对应关系。</font>
5. **<font style="color:rgb(6, 6, 7);">零样本和少样本迁移能力的提升</font>**<font style="color:rgb(6, 6, 7);">：</font>
   - <font style="color:rgb(6, 6, 7);">GLIP通过结合上述技术，展现出优秀的零样本和少样本迁移能力，在多个视觉-语言任务上都表现出色。</font>
6. **<font style="color:rgb(6, 6, 7);">多任务统一</font>**<font style="color:rgb(6, 6, 7);">：</font>
   - <font style="color:rgb(6, 6, 7);">GLIP通过统一结构的中心思想，将固定类别的分类问题重构为一个开集（open-vocabulary）vision-language匹配问题，进一步统一了定位和视觉语言理解任务</font>



> Our approach unifies the phrase grounding and object detection tasks in that object detection can be cast as context-free phrase grounding while phrase grounding can be viewed as a contextualized object detection task.

该方法在预训练上统一了 phrase grounding 和 object detection



#### GLIPv2

总的来说是拓展了 GLIP 的任务丰富程度，把分割任务也带进来了

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733899417915-ea11e36b-ca94-488f-ac64-f29e36b4948c.png)



**Localization + VL understand = grounded VL understanding**







### 技术细节

#### GLIP

**数据集**：_In our experiments, we pre-train GLIP on 27M grounding data, including 3M human-annotated and 24M web-crawled image-text pairs._

__

为了将这种目标检测和 phrase grounding 框架的统一，改造了输入方式，原本目标检测在预测的时候只用输入图片就行了，这里还要输入所有的类别，当做 text prompt。 For example, the text prompt for COCO object detection [37] is a text string that consists of 80 phrases, i.e., the 80 COCO object、 class names, joined by “. ”, as shown in Figure 2 (Left).



![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733823508169-fe654cb2-be4e-4aaf-ac02-58429f00aa4f.png)

然后目标检测的 loss 这一块，原本的是类别 loss + 框回归 loss。**这里改成了 类别 loss 改成了 word alignment loss: i.e. **_**<font style="color:#DF2A3F;">dot product of the region (or box) visual features and the token (or phrase) language features,</font>**_

语言特征使用一个语言模型来提取的，所以一样，这里有两个 encoder，one for text，one for image。而且额外的一点是 Text 特征和 Image 特征在每一层都会融合 (文章说 CLIP 仅仅在最后一层才融合）



在没有 grounding box 的数据集里，可以用预训练好的 model 来做输出，以此得到 grounding box。然后语言这一块，caption 中的名词可以被已训练好的 NLP Parser 摘出来。在 24Million 的数据集中，有 78.1 million 个 high confidence 的 phrase-box pseudo annotations, 然后有 58.4 million 个 unique noun phrases。



在预测的时候，GLIP 缩小了目标检测任务的定义范围，即不用在开放集上预测出全部的框，只用在输入的 text prompt 里选就行了：GLIP offers a new perspective: the model does not need to propose every possible novel objects from an open set; rather it only needs to propose objects mentioned in the text prompt as the detection branch is conditioned on the prompt.



##### 3.1 UniFied Formulation

在目标检测任务中，loss 一般定义为

$ \mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{loc} $

+ 在 two-stage 的目标检测框架中，有一个独立的 RPN （Region Proposal Network）网络 和其对应的 loss $ \mathcal{L}_{RPN} $来区分前景和背景然后来优化 anchors。因为 $ \mathcal{L}_{RPN} $没有用到目标类别的语义信息，所以我们将其揉进了 $ \mathcal{L}_{loc} $中。
+ 在 one-stage 检测框架中，$ \mathcal{L}_{loc} $也同样包含一个叫 `centerness loss`的损失。



目标检测框对应的分类器往往是一个简单的线性层，此时分类损失可以被写成

$ O = \text{Enc}_{I}(\text{Img}), S_{\text{cls}} = OW^{T}, \mathcal{L} = loss(S_{\text{cls}}; T)  $

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733833895387-23eada74-6b7c-4699-937a-a9c981c89e51.png)



##### Language-Aware Deep Fusion

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733887877772-f8d0a3dc-666d-4cb6-83cc-0fa69f3a051a.png)



![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733888156874-b09302c8-4712-49ff-aaa9-eaf53e73ce94.png)



##### 模型结构

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733897955491-105ae515-3fc9-4a8f-9240-42b127e060bc.png)

这里最好的出自于微软自己出的 Swin-Tiny 和 Swin-Large 两个 backbone





#### GLIPv2

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733900094963-c26f70fa-0318-4a1a-8316-d4f71708fd5a.png)

加了一个 inter contrrastive loss，也就是为了增加 negative pairs （左），增加了$ \mathcal{L}_{mlm} $





## CLIP-in-Medical-Imaging

[CLIP in medical imaging: A comprehensive survey | 2023.12 ](https://arxiv.org/abs/2312.07353)



![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733900645322-03cbda49-673f-4bb3-bf2c-d77eed021bc2.png)

CLIP 在医学影像的应用中自成一派，根据评测效果，[BiomedCLIP](https://arxiv.org/abs/2303.00915) 和 [PubMedCLIP](https://github.com/sarahESL/PubMedCLIP/tree/main/PubMedCLIP) 效果最好。每个数据集也有自己代表性的 CLIP 模型

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733900783750-b265667e-559d-4010-9074-172bd1a181a8.png)





### PubMedCLIP

[https://aclanthology.org/2023.findings-eacl.88/](https://aclanthology.org/2023.findings-eacl.88/)

代码开源，权重也开源

[https://huggingface.co/flaviagiammarino/pubmed-clip-vit-base-patch32](https://huggingface.co/flaviagiammarino/pubmed-clip-vit-base-patch32)





#### 训练数据集

文章中提到了两个主要的医学视觉问答（MedVQA）数据集，它们是：

1. **VQA-RAD数据集**：
   - 包含315张图像和3,515个英文语言的问题-答案对。
   - 遵循先前工作的分割，测试集中的所有图像也都出现在训练集中，但测试集中的这些问题-答案对在训练集中是未见过的。
2. **SLAKE数据集**：
   - 包含642张图像和超过7,000个问题-答案对，数据集中包含英文和中文问题，但本文中只使用了英文子集。
   - 在原始数据分割中，测试集中的所有图像在训练集中是未见过的。

此外，文章还提到了用于训练PubMedCLIP的**ROCO数据集**：

+ 包含超过80,000个样本，涵盖多种成像方式，如超声波、X射线、PET扫描、CT扫描、MRI、血管造影等，来自人体的不同部位，例如头部、颈部、脊柱、胸部、腹部、手、脚、膝盖和骨盆等。
+ 文本来自PubMed文章中与图像相关的相对较短的标题（平均长度为20个单词），提供了关于图像内容的丰富解释性信息。

这些数据集为PubMedCLIP模型的训练和评估提供了丰富的医学图像和文本对，使其能够在医学视觉问答任务中取得良好的性能。

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733901211586-35086fbd-7058-4aaf-bf9d-203fbdb401b5.png)



其他的训练方式和任务构造方式都和 CLIP 一样



**微调还是从 0 训练？**

+ 微调



### BioMedCLIP

paper: [https://arxiv.org/abs/2303.00915](https://arxiv.org/abs/2303.00915)

huggingface: [https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733901823033-15aebdfd-aa28-4978-905a-23b01ea84591.png)

#### 这篇论文的标题是“BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs”，主要介绍了一个基于大规模科学文章中的图像-文本对预训练的多模态生物医学基础模型BiomedCLIP。以下是论文中提到的数据集和数据量的信息：

### 数据集和数据量

1. **PMC-15M**
   - 这是一个新创建的数据集，包含1500万个生物医学图像-文本对，这些数据来自4.4百万篇科学文章。
   - 数据集的规模是现有生物医学多模态数据集（如MIMIC-CXR）的两个数量级大。
   - 覆盖了三十多种主要的生物医学图像类型，为生物医学研究和临床实践提供了一个多样化和具有代表性的数据库。
2. **PMC-Fine-Grained-46M**
   - 通过将PMC-15M中的每个科学图表分割成单独的小图来创建，包含4600万个图像-文本对。
   - 这个数据集用于进一步细化图像类别的分布和覆盖范围。

### 数据集的多样性和覆盖范围

+ **图像类型**：PMC-15M中包含的图像类型非常多样化，从一般的生物医学插图（如统计图表、流程图）到放射影像（如磁共振、计算机断层扫描、X光）再到数字病理和显微镜图像（如光学显微镜、电子显微镜）等。





**重新训练还是微调？**

+ 重新训练的，换了 backbone，把 backbone 换成了医疗领域特定的 text-encoder



![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1733902009955-5d924e56-dd95-43cc-b566-ff4d0984af8b.png)





## PyramidCLIP

[https://arxiv.org/pdf/2204.14095](https://arxiv.org/pdf/2204.14095)

> Our main contributions are summarized as follows: 
>
> (i) We propose PyramidCLIP for more accurate image-text alignment for vision-language model pre-training, which effectively constructs two input pyramids at both sides of the visual encoder and linguistic encoder, and then align the visual elements and linguistic elements via peer-level semantics alignment and cross-level relation alignment. 
>
> (ii) We **soften the loss term of negative samples during the contrast process to ease the strict constraint, **so as to alleviate the negative effect caused by local similarities. (iii) Extensive experiments demonstrate the effectiveness of PyramidCLIP, which achieves SoTA on several downstream tasks.

主要是做了多个粒度的对齐，以及 hard-label 变成了 soft-label

### 技术细节

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1734339600345-a7f22309-8dcc-4aa7-87e5-a03a4597519a.png)

在训练过程中，每个 Image、text pair， 每个 image 都会转成 2 个 views，一个是 Local View 一个是 Global view。Local View 是通过不同比例的随机剪裁来做的。这里主要是组三个对

第一：Global View 和 text summarization 是掌握全局信息。

第二：Local View 和详细的 Text 是另一个对

第三：object Detection 得到的 ROI Features 和 分类



前两个都走 Image Encoder， Text Encoder 后进行 Contrastive loss 的计算。

第三个是图像经过 Object Detection 后，经过 ROI 提取（预训练的），经过一个线性 embedding 模块，希望他对齐 f（1）的输出结果，作为 f2 的输入。得到的目标文本拼接起来作为新的文本。



然后各个 pair 之间也要计算一些交叉的 loss。



**网络结构：**

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1734340383972-d26d7e2d-8380-43a5-933d-a1d347b1cce8.png)



#### soft label

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1734340749002-694b6569-8b38-430b-ae8c-954681ac4201.png)

这里把匹配/不匹配的 1 和 0 的关系进行一定程度的平滑。这里的平滑就是引入一个 $ \alpha = 0.2  $来解决这个事儿。这里的 N 是 mini-batch 的数量，

最后，基于 这样的 smooth loss 的设计，总共 Loss 会分为三块，一个是 peer loss （global image 对 Text Summary， local image 对 text）和 cross loss （global、local）

## 




## MotionCLIP (ECCV 2022)

项目地址：[https://guytevet.github.io/motionclip-page/](https://guytevet.github.io/motionclip-page/)

code 地址 [https://github.com/GuyTevet/MotionCLIP](https://github.com/GuyTevet/MotionCLIP)

项目地址很有意思，建议去看看

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1734404135347-a01c1a37-27ef-49fa-bfbc-22dcd3fcddbe.png)

![MotionCLIP overview. A motion auto-encoder is trained to simultaneously reconstruct motion sequences while aligning their latent representation with corresponding texts and images representations in CLIP space.](https://cdn.nlark.com/yuque/0/2024/png/2379769/1734404158955-4da2db26-55e9-4368-862c-f06f252db20d.png)

**Abstract**:

> We introduce MotionCLIP, **a 3D human motion auto-encoder featuring a latent embedding that is disentangled, well behaved, and supports highly semantic textual descriptions**. MotionCLIP gains its unique power by aligning its latent space with that of the Contrastive Language-Image Pre-training (CLIP) model. Aligning the human motion manifold to CLIP space implicitly infuses the extremely rich semantic knowledge of CLIP into the manifold. In particular, it helps continuity by placing semantically similar motions close to one another, and disentanglement, which is inherited from the CLIP-space structure. M**<font style="color:#000000;">otionCLIP comprises a transformer-based motion auto-encoder, trained to reconstruct motion while being aligned to its text label's position in CLIP-space.</font>** We further leverage CLIP's unique visual understanding and inject an even stronger signal through aligning motion to rendered frames in a self-supervised manner. We show that although CLIP has never seen the motion domain, MotionCLIP offers unprecedented text-to-motion abilities, allowing out-of-domain actions, disentangled editing, and abstract language specification. For example, the text prompt "couch" is decoded into a sitting down motion, due to lingual similarity, and the prompt "Spiderman" results in a web-swinging-like solution that is far from seen during training. In addition, we show how the introduced latent space can be leveraged for motion interpolation, editing and recognition.
>
> <font style="color:rgb(6, 6, 7);">MotionCLIP 是一个3D运动自编码器，它能够产生一个分离的、表现良好的潜在嵌入，这个嵌入支持高度语义化和精细的描述。为此，我们采用了CLIP，这是一个大规模的视觉-文本嵌入模型。我们的关键洞见是，尽管CLIP没有在运动领域受过训练，但我们可以通过将其强大且语义化的结构强制应用到运动领域，从而继承其潜在空间的许多优点。为了实现这一点，我们训练了一个基于Transformer的自编码器，使其与CLIP的潜在空间对齐，使用的是现有的运动文本标签。</font>
>
> <font style="color:rgb(6, 6, 7);">换句话说，我们训练了一个编码器来找到输入序列在CLIP空间中的正确嵌入，以及一个解码器，它能够为给定的CLIP空间潜在代码生成最合适的运动。为了进一步提高与CLIP空间的对齐度，我们还利用了CLIP的视觉编码器，并通过合成渲染帧以自监督的方式引导对齐。正如我们所展示的，这一步对于跨领域泛化至关重要，因为它允许对运动进行更细粒度的描述，这是使用文本无法实现的。</font>



简单来说就是用 transformer 做一个 Autoencoder，但 loss 不仅仅是常用的 Autoencoder 的 loss，还有在 latente space 对齐 CLIP 的 image 和 text embedding 的 loss。作者认为将人物动作对齐 CLIP 空间有两个好处，将几何移动的空间对齐语义空间能够让 语义描述动作 获益（也就是 motion 能够更加对齐语义）。另一方面是这么干在 motion 的 latent space 中也能够获得更好的结果。

作者给了个例子来说明学习到的语义、动作对齐得到的好结果：For example, the CLIP embedding for the phrase “wings" is decoded into a flapping motion like a bird, and “Williams sisters" into a tennis serve, since these terms are encoded close to motion seen during training, thanks to CLIP’s semantic understanding.

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1734405512561-9901e80b-1476-4d74-b914-3396d0c6bdb9.png)

#### 技术细节

<font style="color:rgb(0,0,0);">Our training process is illustrated in Figure 2. We train a transformerbased motion auto-encoder, while aligning the latent motion manifold to CLIP joint representation. We do so using </font>**<font style="color:rgb(0,0,0);">(i) a </font>**_**<font style="color:rgb(0,0,0);">Text Loss</font>**_**<font style="color:rgb(0,0,0);">,</font>**<font style="color:rgb(0,0,0);"> connecting motion representations to the CLIP embedding of their text labels, and </font>**<font style="color:rgb(0,0,0);">(ii) an </font>**_**<font style="color:rgb(0,0,0);">Image Loss</font>**_**<font style="color:rgb(0,0,0);">, </font>**<font style="color:rgb(0,0,0);">connecting motion representations to CLIP embedding of rendered images that depict the motion visually.</font>

在推理的时候，可以用 semantic editing，例如 要做一个风格迁移的话，找到一个 latent vector 表征这个 style，然后简单的将其加在 motion 的 latent vector 上。或者要做分类的话，就看在 latent space 距离那个类的 vector 更近就好。更进一步，我们用 CLIP text encoder 来做 text-to-motion：text 进来之后用 CLIP text encoder 做 encoding，然后直接输入 motion decoder。

动作序列用 [SMPL](https://smpl.is.tue.mpg.de/) 模型来表示，一个长度为$ T $的序列表示为$ p_{1:T}, p_{i} \in \mathbb{R}^{24 \times 6} $，用一个 6D 的维度 和 24 个点来表示，其中一个点定义一个身体的全局方向，其余 23 个 SMPL 点表示关节，在第 $ i $帧的结果。矩阵的节点位置$ v_{1:T} $根据 SMPL 的规定来计算（$ \beta = 0 $,无性别的身体）



##### Transformer Encoder.

Encoder 用 $ E $表示，将动作序列 $ p_{1:T} $输入得到潜空间变量 $ z_{p} $.  动作序列被一个线性转换层 独立地 嵌入了 encoder 的维度，然后加入标准的位置编码，然后和一个额外的 prefix token $ z_{tk} $ 一起输入 Transformer Encoder。 潜变量空间是 Transformer Encoder 的第一个输出$ z_p $（其余的被丢掉了）



$ z_p = E(z_{tk}, p_{1:T}) $



##### Transformer Decoder

 $ D $，基于一个输 入$ z_p $ 预测一个行动序列 $ \hat{p}_{1:T} $。这个$ z_p $输入后，作为 key 和 value，而 query 是一个简单的$ 1:T $的位置编码。Transformer 给每一帧都输出一个 representation。然后用线性层将其变为动作空间。即 $ \hat{p}_{1:T} = D(z_p) $. 接下来用一个可微分的 SMPL layer 来得到矩阵关节数位置 $ \hat{v}_{1:T} $



##### Losses 损失函数

这个 Autoencoder 通过 L2 Loss 在关节位置、方向和运动速度来重建动作序列来表征 motion。即

$ \mathcal{L}_{\text{tecon}} = \frac{1}{|p|(T)} \sum^{T}_{i=1} \parallel p_i - \hat{p}_i \parallel ^2 + \frac{1}{|v|(T)} \sum^{T}_{i=1} \parallel v_i - \hat{v}_i \parallel ^2 + \frac{1}{|p|(T-1)} \sum^{T-1}_{i=1} \parallel (p_{i+1} - p_{i}) - (\hat{p}_{i+1} -  \hat{p}_i) \parallel ^2 $

##### 
给定 text-motion 和 image-motion 的 pairs 后，$ (p_{1:T}, t), (p_{1:T}, s) $，我们利用 cosine 距离来将将动作表征和 CLIP 中 text 的向量、CLIP 中 image 的向量关联起来

$ \mathcal{L}_{\text{text}} = 1 - cos(\text{CLIP}_{text}(t), z_p) $

$ \mathcal{L}_{\text{image}} = 1 - cos(\text{CLIP}_{image}(s), z_p) $



文本数据可以从数据集中标签而来， 图像数据可以用 motion 空间渲染而来，所以总的来说，损失函数包括



$ \mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{text}} \mathcal{L}_{\text{text}} + \lambda_{
\text{image}} \mathcal{L}_{\text{image}}  $

#### 训练细节

We train a transformer auto-encoder with 8 layers for each encoder and decoder as described in Section 3. We align it with the CLIP-ViT-B/32 frozen model.

 Both $ \lambda $ values are set to 0.01 throughout our experiments.



![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1734417108930-350c0d68-f592-4424-863a-b8439621e552.png)

## SigLIP

[Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/pdf/2303.15343)

Google 做的

> 论文提出了一种简单的配对 Sigmoid 损失函数（SigLIP），用于语言图像预训练。与传统的基于 softmax 归一化的对比学习方法不同，sigmoid 损失仅在图像-文本对上操作，不需要全局视图对成对相似性进行归一化。这种损失函数允许进一步扩大批次大小，同时在较小的批次大小下也表现更好。结合锁定图像调整，使用仅有的四个 TPUv4 芯片，研究者们训练了一个 SigLiT 模型，在两天内达到了 84.5% 的 ImageNet 零样本准确率。通过将批次大小从损失中解耦，研究者们还研究了示例与对的比例以及负样本与正样本的比例的影响。最后，<font style="color:#DF2A3F;">研究者们将批次大小推至极致，高达一百万，发现增加批次大小的好处迅速减少，一个更合理的批次大小为 32k 就足够了。</font>研究者们发布了他们的模型，并希望他们的研究能激发对提高语言图像预训练质量和效率的进一步探索。

项目地址：[https://github.com/google-research/big_vision](https://github.com/google-research/big_vision)



pseudo code

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1734342621577-33e0e296-8250-4149-be12-82722fe63a1b.png)

```python
# img_emb		: image model embedding [n, dim]
# txt_emb		: text model embedding [n,, dim]
# t_prime, b	: learnable temperaature and bias
# n				: mini-batch size

t = exp(t_prime)
zimg = l2_normalize(img_emb)
ztxt = l2_normalize(txt_emb)
logits = dot(zimg, ztxt.T) * t + b
labels = 2*eye(n) - ones(n) # -1 with diagnonal 1
l = -sum(log_sigmoid(labels * logits)) /n
```



#### 训练细节

+ 发现 pretrain-backbone 的时候，不用 weight-decay 效果还好些 （个人推测可能是因为 batch 够大）
+ 发现在 batch-size 的调整中，对于这个目标函数下的训练结果而言，并不是越大越好

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1734401107570-6a0d914d-709a-444b-b72b-f2a9f557737c.png)



**类别平衡处理**

在 SigLIP 的训练中，针对每个 batch 的正负样本的不同配比做了尝试。

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1734402763778-49adaf99-eb0c-4354-bdd3-ffe2c1c878cc.png)

> We run experiments in the SigLiT setup at batch-size 16 k for 900 M steps and vary the composition of the batch by masking out (i.e. ignoring) enough negative examples to reach a target “positive : negative” ratio, masking in the following ways:
>
> <font style="color:#000000;">• </font>**<font style="color:#000000;">Random</font>**<font style="color:#000000;">: Randomly choose negative pairs to mask. 随机屏蔽掉一部分负例</font>
>
> <font style="color:#000000;">• </font>**<font style="color:#000000;">Hard</font>**<font style="color:#000000;">: Keep hardest negative pairs (highest loss). 保留最难的负例</font>
>
> <font style="color:#000000;">• </font>**<font style="color:#000000;">Easy</font>**<font style="color:#000000;">: Keep easiest negatives pairs (lowest loss). 保留最简单的负例</font>
>
> <font style="color:#000000;">• </font>**<font style="color:#000000;">Hard + matching total pairs seen</font>**<font style="color:#000000;">: Masking examples while training for a fixed number of steps does decrease the total number of pairs seen during training. Hence in the matched pairs setting, we increase the number of training steps by the masking ratio in order to keep the number of pairs seen constant. 由于在每个批次中masking掉一些负例会减少模型在训练过程中看到的总对数（pairs seen），这可能会影响模型的学习效果。为了补偿这种减少，研究者们提出增加训练的步数（training steps）。具体来说，如果通过masking减少了一定比例的负例，那么他们会增加相应的训练步数，以确保模型在整个训练过程中看到的正例和负例的总对数保持不变。</font><font style="color:rgb(6, 6, 7);">这种方法的目的是，在减少负例数量以平衡正负例比例的同时，通过增加训练步数来保持模型接触到的总数据量不变，从而尽可能地减少由于masking导致的信息损失。这样可以帮助模型更有效地从难区分的负例中学习，同时保持训练数据的丰富性。</font>





#### 技术细节

原本的 softmax 目标函数

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1734343285313-b6ef7240-f9f1-4470-bf22-aff799627370.png)

$ f $是 image encoder，$ g $是 text encoder。



新的 sigmoid loss 目标函数

> The sigmoid-based loss processes every image-text pair independently, effectively turning the learning problem into the standard binary classification on the dataset of all pair combinations, with a positive labels for the matching pairs ($ I_i $,$ T_i $) and negative labels for all other pairs ($ I_i $,$ T_{j \ne i} $). It is defined as follows

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1734343466468-e5bbe006-201e-48bc-9190-2050f8f7a480.png)



$ z_{ij} $是给定的 image、text 对的 label，如果是成对的就是 1，不成对就是-1. 由于类别不平衡，引入了可学习的 t 和 b ，初始设置为 log10 和-10，这样会让开始的学习比较平衡。





**有效地 Chunk implementation**

Contrastive loss 的训练很明显用到了数据并行的策略，由于要搜集所有设备上的 embedding 结果，很明显通信开销会很大，而且要做 Contrastive loss 这样的计算，要有 $ B^2 $的针对相似度的计算量

然而 sigmoid loss 的内存效率较高，每个 device 上，分到的$ b=\frac{B}{D} $的数据，loss 函数可以被重写为：

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1734344111883-21c5e0d4-74da-4c2c-93a3-06df43df7299.png)

每个对就是独立的，不用看别的对，通信开销大大降低，有以下

+ D （每个 device）
+ D （要交换复例）
+ 本地的正例 + 其他的 device 的负例

也就是正例本地算，负例从其他 device 来。

![](https://cdn.nlark.com/yuque/0/2024/png/2379769/1734343254126-e89d7de1-1932-45ce-a75e-03ea016a0138.png)





使用方法

[https://hugging-face.cn/docs/transformers/model_doc/siglip](https://hugging-face.cn/docs/transformers/model_doc/siglip)

```python
from transformers import pipeline
from PIL import Image
import requests

# load pipe
image_classifier = pipeline(task="zero-shot-image-classification", model="google/siglip-base-patch16-224")

# load image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# inference
candidate_labels = ["2 cats", "a plane", "a remote"]
outputs = image_classifier(image, candidate_labels=candidate_labels)
outputs = [{"score": round(output["score"], 4), "label": output["label"] } for output in outputs]
print(outputs)
[{'score': 0.1979, 'label': '2 cats'}, {'score': 0.0, 'label': 'a remote'}, {'score': 0.0, 'label': 'a plane'}]
```



# 总结

可以看到 CLIP 的出世，引发了很多很多的后续工作，无论是从各个角度尝试改善 CLIP 的，包括但不限于

+ 长度：long-clip
+ 粒度：GLIP，GLIP2 等，PyramidCLIP
+ 任务复杂度：BLIP、GLIP 系列
+ 标签的硬度：PyramidCLIP、SoftCLIP
+ 计算效率：SigLIP 等

以及领域特有的 CLIP，例如医疗领域。

甚至有的还会利用 CLIP 将原本的工作流程对齐 CLIP 的潜变量空间（MotionCLIP），来获得动作、语言和视觉语义的对齐。



























