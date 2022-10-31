import copy
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
######
PAD_IDX=0
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
# Transformers
from typing import *
'''
Transformer
- Encoder
    - input : sentence -> context vector => "문장의 문맥 함축"
    - context vector를 제대로 생성해 압축해내는 것을 목표

- Decoder
    - input : context vector, sentence -> sentence
'''
class Transformer(nn.Module):

    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    # c = Encoder(x)  / x: sentence, c: context
    def encode(self, x):
        out = self.encoder(x)
        return out

    # y = Decoder(c, z) / y,z: sentence, c: context
    def decode(self, z, c):
        out = self.decode(z, c)
        return out

    def forward(self, x, z):
        c = self.encode(x)
        y = self.decode(z, c)
        return y
'''
Encoder
- Encoder Block이 N개 쌓인 형태
    - input과 output의 형태 동일 -> Encoder Block은 shpae에 대해 멱등함
    - 첫번째 Encoder Block의 input은 전체 Encoder의 input으로 들어오는 문장 Embedding이 됨
    - 첫번째 block이 output을 생성하면, 이를 두번째 block의 input으로 사용....반복
    - 가장 마지막 N번째 block의 output이 전체 Encoder의 output -> 즉, context vector가 됨
    - context vector도 Encoder의 input sentence와 동일한 shape을 가짐 -> Encoder 전체는 shape에 대해 멱등함
- Encoder Block을 왜 겹쳐 쌓을까
    - input으로 들어오는 vector를 겹겹이 context vector를 쌓아 낮은 수준의 context에서 높은 수준의 context로 저장되게 함
'''
class Encoder(nn.Module):

    def __init__(self, encoder_block, n_layer): # n_layer: Encoder Block의 개수
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_block))

    # forward: Encoder Block을 순서대로 실행하며, 이전 block의 output을 이후 block의 input으로 넣음
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

'''
Encoder Block
- Multi-Head Attention Layer, Position-wise Feed-Forward Layer로 구성됨
'''
class EncoderBlock(nn.Module):

    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff

    def forward(self, x):
        out = x
        out = self.self_attention(out)
        out = self.position_ff(out)
        return out
'''
Multi-head Attention
- Scaled Dot-Product-Attention을 병렬적으로 여러 개 수행하는 layer
- Attention
    - 넓은 범위의 전체 data에서 특정한 부분에 집중
    - Scaled Dot-Product Attention 자체를 줄여서 Attention으로 부르기도 함
    - ex) The animal didn't cross the street, because it was too tired
        : 'it'과 'animal'을 연결시키기 위해 두 token 사이의 연관 정도를 계산해내는 방법론
        : 같은 문장 내의 두 token 사이의 Attention을 계산하는 것 -> Self-Attention이라 표현
        : 서로 다른 두 문장에 각각 존재하는 두 token 사이의 Attetntion -> Cross-Attention이라 표현
- RNN vs Self-Attention
: RNN의 hidden state와 같이, RNN과 Self-Attention의 목적은 같음.
: Self-Attention은 2가지 장점을 가짐
    1.
        - RNN은 i시점의 hidden state h_(i)를 구하기 위해서는 h_(i-1)가 필요.
        - RNN은 앞에서부터 순차 계산을 하며 구하는 방법밖에 없어 병렬 처리가 불가능
        - Self-Attention은 모든 token 쌍 사이의 attention을 한 번의 행렬 곱으로 구해내기 때문에 병렬 처리가 가능
    2.  
        - RNN은 시간이 진행될수록 오래된 시점의 token에 대한 정보가 점차 희미해짐. 
        - 서로 거리가 먼 token 사이의 관게에 대한 정보는 제대로 반영되지 못함
        - Self-Attention은 문장에 token이 n개 있다고 가정할 경우, n*n번 연산을 수행해 모든 token들 사이의 관계를 직접 구함
        - 중간의 다른 token들을 거치지 않고 바로 direct한 관계를 구하는 것이기 때문에 RNN에 비해 더 명확하게 관계를 구할 수 있음

Attention
:About Self-Attetnion
ex) The animal didn't cross the street, because it was too tired
- Attention
    - Query vector: 현재 시점의 token
    - Key vector: attention을 구하고자 하는 대상 token
    - Value vector: attention을 구하고자 하는 대상 token을 의미(Key와 동일한 token)
    : Key와 Value는 문장의 처음부터 끝까지 탐색
    : Query는 고정되어 하나의 token을 가리킴
    => Query와 가장 부합하는(Attention이 가장 높은)token을 찾기 위해 Key, Value는 문장의 처음부터 끝까지 탐색
    : input으로 들어오는 token embedding vector를 fully connected layer에 넣어 3가지 vector를 생성.
    : 3가지 vector를 생성해내는 FC layer는 모두 다르기 때문에, 3개의 서로 다른 FC layer(Query, Key, Value)가 존재
    -> FC layer들은 모두 같은 input shape, output shape을 가짐 -> 3가지의 vector의 dimension을 d_k로 명명
    -> d_k의 k는 Kyer를 의미. -> 논문의 notation에서 이를 채택했기 때문에 k로 함.
- Pad Masking
    - 길이(seq_len)를 맞추기 위한 padding
    - padding된 것을 attention을 부여되지 않게 하기 위한 Pad Masking -> softmax 이전에는 반드시 Masking이 수행되어야 함.
'''
# model에 들어오는 input은 mini-batch이기 때문에, Q,K,V의 shape에 n_batch가 추가됨
# Q,K,V는 n_batch*seq_len*d_k의 shape을 가짐
def calculate_attention(query, key, value, mask):
    # query, key, value: (n_batch, seq_len, d_k)
    # mask: (n_batch, seq_len, seq_len)
    d_k = key.shape[-1]
    attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q * K^T, (n_batch, seq_len, seq_len)
    attention_score = attention_score / math.sqrt(d_k)
    if mask is not None:
        attention_score = attention_score.masked_fill(mask==0, -1e9)
    attention_prob = F.softmax(attention_score, dim=-1) # (n_batch, seq_len, seq_len)
    out = torch.matmul(attention_prob, value) # (n_batch, seq_len, d_k)
    return out
'''
Multi-Head Attention Layer
* Transformer는 Scaled Dot Attention을 한 Encoder Layer마다 1회씩 수행하는게 아니라, 병렬적으로 h회 각각 수행하고 결과를 종합해 사용됨
    - 최종적으로 생성된 matrix(n*d_model)를 FC layer에 넣어 multi-head attention의 input과 같은 shape(n*d_embed)의 matrix로 변환하는 과정 필요
    - 마지막 FC lyaer의 input dimension은 d_model, output dimension은 d_embed가 됨
'''
class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_model, h, qkv_fc, out_fc):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        # deepcopy 사용-> 실제로 서로 다른 weight를 갖고 별개로 운용되게 하기 위함.
        # copy없이 구하게 되면 항상 Q,K,V가 모두 같은 값이 됨.
        self.q_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.k_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.v_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.out_fc = out_fc # (d_model, d_embed)

    def forward(self, *args, query, key, value, mask=None):
        # query, key, value: (n_batch, seq_len, d_embed) -> input sentence embedding
        # mask: (n_batch, seq_len, seq_len) -> mini-batch & (seq_len, seq_len)
        # return value: (n_batch, h, seq_len, d_k)
        n_batch = query.size(0)

        def transform(x, fc): # (n_batch, seq_len, d_embed)
            out = fc(x) # (n_batch, seq_len, d_model)
            out = out.view(n_batch, -1, self.h, self.d_model//self.h) # (n_batch, seq_len, h, d_k) -> d_model을 h와d_k로 분리. 각각을 하나의 dimension으로.
            out = out.transpose(1, 2) # (n_batch, h, seq_len, d_k)
            return out

        query = transform(query, self.q_fc) # (n_batch, h, seq_len, d_k)
        key = transform(key, self.k_fc) # (n_batch, h, seq_len, d_k)
        value = transform(value, self.v_fc) # (n_batch, h, seq_len, d_k)

        out = self.calculate_attention(query, key, value, mask) # (n_batch, h, seq_len, d_k)
        out = out.transpose(1,2) # (n_batch, h, seq_len, d_k) -> h와 seq_len의 순서를 바꾸고 h와 d_k를 d_model로 결합
        out = out.contiguous().view(n_batch, -1, self.d_model) # (n_batch, seq_len, d_model)
        out = self.out_fc(out) # (n_batch, seq_len, d_embed) -> FC layer을 거쳐 d_model을 d_embed로 변환
        return out
'''
EncoderBlock 수정
- forward()인자는 x, mask가 됨
- query, key, value를 받아야하므로 인자 변환
- Multi-Head Attention Layer의 forward()인자는 최종적으로 x, x, x, mask가 됨
'''
class EncoderBlock(nn.Module):

    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff

    def forward(self, src, src_mask):
        out = src
        out = self.self_attention(query=out, key=out, value=out, mask=src_mask)
        out = self.position_ff(out)
        return out
'''
Encoder 수정
- mask 인자받기 위해 forward()인자에 mask추가
- 이를 각 sublayer의 forward()에 넘겨줌
'''
class Encoder(nn.Module):

    def __init__(self, encoder_layer, n_layer): # n_layer: Encoder Layer의 개수
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_layer))

    def forward(self, src, src_mask):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)
        return out
'''
Transform 수정
- forward()인자에 src_mask 추가
- 이를 encoder의 forward()에 넘겨줌
'''
class Transformer(nn.Module):

    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    # c = Encoder(x)  / x: sentence, c: context
    def encode(self, src, src_mask):
        out = self.encoder(src, src_mask)
        return out

    # y = Decoder(c, z) / y,z: sentence, c: context
    def decode(self, z, c):
        out = self.decode(z, c)
        return out

    def forward(self, src, tgt, src_mask):
        encoder_out = self.encode(src, src_mask)
        y = self.decode(tgt, encoder_out)
        return y
'''
Pad Mask Code -> pad masking 생성
make_pad_mask()
- 인자
    - query, key
    - 각각은 n_batch * seq_len의 shape
- embedding을 획득하기 전, token sequence 상태로 들어옴
- <pad>의 index: pad_idx
- pad_idx와 일치하는 token들은 모두 0, 그 외에는 모두 1인 mask 생성
'''
def make_pad_mask(self, query, key, pad_idx=1):
    # query : (n_batch, query_seq_len)
    # key : (n_batch, key_seq_len)
    query_seq_len, key_seq_len = query.size(1), key.size(1)

    key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2) # (n_batch, 1, 1, key_seq_len)
    key_mask = key_mask.repeat(1, 1, query_seq_len, 1) # (n_batch, 1, query_seq_len, key_seq_len)

    query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3) # (n_batch, 1, query_seq_len, 1)
    query_mask = query_mask.repeat(1, 1, 1, key_seq_len) # (n_batch, 1, query_seq_len, key_seq_len)

    mask = key_mask & query_mask
    mask.requires_grad = False
    return mask
'''
Cross-Attention의 경우 query는 source, key는 target과 같이 서로 다른 값이 들어올 수 있음
'''
def make_src_mask(self, src):
    pad_mask = self.make_pad_mask(src, src)
    return pad_mask
'''
Position-wise Feed Forward Layer
- 단순하게 2개의 FC Layer를 갖는 Layer
- 각 FC Layer은 (d_embed * d_ff), (d_ff * d_ember)의 weight matrix를 가짐
- Multi-Head Attention Layer의 output을 input으로 받아 연산 수행 -> 다음 Encoder Block에게 output을 넘겨줌
- 첫 번째 FC Layer의 output에 ReLU()를 적용
'''
class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self, fc1, fc2):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = fc1 # (d_embed, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = fc2 # (d_ff, d_embed)

    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
'''
Residual Connection Layer
- Multi-Head Attention Layer과 같이 Encoder Block을 구성하는 Layer 
- y = f(x) 를 y = f(x) + x로 변경하는 것
    - output을 그대로 이용하지 않고, output에 input을 추가적으로 더한 값을 사용
    - Back Propagation 도중 발생할 수 있는 Gradient Vanishing을 방지 가능
'''
class ResidualConnectionLayer(nn.Module):

    def __init__(self):
        super(ResidualConnectionLayer, self).__init__()

    def forward(self, x, sub_layer):
        out = x
        out = sub_layer(out)
        out = out + x
        return out
'''
Encoder Block 수정
- residuals에 Residual Connection Layer 2개 생성
- forward()
    - residuals[0]은 multi_head_attention_layer을 감싸고
    - residuals[1]은 position_ff를 감싼다
'''
class EncoderBlock(nn.Module):

    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer() for _ in range(2)]

    def forward(self, src, src_mask):
        out = src
        out = self.residuals[0](out, lambda out: self.self_attention(query=out, key=out, value=out, mask=src_mask))
        out = self.residuals[1](out, self.position_ff)
        return out
'''
Decoder
- Teacher Forcing in Transformer(Subsequent Maksing)
    - ground turth의 embedding을 matrix로 만들어 input으로 그대로 사용 시, 
      Decoder에서 Self-Attention 연산 수행할 때 출력해야하는 정답까지 알아야 하는 상황 발생
      => Masking을 적용해야 함
    - i번쨰 token 생성 시, 1 ~ i-1의 token은 보이지 않도록 처리를 해야 하는 것 => subsequent masking 이라 표현
'''
def make_subsequent_mask(query, key):
    # query: (n_batch, query_seq_len)
    # key: (n_batch, key_seq_len)
    query_seq_len, key_seq_len = query.size(1), key.size(1)

    tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8') # lower triangle without diagonal
    mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)
    return mask
# Decode의 mask는 subsequent masking과 pad masking이 적용되어야 함
def make_tgt_mask(self, tgt):
    pad_mask = self.make_pad_mask(tgt, tgt)
    seq_mask = self.make_subsequent_mask(tgt, tgt)
    mask = pad_mask & seq_mask
    return mask
'''
Transformer 수정
- Decoder에서 사용할 subsequent+pad mask(tgt_mask)
- forward()
    - Decoder의 forward()호출 시 tgt_mask 인자 추가
'''
class Transformer(nn.Module):

    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, src, src_mask):
        out = self.encoder(src, src_mask)
        return out

    def decode(self, tgt, encoder_out, tgt_mask):
        out = self.decode(tgt, encoder_out, tgt_mask)
        return out

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        encoder_out = self.encode(src, src_mask)
        y = self.decode(tgt, encoder_out, tgt_mask)
        return y
'''
Decoder Block
- Multi-Head Attention Layer이 2개 존재
    - 첫번째 : Self-Multi-Head Attention Layer
        : Deccoder의 input으로 주어지는 sentence 내부의 Attention 계산
        : mask로 들어오는 인자가 pad masking + subsequent masking 적용됨
        : Ground Truth sentence에 내부에서의 Attention을 계산
    - 두번째: Cross-Multi-Head Attention Layer
        : Encoder에서 주어진 context를 Key, Value로 사용
- Position-wise Feed-Forward Layer
'''
class DecoderBlock(nn.Module):

    def __init__(self, self_attention, cross_attention, position_ff):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer() for _ in range(3)]

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        out = self.residuals[0](out, lambda out: self.self_attention(query=out, key=out, value=out, mask=tgt_mask))
        out = self.residuals[1](out, lambda out: self.cross_attention(query=out, key=encoder_out, value=encoder_out, mask=src_tgt_mask))
        out = self.residuals[2](out, self.position_ff)
        return out
'''
Factory Method
- Transformer를 생성하는 build_model() 작성
'''
def build_model(src_vocab_size, tgt_vocab_size, device=torch.device('cpu'), max_len=256, d_embed=512, n_layer=6, d_model=512, h=8, d_ff=2048):
    import copy
    copy = copy.deepcopy

    src_token_embed = TokenEmbedding(d_embed = d_embed, vocab_size = src_vocab_size)
    tgt_token_embed = ToeknEmbedding(d_embed = d_embed, vocab_size = tgt_vocab_size)

    pos_embed = PositionalEncoding(d_embed = d_embed, max_len = max_len, device = device)
    src_embed = TransformerEmbedding(token_embed = src_token_embed, pos_embed = copy(pos_embed))
    tgt_embed = TransformerEmbedding(token_embed = tgt_token_embed, pos_embed = copy(pos_embed))

    attention = MultiHeadAttentionLayer(d_model = d_model, h = h, qkv_fc= nn.Linear(d_embed, d_model), out_fc = nn.Linear(d_model, d_embed))
    position_ff = PositionWiseFeedForwardLayer(fc1 = nn.Linear(d_embed, d_ff), fc2 = nn.Linear(d_ff, d_embed))

    encoder_block = EncoderBlock(self_attention = copy(attention), position_ff = copy(position_ff))
    decoder_block = DecoderBlock(self_attention = copy(attention), cross_attention = copy(attention), position_ff = copy(position_ff))

    encoder = Encoder(encoder_block = encoder_block, n_layer = n_layer)
    decoder = Decoder(decoder_block = decoder_block, n_layer = n_layer)
    generator = nn.Linear(d_model, tgt_vocab_size)

    model = Transformer(
        src_embed = src_embed,
        tgt_embed = tgt_embed,
        encoder = encoder,
        decoder = decoder,
        generator = generator
    ).to(device)

    model.device = device

    return model
