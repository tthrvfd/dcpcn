import torch
from torch import cosine_similarity, nn
from torch.nn import functional as F
import sys
import distributed as dist_fn

class Quantize(nn.Module):  
    """
    作用：输入特征，然后进行离散化
    输入特征为 B m n DIM
    离散特征为 K DIM
    """
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())
    
    def merge(self,othervq,vqindex,referenceNum):
        
        feat = othervq.embed_code(vqindex).detach() # B 128 C
        flatten = feat.reshape(-1,self.dim) # B * 128 C
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )# B * 128  8192
        _,embed_index = torch.topk(-dist,k=referenceNum) # B * 128
        embed_c = self.embed.transpose(0,1)
        embed_c = embed_c.detach()
        for id in range(referenceNum):
            embed_c[embed_index[:,id]] = embed_c[embed_index[:,id]] * 0.99999 + flatten *0.00001
        embed_c = embed_c.transpose(0,1)
        self.embed.data.copy_(embed_c)
    
    def adaptiveMerge(self, feat):
        flatten = feat.reshape(-1,self.dim)
        flatten, inverse_indices = torch.unique(flatten, dim=0, return_inverse=True, sorted=True)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _,embed_index = torch.topk(-dist,k=1)
        embed_c = self.embed.transpose(0,1)
        embed_c = embed_c.detach()
        origin = embed_c[embed_index[:,0]]
        toAdd = flatten
        cosine_similarities = F.cosine_similarity(origin, toAdd, dim=1).unsqueeze(1)
        # 计算每行的余弦距离
        cosine_distances = 1 - cosine_similarities
        # 根据余弦距离计算自适应系数
        adaptive_coefficients_1 = (cosine_distances) / 2
        adaptive_coefficients_2 = 1 - adaptive_coefficients_1
        # print(adaptive_coefficients_1)
        # 进行自适应相加
        result_matrix = 0.999 * origin + 0.001*(adaptive_coefficients_1 * origin + \
                        adaptive_coefficients_2 * toAdd)
        embed_c[embed_index[:,0]] = result_matrix
        embed_c = embed_c.transpose(0,1)
        self.embed.data.copy_(embed_c)
       
    def getDim(self):
        return self.dim
    
    def getEmbed(self):
        return self.embed

    # def setEmberd


    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)# 512
            embed_sum = flatten.transpose(0, 1) @ embed_onehot  # 64 512
            
            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))
    
if __name__ == "__main__":
    a = torch.randn(5,128,756)
    b = Quantize(756,1024)
    b.adaptiveMerge(a)
    # print(a)
    # t,_,index = b(a)
    # print(t)
    # print(a)
    # print(index.shape)
    # b.merge(b,index,10)
