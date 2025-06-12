import torch
import torch.nn as nn

class Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean',
                prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None,
                  top_k=None, batchwise_prompt=False, prompt_key_init='uniform',):
        super().__init__()

        self.length = length
        self.embed_dim_prompt = embed_dim # 768
        self.embed_dim_key = 512
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        
        self.top_k = top_k

        self.batchwise_prompt = batchwise_prompt

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, self.embed_dim_prompt)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
        
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, self.embed_dim_key)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            # removed others by parisa
            x_embed_mean = cls_features

            prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C

            similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
            # x_embed_norm: [B, 512], prompt_norm: [768, 10]
            
            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
            else:
                idx = prompt_mask # B, top_k
                
            batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

            out['prompt_idx'] = idx

            # Debugging, return sim as well
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx] # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = torch.sum(sim) / cls_features.shape[0] # Scalar

            out['reduce_sim'] = reduce_sim
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        # i change the torch.cat([batched_prompt, x_embed], dim=1) to this bellow
        out['prompted_embedding'] = batched_prompt

        return out