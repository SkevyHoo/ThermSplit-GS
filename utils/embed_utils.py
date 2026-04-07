import torch
import torch.nn as nn

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    

def get_embedder(multires, i=1):
    # 如果包含原始输入：i * (1 + 2 * multires)
    # 如果不包含原始输入：i * 2 * multires
    # 其中i 是输入维度，multires是频率数量。
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim 
    
    
if __name__ == "__main__":
    embed, out_dim = get_embedder(10, 3)
    print(f"output dimension: {out_dim}")  # output dimension: 63
    
    # random_2d = torch.rand(3, 100)
    
    input= torch.rand(100).unsqueeze(0).repeat(3, 1)
    print(f"Input shape: {input.shape}")  # torch.Size([3, 100])
    
    output = embed(input) 
    print(f"Output shape: {output.shape}") # torch.Size([3, 2100])
   