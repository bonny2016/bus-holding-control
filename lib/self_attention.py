import torch
import torch.nn as nn

torch.set_printoptions(precision=2, sci_mode=False)
class SelfAttention(nn.Module):
    #head_dim = n_stations + 1, heads = n_buses
    def __init__(self, head_dim, heads):
        super(SelfAttention, self).__init__()
        # self.embed_size = embed_size
        self.heads = heads
        self.head_dim = head_dim
        self.embed_size = heads * head_dim
        proj_d = 6
        # self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, proj_d, bias=False)
        self.queries = nn.Linear(self.head_dim, proj_d, bias=False)
        self.fc_out = nn.Linear(self.embed_size , self.embed_size)

    def forward(self, values, keys, queries):
        # value shape: (N, self.heads, self.head_dim)
        # key shape: (N, self.heads, self.head_dim)
        # query shape: (N, self.heads, self.head_dim)

        # Get number of training examples (batch_number:1 or 64)
        N = queries.shape[0]

        # get len of value, key and query. (time steps)
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        
        #project into some other space, so that they can be compared with. TODO: change dim to smaller 
        # values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, head_dim)
        queries = self.queries(queries)  # (N, query_len, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqd,nkd->nqk", [queries, keys])
        # queries shape: (N, heads, heads_dim),
        # keys shape: (N, heads, heads_dim)
        # energy: (N, heads, heads)

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)
        # attention shape: (N, heads, query_len, key_len)
        
        # print("---------------------original value:---------------------")
        # print(values)
        # print("attention:")
        # print(attention)
        out = torch.einsum("nqk,nkh->nqh", [attention, values])
        # print("attentioned value:")
        # print(out)
        out = out.reshape(
            N, self.heads * self.head_dim
        )
        
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)
        return out

class TransformerBlock(nn.Module):
    # steps, heads, head_dim = shape
    def __init__(self, head_dim, heads, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(head_dim, heads)
        self.embed_size = head_dim * heads
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, forward_expansion * self.embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * self.embed_size, self.embed_size),
        )

    def forward(self, value, key, query):
        shape = query.shape
        attention = self.attention(value, key, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.norm1(attention + query.view(shape[0],-1))
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out

class Encoder(nn.Module):
    def __init__(
        self,
        head_dim, 
        heads,
        num_layers,
        device,
        forward_expansion
    ):

        super(Encoder, self).__init__()
        self.embed_size = head_dim * heads
        self.device = device
        self.position_embedding = nn.Embedding(heads, head_dim)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    head_dim,
                    heads,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )


    def forward(self, x):
        #assume steps is 1
        N, steps, heads, head_dim = x.shape
        # print("---------------------------------------")
        # print("initial x.shape:", x.shape)
        x = torch.squeeze(x, 1)
        # print("squeezed x.shape:", x.shape)
        N, heads, head_dim = x.shape
        positions = torch.arange(0, heads).expand(N, heads).to(self.device)
        x = x.to(self.device)
        pos_emb = self.position_embedding(positions)
        out = x + pos_emb
       
        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out)

        return out