import torch as t
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
import numpy as np
import math

device = 'cuda' if t.cuda.is_available() else 'cpu'

class PD_Block(nn.Module):

    """
    The block is designed to comply to the post-LN residual block structure in Transformers and the LRU.
    """

    def __init__(self,
                 embed_size: int,
                 hidden_size: int,
                 dictionary_size: int = 8,
                 hidden_D_multiple: int = 2,
                 dropout_rate: float = 0.01,
                 **kwargs):
        
        super(PD_Block, self).__init__()

        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.dict_size = dictionary_size
        
        # Post-norm
        self.norm = nn.LayerNorm(embed_size)

        self.drop = nn.Dropout(p=dropout_rate)
        self.dropout_rate = dropout_rate

        # GELU-activated MLP generates the magnitudes of the entries of the diagonal matrices
        # The magnitudes are constrained to the interval (0,1) using the sigmoid function
        self.D_magnitude_generator = nn.Sequential(
            nn.Linear(embed_size, hidden_D_multiple * hidden_size),
            nn.GELU(),
            nn.Linear(hidden_D_multiple * hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # GELU-activated MLP generates the phases of the entries of the diagonal matrices
        # The phases are constrained to the interval (0,2*pi) using the sigmoid function followed by scaling
        self.D_phase_generator = nn.Sequential(
            nn.Linear(embed_size, hidden_D_multiple * hidden_size),
            nn.GELU(),
            nn.Linear(hidden_D_multiple * hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # Selector of dictionary matrices
        self.S = nn.Linear(embed_size, dictionary_size, bias=False)

        # Matrix dictionary
        self.A_dict = nn.Parameter(t.randn(hidden_size, hidden_size, dictionary_size)/np.sqrt(hidden_size))   

        # Glorot initialization with halved variance
        self.B_re = nn.Parameter(t.randn(hidden_size, embed_size)/np.sqrt(2*hidden_size))
        self.B_im = nn.Parameter(t.randn(hidden_size, embed_size)/np.sqrt(2*hidden_size))

        # Glorot initialization with halved variance
        self.D = nn.Parameter(t.randn(embed_size)/np.sqrt(embed_size))

        # Readout of complex-valued states
        self.readout = nn.Linear(2*embed_size, embed_size)

    def forward(self, x: t.Tensor) -> t.Tensor:
        
        """

        Input:  x (B, L, E) - Batched input sequence tensor
        Output: y (B, L, E) - Batched output sequence tensor


        Notation:

        B: Batch Size
        L: Sequence Length
        E: Embedding Dimension
        N: Hidden State Dimension
        K: Transition Matrix Dictionary Size
        """

        B, L, E = x.shape
        K = self.dict_size
        N = self.hidden_size

        """
        Initializing the hidden states tensor and the one-hot initial hidden state
        """
        
        # Reserve space for the hidden states
        hidden_states = t.complex(real=t.zeros(B, L, N), imag=t.zeros(B, L, N)).to(device)

        # Initialize x_0 to the one-hot vector [1,0,...,0]
        # B x N
        hidden_state = t.zeros(B, N).to(device)
        hidden_state[:,0] = 1
        hidden_state = t.complex(real=hidden_state, imag=t.zeros_like(hidden_state))

        """
        Generating the M matrices
        """

        # B x L x K
        selection_weights = F.softmax(self.S(x), dim=-1)

        # B x L x N x N
        M =  einsum(self.A_dict, selection_weights, 'n1 n2 k, b l k -> b l n1 n2')

        """
        Transforming the M matrix into one-hot form while keeping the gradient path in accordance with the column-wise softmax of M
        """

        y_soft = F.softmax(M, dim=-1)
        num_classes = y_soft.shape[-1]
        y_hard = t.argmax(y_soft, dim=-1)

        # Conversion to one-hot vectors
        y_hard = F.one_hot(y_hard, num_classes=num_classes)
        # B x L x N x N
        P = (y_hard - y_soft).detach() + y_soft

        """
        Generating the complex-valued diagonal matrices
        """

        # B x L x N
        magnitudes_raw = self.D_magnitude_generator(x)
        magnitudes = t.complex(real=magnitudes_raw, imag=t.zeros_like(magnitudes_raw))

        # B x L x N
        phases_raw = 2*math.pi*self.D_phase_generator(x)
        phases = t.exp(t.complex(real=t.zeros_like(phases_raw), imag=phases_raw))

        # B x L x N
        D = magnitudes * phases

        """
        Combining the P and D matrices 
        """

        # B x L x N x N
        transition_matrices = D.unsqueeze(-1) * P

        """
        Transform the input
        """

        # Input transformation
        # H x E
        B_mat = t.complex(real = self.B_re, imag = self.B_im)

        # B x L x H
        b = t.matmul(B_mat.view(1, 1, N, self.embed_size).expand(B, L, -1, -1), t.complex(real=x, imag=t.zeros_like(x)).view(B, L, E, 1)).squeeze(-1)

        """
        Sequential recurrence
        """

        for i in range(L):
            # Compute the hidden state at time t
            hidden_state = t.einsum('bmn, bn -> bm', transition_matrices[:,i,:,:], hidden_state) + b[:,i,:]
            # Store it in the initialized tensor
            hidden_states[:,i,:] = hidden_state

        """
        Linear readout
        """

        # B x L x 2E
        linear_readout_re_im = t.cat((t.real(hidden_states), t.imag(hidden_states)), dim=-1)

        # B x L x E
        output = self.readout(self.drop(linear_readout_re_im)) + self.D * x

        return self.norm(output)

class PD(nn.Module):
    def __init__(self,
                 output_size: int,
                 input_size: int,
                 state_size: int,
                 embed_size: int,
                 dictionary_size: int = 8,
                 return_all_outputs: bool = False,
                 num_layers: int = 2,
                 dropout_rate: float = 0.01,
                 **kwargs):
        
        super(PD, self).__init__()

        self.return_all_outputs = return_all_outputs
        self.num_layers = num_layers

        self.blocks = nn.ModuleList(PD_Block(
            embed_size=embed_size, 
            hidden_size=state_size, 
            dictionary_size=dictionary_size, 
            dropout_rate = dropout_rate) for _ in range(num_layers))
        
        self.embedding = nn.Embedding(input_size, embed_size)
        
        self.readout = nn.Linear(embed_size, output_size)

    def mask_grads(self):
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:

        x = self.embedding(x)

        for i in range(self.num_layers):
            x = self.blocks[i](x)

        output = self.readout(x)

        if not self.return_all_outputs:
            return output[:, -1, :]
        else:
            return output
        
