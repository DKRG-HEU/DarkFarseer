import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class SeTS(nn.Module):
    '''
    # Style-enhanced Temporal-then-spatial Architecture (SeTS)
    Please note that to achieve better performance, we have incorporated hidden dimensions based on DLinear.
    > DLinear Link: https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py

    ## Input:
    * imputation: hidden state of kriging model
    * x_local: original sequences of virtual Nodes' 1-hop neighbors
    * khop_index: indices of the former

    ## Output:
    * enhanced_hidden_state: its dimensions remain consistent with those during input.
    '''
    def __init__(self) -> None:
        super(SeTS, self).__init__()

        self.prompt_embedding_dim = 64
        self.time_window = 24

        self.linear1 = nn.Linear(1, self.prompt_embedding_dim)
        self.linear2 = nn.Linear(1, self.prompt_embedding_dim)
        kernel_size = 17
        self.decompsition = series_decomp(kernel_size)
        self.Linear_Seasonal = nn.Linear(self.time_window, self.prompt_embedding_dim)
        self.dropout_seasonal = nn.Dropout(p=0.2)
        self.Linear_Seasonal_Dense = nn.Linear(self.prompt_embedding_dim, self.time_window)
        
        self.Linear_Trend = nn.Linear(self.time_window, self.prompt_embedding_dim)
        self.dropout_trend = nn.Dropout(p=0.2)
        self.Linear_Trend_Dense = nn.Linear(self.prompt_embedding_dim, self.time_window)

    def forward(self, imputation, x_local, khop_index):
        bb, ss, nn = x_local.size()

        if x_local.size(2) == 0:
            return imputation
        
        seasonal_init, trend_init = self.decompsition(x_local)

        # merge the batch_size and num_nodes dimensions for Channel Independent
        seasonal_init = rearrange(seasonal_init, 'b s n -> (b n) s 1')
        trend_init = rearrange(trend_init, 'b s n -> (b n) s 1')

        seasonal_init = self.linear1(seasonal_init)
        trend_init = self.linear2(trend_init)

        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)

        seasonal_output = self.dropout_seasonal(F.relu(self.Linear_Seasonal(seasonal_init)))
        seasonal_output = self.Linear_Seasonal_Dense(seasonal_output)
        trend_output = self.dropout_trend(F.relu(self.Linear_Trend(trend_init)))
        trend_output = self.Linear_Trend_Dense(trend_output)

        seasonal_output, trend_output = seasonal_output.permute(0,2,1), trend_output.permute(0,2,1)
        styled_mts = seasonal_output + trend_output
        styled_mts = rearrange(styled_mts, '(b n) s d -> b d s n', b=bb, n=nn)

        khop_index = khop_index.unsqueeze(-1)
        indices = torch.nonzero(khop_index, as_tuple=True)[1]

        batch_size, hidden_size, time_len, _ = styled_mts.size()
        expanded_mts = torch.ones((batch_size, hidden_size, time_len, khop_index.size(1)), device=styled_mts.device)
        
        # Assign the values from styled_mts to the correct positions in expanded_mts
        expanded_mts[:, :, :, indices] = F.sigmoid(styled_mts)

        imputation = imputation * expanded_mts  # b d s n
        
        return imputation