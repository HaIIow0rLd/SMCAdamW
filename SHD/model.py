import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate

class SNN(nn.Module):
    def __init__(self, input_channels=700, hidden_channels=256, num_classes=20, T=16, tau=2.0):
        super().__init__()
        self.T = T
        self.hidden_channels = hidden_channels
        self.dropout = nn.Dropout(p=0.5) 
        
        self.fc_in = nn.Linear(input_channels, hidden_channels)
        self.fc_rec = nn.Linear(hidden_channels, hidden_channels)
        self.lif1 = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())
        
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.lif2 = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())
        
        self.part2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),           
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            nn.Dropout(p=0.5),                    
            
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, x):
        x = x.squeeze(-1).squeeze(-1)
        # x = x.permute(1, 0, 2)
        N, T_steps, _ = x.shape
        
        feat_list = []
        spikes_last = torch.zeros(N, self.hidden_channels, device=x.device)
        
        for t in range(T_steps):
            frame = x[:, t, :]
            
            spikes_last_dropped = self.dropout(spikes_last)
            
            inp = self.fc_in(frame) + self.fc_rec(spikes_last_dropped)
            spikes_t = self.lif1(inp)
            spikes_last = spikes_t
            
            x2 = self.fc2(spikes_t)
            feat = self.lif2(x2)
            feat_list.append(feat)
            
        features = torch.stack(feat_list, dim=0)
        
        output_list = []
        for t in range(T_steps):
            out = self.part2(features[t])
            output_list.append(out)
            
        outputs = torch.stack(output_list, dim=0)
        return outputs.mean(0)


def create_model(model_type, input_size, num_classes, time_steps=100):
    # 默认 hidden_channels 从 128 增加到 256
    default_hidden = 256 
    
    if model_type == 'snn':
        return SNN(input_channels=input_size, hidden_channels=default_hidden, num_classes=num_classes, T=time_steps)

