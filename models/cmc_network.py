import torch
import torch.nn as nn

class CMCLinearClassifier(nn.Module):
    def __init__(self, model, encoder_dim):
        super().__init__()
        self.transformer = model
        self.fc1 = nn.Linear(in_features=encoder_dim*2, out_features=10)

    def forward(self, x):
        output_l, output_ab = self.transformer(x)
        output = torch.cat((output_l.detach(),output_ab.detach()),dim=1)
        output = self.fc1(output)
        return output

