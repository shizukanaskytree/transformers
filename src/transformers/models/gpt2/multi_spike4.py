import torch
import torch.nn as nn

class MultiSpike4(nn.Module):

    class quant4(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            tmp = torch.clamp(input, min=0, max=8)

            """
            If i use `spikes = torch.round(tmp)`, attn layer will produce all 0s
            """
            ### spikes = torch.round(tmp)

            spikes = torch.where(tmp >= 0, torch.ceil(tmp), torch.floor(tmp))
            return spikes
            # return torch.round(torch.clamp(input, min=0, max=4))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            # print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 8] = 0
            return grad_input

    def forward(self, x):
        return self.quant4.apply(x)
