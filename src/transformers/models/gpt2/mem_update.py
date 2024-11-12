import torch
import torch.nn as nn

from .multi_spike4 import MultiSpike4

decay = 0.25 # decay constants

class mem_update(nn.Module):
    def __init__(self, act=False):
        super(mem_update, self).__init__()
        # self.actFun= torch.nn.LeakyReLU(0.2, inplace=False)

        self.act = act
        self.qtrick = MultiSpike4()  # change the max value

    def forward(self, x):
        ### spike 是一个与 x[0] 形状相同的零张量，用于保存当前时刻的脉冲。
        ### EN: spike is a zero tensor with the same shape as x[0], used to store the spike at the current time step.
        spike = torch.zeros_like(x[0]).to(x.device)

        ### output 是一个与 x 形状相同的零张量，用于存储每个时间步的脉冲输出。
        ### EN: output is a zero tensor with the same shape as x, used to store the spike output at each time step.
        output = torch.zeros_like(x)

        ### mem_old 初始化为 0，用于记录前一时刻的膜电位。
        ### EN: mem_old is initialized to 0 to record the membrane potential of the previous time step.
        mem_old = 0

        ### time_window 是时间窗口的长度，从 x 的第一个维度获取。
        ### EN: time_window is the length of the time window, obtained from the first dimension of x.
        time_window = x.shape[0]

        """
        EN:
        Loop through the time window and update the membrane potential.
        For each time step i, if i >= 1, update mem to (mem_old - spike.detach()) * decay + x[i].
            (mem_old - spike.detach()) calculates the membrane potential at the previous time step minus the spike,
            so that the generated spike affects the membrane potential.
            decay is a decay constant used to simulate the decay process of the membrane potential.
        If i == 0, set mem directly to x[i].

        CN:
        循环遍历时间窗口并更新膜电位
        遍历每个时间步 i, 若 i >= 1，则:
            mem 更新为 (mem_old - spike.detach()) * decay + x[i]。
            (mem_old - spike.detach()) 计算前一时刻的膜电位减去脉冲，使得生成的脉冲会影响膜电位。
            decay 是衰减常数，用于模拟膜电位的衰减过程。
        若 i == 0，即第一个时间步，将 mem 直接设为 x[i]。
        """
        for i in range(time_window):
            if i >= 1:
                mem = (mem_old - spike.detach()) * decay + x[i]

            else:
                mem = x[i]

            """
		    调用 self.qtrick(mem) 根据当前膜电位生成脉冲，并赋值给 spike。
		    将 mem 的值赋给 mem_old，以便下一个时间步使用。
		    将生成的 spike 存储到 output[i] 中，构建完整的输出张量。
            """
            spike = self.qtrick(mem)
            mem_old = mem.clone()
            output[i] = spike

        # print(output[0][0][0][0])
        return output
