from torch.autograd import Function
import torch


class Proxy_table(Function):
    def __init__(self, table, alpha=0.01):
        super(Proxy_table, self).__init__()
        self.table = table
        self.alpha = alpha

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.table.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.table)
        for x, y in zip(inputs, targets):
            if y != -1.:
                self.table[y] = self.alpha * self.table[y] \
                                + (1. - self.alpha) * x
                self.table[y] /= self.table[y].norm()
        return grad_inputs, None
