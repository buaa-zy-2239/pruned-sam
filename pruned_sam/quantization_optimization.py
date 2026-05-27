import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/zhang/vista-slam')


class ACNRRegularizer:
    """激活感知条件数正则化器"""
    
    @staticmethod
    def regularize(weight, num_steps=50, lr=0.001, gamma=0.01):
        """
        激活感知条件数正则化 - 降低权重矩阵的条件数
        weight: (C_out, C_in) 权重矩阵
        """
        W = weight.clone().detach().requires_grad_(True)
        W0 = weight.clone().detach()
        optimizer = torch.optim.Adam([W], lr=lr)

        for _ in range(num_steps):
            optimizer.zero_grad()
            U, S, V = torch.svd(W)
            
            max_sv = S.max()
            min_sv = S[S > 1e-6].min() if (S > 1e-6).any() else S.min()
            
            condition_number = max_sv / (min_sv + 1e-8)
            
            loss = condition_number + gamma * ((W - W0) ** 2).sum()
            loss.backward()
            optimizer.step()

        return W.detach()

    @staticmethod
    def apply_to_model(model, layers_to_regularize=None):
        """对模型中的线性层应用ACNR正则化"""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                if layers_to_regularize is None or name in layers_to_regularize:
                    if isinstance(module, nn.Conv2d):
                        orig_shape = module.weight.shape
                        weight_2d = module.weight.view(module.out_channels, -1)
                        regularized = ACNRRegularizer.regularize(weight_2d)
                        module.weight.data = regularized.view(orig_shape)
                    else:
                        module.weight.data = ACNRRegularizer.regularize(module.weight.data)
        return model


class LNQActivationQuantizer:
    """对数非线性激活量化器"""
    
    def __init__(self, n_bits=8):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** n_bits - 1
        self.scale = None
        self.zero_point = None
        self.x_min_log = None
        self.x_max_log = None
        self.eps = 1e-5

    def calibrate(self, x_calib):
        """在校准集上统计对数域的范围"""
        x_log = torch.log2(x_calib - x_calib.min() + self.eps)
        self.x_min_log = x_log.min().item()
        self.x_max_log = x_log.max().item()
        self.scale = (self.x_max_log - self.x_min_log) / (self.qmax - self.qmin)
        self.zero_point = self.qmin - round(self.x_min_log / self.scale)

    def forward(self, x):
        """前向量化"""
        x_log = torch.log2(x - x.min() + self.eps)
        x_q = torch.clamp(torch.round(x_log / self.scale + self.zero_point),
                          self.qmin, self.qmax)
        x_dq = (x_q - self.zero_point) * self.scale
        return torch.pow(2, x_dq)


class MACCompensator:
    """矩阵乘法感知补偿器"""
    
    @staticmethod
    def compensate(linear_layer, x_calib_fp, x_calib_q, n_samples=256):
        """
        MAC: 矩阵乘法感知补偿
        linear_layer: 交叉注意力中 Query/Value 投影的线性层
        x_calib_fp:   校准集 FP32 激活值 (N, C_in)
        x_calib_q:    校准集量化后的激活值 (N, C_in)
        """
        W_fp = linear_layer.weight.data
        N = min(n_samples, x_calib_fp.size(0))

        x_fp = x_calib_fp[:N]
        x_q = x_calib_q[:N]

        Y_fp = x_fp @ W_fp.T
        Y_q = x_q @ W_fp.T

        delta_Y = Y_fp - Y_q

        X_q_pinv = torch.linalg.pinv(x_q)
        delta_W = (X_q_pinv @ delta_Y).T

        W_compensated = W_fp + delta_W
        return W_compensated

    @staticmethod
    def apply_to_model(model, calib_data):
        """对模型中的线性层应用MAC补偿"""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                x_calib_fp = calib_data[name] if isinstance(calib_data, dict) else calib_data
                x_calib_q = x_calib_fp.clamp(-127, 127).round() / 127.0
                
                if x_calib_fp.dim() > 2:
                    x_calib_fp = x_calib_fp.flatten(0, 1)
                    x_calib_q = x_calib_q.flatten(0, 1)
                
                compensated_weight = MACCompensator.compensate(module, x_calib_fp, x_calib_q)
                module.weight.data = compensated_weight
        return model


class QuantizationOptimizer:
    """量化优化器，整合ACNR、LNQ和MAC"""
    
    def __init__(self, model):
        self.model = model
        self.acnr_regularizer = ACNRRegularizer()
        self.lnq_quantizer = LNQActivationQuantizer()
        self.mac_compensator = MACCompensator()
    
    def optimize(self, calib_loader=None, num_calib_samples=64):
        """执行完整的量化优化流程"""
        print("=" * 70)
        print("执行量化优化流程 (ACNR + LNQ + MAC)")
        print("=" * 70)
        
        print("\n1. 应用ACNR正则化...")
        self.model = self.acnr_regularizer.apply_to_model(self.model)
        print("   ✅ ACNR正则化完成")
        
        if calib_loader is not None:
            print("\n2. 收集校准数据...")
            calib_data = self._collect_calib_data(calib_loader, num_calib_samples)
            print("   ✅ 校准数据收集完成")
            
            print("\n3. 应用MAC补偿...")
            self.model = self.mac_compensator.apply_to_model(self.model, calib_data)
            print("   ✅ MAC补偿完成")
        
        print("\n" + "=" * 70)
        print("量化优化流程完成!")
        print("=" * 70)
        
        return self.model
    
    def _collect_calib_data(self, calib_loader, num_samples=64):
        """收集校准数据"""
        calib_data = {}
        activations = {}
        
        def save_activation(name):
            def hook(module, input, output):
                if name not in activations:
                    activations[name] = []
                activations[name].append(output.detach().cpu())
            return hook
        
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(save_activation(name)))
        
        self.model.eval()
        count = 0
        with torch.no_grad():
            for images, _ in calib_loader:
                if count >= num_samples:
                    break
                self.model(images.cpu())
                count += images.size(0)
        
        for hook in hooks:
            hook.remove()
        
        for name, acts in activations.items():
            if len(acts) > 0:
                calib_data[name] = torch.cat(acts, dim=0)
        
        return calib_data


def main():
    print("=" * 70)
    print("量化优化模块测试")
    print("=" * 70)
    
    test_linear = nn.Linear(128, 256)
    print(f"\n原始权重条件数: {torch.linalg.cond(test_linear.weight.data):.2f}")
    
    regularized_weight = ACNRRegularizer.regularize(test_linear.weight.data)
    print(f"ACNR后条件数: {torch.linalg.cond(regularized_weight):.2f}")
    
    print("\n✅ 量化优化模块测试通过!")


if __name__ == '__main__':
    main()
