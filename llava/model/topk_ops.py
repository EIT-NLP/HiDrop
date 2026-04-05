import torch
from llava.constants import IGNORE_INDEX

PAD_INDEX = IGNORE_INDEX
PAD_VALUE = -1e9

class DifferentiableTopK:
    """ Differentiable top-k operator: Elements with large importance are kept."""
    def __init__(self, lambda_: float=1.0, normalize: bool=True, mask_threshold: float=0.5):
        """ Ref: https://github.com/LKJacky/Differentiable-Model-Scaling/blob/main/dms/dtopk_src.py

            lambda_: 超参数，控制平滑掩码的极性，选择的严格程度, 默认值为 1。
            normalize: 是否对重要性分数进行归一化, 默认是 True
            mask_threshold: 掩码阈值，用于选择高于阈值的元素
        """
        self.lambda_ = lambda_
        self.normalize = normalize
        self.mask_threshold = mask_threshold

    def _dtopk(self, c: torch.Tensor, a: torch.Tensor):
        """
        Args:
            c: 每个元素的重要性分数
            a：每个元素的剪枝比率活选择比例
        Return:
        """
        # print(c.device) # cuda:0
        # print(a.device) # cpu
        # print(self.lambda_.device)  # float no .device
        y = (c - a) * c.numel() * self.lambda_  # 计算每个元素的差异
        y = y.sigmoid()  # 通过 sigmoid 函数将 y 映射到 [0, 1] 区间，生成一个平滑的掩码
        return y  # 对每个元素的“被选中概率”或“平滑掩码值”

    def _getsoftmask(self, c: torch.Tensor, a: torch.Tensor):
        """
        Args:
            c (torch.Tensor): 每个元素的重要性分数。
            a (torch.Tensor): 每个元素的剪枝比率，剪掉的比例
        Returns:
            soft masks of elements 平滑的掩码值。
        """
        if c.numel() == 1:
            return (-(0.5 - a) * self.lambda_).sigmoid()
        else:
            if self.normalize:  # 基于排名的归一化
                c_compare = c.unsqueeze(-1) - c.unsqueeze(-2)  # [bs, N,N]
                c_ = (c_compare >= 0).float().mean(dim=-1)  # normalize to [0,1], [bs, N]
            else:
                c_ = c

            imp = self._dtopk(c_, a)  # [0,1]

            return imp

    def check_gradient(self):
        """ 检查当前DTopK算子是否是可微的 """
        c = torch.randn(5, requires_grad=True)  # 重要性分数
        a = torch.tensor([0.3], requires_grad=False)  # 剪枝比率

        # 计算 Soft Top-K 掩码
        soft_mask = self._getsoftmask(c, a)
        selected_elements = c * soft_mask

        # 假设损失是掩码的和
        loss = selected_elements.sum()

        # 反向
        loss.backward()

        if c.grad is not None:
            print("Ok! The DTopK operator is differentiable.")
        else:
            raise Exception("Wrong! The DTopK operator is not differentiable.")

    def apply_ratio(self, c: torch.Tensor, a: torch.Tensor):
        device = c.device
        mask = self._getsoftmask(c, a)
        # selected_elements = c * mask
        # keep_indices = (mask > self.mask_threshold).nonzero().squeeze()
        # if keep_indices.dim() == 0:
        #     # print(f"dim=0, {keep_indices}")
        #     keep_indices = keep_indices.unsqueeze(0)
        # #     print(f"dim=0, {keep_indices}")
        # # print(keep_indices.T)
        # return keep_indices, c[keep_indices]

        if c.dim() == 1:
            # 处理 1D: mask.shape == [N]
            # nonzero(as_tuple=False) 返回形状 [k, 1]; squeeze 之后若 k==1，会变成标量，需要判断；若 k>1，则是 [k]
            keep_indices = (mask > self.mask_threshold).nonzero(as_tuple=False).squeeze()

            if keep_indices.dim() == 0:  # 只剩一个标量
                keep_indices = keep_indices.unsqueeze(0)  # 变成 [1]

            return keep_indices, c[keep_indices]

        elif c.dim() == 2:
            bs = c.shape[0]

            # # 处理 2D: mask.shape == [bs, N]
            keep_indices = (mask > self.mask_threshold).nonzero(as_tuple=False)
            # # 这里通常是形状 [k, 2]，第一列是 batch索引，第二列是特征索引
            # # 你可以根据需要保留扁平输出，或者保留 (batch, idx) 信息。
            # keep_elements = c[keep_indices[:, 0], keep_indices[:, 1]].reshape(bs, -1)
            # keep_indices = keep_indices[:,1].reshape(bs, -1)
            # return keep_indices, keep_elements

            # 先把每个batch的 indices/value 分组
            group_indices = [[] for _ in range(bs)]
            group_values = [[] for _ in range(bs)]

            for i in range(keep_indices.shape[0]):
                b_idx = keep_indices[i, 0].item()
                f_idx = keep_indices[i, 1].item()
                group_indices[b_idx].append(f_idx)
                group_values[b_idx].append(c[b_idx, f_idx].item())

            # 找到某个batch里最多选了多少个
            max_len = max(len(lst) for lst in group_indices)

            # 构造对齐后的 Tensor
            keep_indices = torch.full((bs, max_len), PAD_INDEX, dtype=torch.long)    # [bs, max_len]
            keep_values = torch.full((bs, max_len), PAD_VALUE, dtype=c.dtype)    # [bs, max_len]

            for b in range(bs):
                n = len(group_indices[b])
                if n > 0:
                    keep_indices[b, :n] = torch.tensor(group_indices[b], dtype=torch.long)
                    keep_values[b, :n] = torch.tensor(group_values[b], dtype=c.dtype)

            keep_indices = keep_indices.to(device)
            keep_values = keep_values.to(device)
            return keep_indices, keep_values
        else:
            raise ValueError(f"apply_ratio only supports 1D or 2D, but got shape: {c.shape}")

    def apply_k(self, c: torch.Tensor, k: int):
        """
            k (int): the k in "top-k"
        """
        device = c.device
        N = c.shape[-1] # 最后一维, 1D和2D Tensor都支持
        a = torch.Tensor([1 - (k / N)]).to(device)
        return self.apply_ratio(c, a)

    def test_dtopk_1D_2D(self, num_rounds=100000, dims=None):

        consistent_count = 0  # 记录行为一致的次数
        inconsistent_cases = []  # 保存行为不一致的案例

        for round_idx in range(num_rounds):
            if dims is None:    # 随机决定是 1D 还是 2D
                dims = torch.randint(1, 3, (1,)).item()  # 1 或 2

            if dims == 1:   # 1D
                N = torch.randint(2, 2000, (1,)).item()
                x = torch.randn(N)  # 形状 [N]
            else:   # 2D
                bs = torch.randint(1, 10, (1,)).item()  # 批次大小，随意小范围
                N = torch.randint(2, 200, (1,)).item()  # 特征大小
                # bs, N = 2, 5
                x = torch.randn(bs, N)  # 形状 [bs, N]
            ratio = torch.rand(1) # ratio: 在[0,1)内的随机剪枝比例(实际上是float)

            retained_indices, retained_elements = self.apply_ratio(x, ratio)

            k = retained_indices.shape[-1]
            keep_idx, keep_elements = self.apply_k(x, k)

            # 验证行为一致性:
            same_elements = torch.allclose(retained_elements, keep_elements, atol=1e-6)

            if same_elements:
                consistent_count += 1
            else:
                inconsistent_cases.append({
                    "round": round_idx,
                    "dims": dims,
                    "ratio": ratio,
                    "x": x,
                    "retained_indices_ratio": retained_indices,
                    "retained_elements_ratio": retained_elements,
                    "retained_indices_k": keep_idx,
                    "retained_elements_k": keep_elements
                })

        # 输出结果
        print(f"总测试次数: {num_rounds}")
        print(f"行为一致次数: {consistent_count}")
        print(f"行为不一致次数: {num_rounds - consistent_count}")

        if inconsistent_cases:
            print("\n行为不一致的案例(仅显示前5个):")
            for case in inconsistent_cases[:5]:
                print(f"Round {case['round']}, dims: {case['dims']}, ratio: {case['ratio']}")
                print(f"x shape: {case['x'].shape}")
                print(f"Retained indices (ratio): {case['retained_indices_ratio']}")
                print(f"Retained elements (ratio): {case['retained_elements_ratio']}")
                print(f"Retained indices (k): {case['retained_indices_k']}")
                print(f"Retained elements (k): {case['retained_elements_k']}")
                print("-----")


# if __name__ == '__main__':
#     dtopk = DifferentiableTopK()
#     dtopk.test_dtopk_1D_2D(num_rounds=1, dims=2)

def test_demo():
    dtopk = DifferentiableTopK(mask_threshold=0.5, normalize=False)

    # 构造一个 2D 张量 (bs=2, N=5)，
    # 让 batch=0 故意很小，batch=1 故意很大，
    # 以便在同一个阈值下，选出数量不同，触发 -1 padding
    c = torch.tensor([
        [0.1, 0.2, 0.3, 0.35, 0.4 ],  # batch=0
        [2.0, 2.1, 2.2, 3.5,  4.0 ]   # batch=1
    ], dtype=torch.float32)

    print("==== Input c ====")
    print(c)

    # 我们用 ratio=0.5 -> a=0.5
    # 这样 batch=0 大概只会选 0.35、0.4(因为它们>0.5要看放缩后sigmoid, 但数值不高, 也可能很少)
    # batch=1 则会选到大多数(因为值很高)
    # 期望 batch=1 选出的数量 > batch=0 => 出现 -1 填充
    a = torch.tensor([0.5])
    keep_idx, keep_vals = dtopk.apply_ratio(c, a)

    print("\n==== apply_ratio with ratio=0.5, threshold=0.5 ====")
    print("keep_indices (padded):\n", keep_idx)
    print("keep_values (padded):\n", keep_vals)

    # 再试一个 apply_k
    # 例如 k=2 => a= 1-(2/5)= 0.6
    # 这样 batch=0 大概率选 0或1个, batch=1 保留很多
    # 也能看到 -1
    print("\n==== apply_k(k=2) ====")
    out_idx_k, out_vals_k = dtopk.apply_k(c, k=2)
    print("Indices:\n", out_idx_k)
    print("Values:\n", out_vals_k)

if __name__ == "__main__":
    test_demo()
