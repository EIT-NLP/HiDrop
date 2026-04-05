import torch

def debug_tensor_shapes(stage_name, **tensors):
    """调试张量形状和有效性"""
    print(f"\n=== {stage_name} ===")
    for name, tensor in tensors.items():
        if tensor is None:
            print(f"{name}: None")
            continue
        
        print(f"{name}: {tensor.shape} | dtype: {tensor.dtype} | device: {tensor.device}")
        
        # 检查是否包含无效值
        if torch.is_floating_point(tensor):
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            if nan_count > 0 or inf_count > 0:
                print(f"  WARNING: {name} has {nan_count} NaN and {inf_count} Inf values")
        
        # 特殊检查attention_mask
        if name == "attention_mask" and tensor.numel() > 0:
            unique_vals = torch.unique(tensor)
            print(f"  attention_mask unique values: {unique_vals.tolist()}")
            # 检查是否有无效的注意力掩码值
            if not all(v in [0, 1, True, False] for v in unique_vals.tolist()):
                print(f"  ERROR: attention_mask contains invalid values!")
        
        # 检查position_ids的合理性
        if name == "position_ids" and tensor.numel() > 0:
            min_pos = tensor.min().item()
            max_pos = tensor.max().item()
            print(f"  position_ids range: [{min_pos}, {max_pos}]")
            if min_pos < 0:
                print(f"  ERROR: position_ids has negative values!")

def validate_tensor_consistency(features, attention_mask, position_ids, labels, stage_name):
    """验证张量之间的一致性"""
    print(f"\n>>> Validating {stage_name} <<<")
    
    if features is None:
        print("ERROR: features is None")
        return False
        
    batch_size, seq_len = features.shape[:2]
    print(f"Base dimensions: batch_size={batch_size}, seq_len={seq_len}")
    
    issues = []
    
    # 检查attention_mask
    if attention_mask is not None:
        if attention_mask.shape[:2] != (batch_size, seq_len):
            issues.append(f"attention_mask shape mismatch: {attention_mask.shape} vs expected {(batch_size, seq_len)}")
    
    # 检查position_ids
    if position_ids is not None:
        if position_ids.shape[:2] != (batch_size, seq_len):
            issues.append(f"position_ids shape mismatch: {position_ids.shape} vs expected {(batch_size, seq_len)}")
    
    # 检查labels
    if labels is not None:
        if labels.shape[:2] != (batch_size, seq_len):
            issues.append(f"labels shape mismatch: {labels.shape} vs expected {(batch_size, seq_len)}")
    
    if issues:
        print("CONSISTENCY ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("All tensors are consistent ✓")
        return True

# 在关键位置添加检查
def check_image_token_indices(image_token_posi, features_shape, stage_name):
    """检查图像token位置的有效性"""
    print(f"\n>>> Checking image token indices at {stage_name} <<<")
    
    batch_size, seq_len = features_shape[:2]
    
    for i, img_pos in enumerate(image_token_posi):
        if img_pos == -1:
            print(f"  Sample {i}: No image token")
            continue
            
        if img_pos < 0 or img_pos >= seq_len:
            print(f"  ERROR: Sample {i} image_pos={img_pos} is out of bounds [0, {seq_len})")
            return False
            
        img_end = img_pos + 576
        if img_end > seq_len:
            print(f"  ERROR: Sample {i} image extends beyond sequence: {img_pos}+576={img_end} > {seq_len}")
            return False
            
        print(f"  Sample {i}: image tokens at [{img_pos}, {img_end})")
    
    return True