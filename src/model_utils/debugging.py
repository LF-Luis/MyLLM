def get_stats(weights):
    # min_val = weights.min().item()
    max_val = weights.max().item()
    mean_val = weights.mean().item()
    std_val = weights.std().item()
    var_val = weights.var().item()
    print("Weight Statistics:")
    # print(f"Min: {min_val}")
    print(f"Max: {max_val}")
    print(f"Mean: {mean_val}")
    print(f"Standard Deviation: {std_val}")
    print(f"Variance: {var_val}")


def get_model_size(model):
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    double_counted = sum(p.numel() for p in model.out_proj.parameters())  # Due to weight-tying
    return trainable_param_count - double_counted
