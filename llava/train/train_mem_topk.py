from llava.train.topk_train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")  # 支持
    # train(attn_implementation="sdpa")  # 不支持获取attention map
    # train(attn_implementation="eager")    # TopK仓库里面是支持的
