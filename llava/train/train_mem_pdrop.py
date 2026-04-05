from llava.train.pdrop_train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")  # 也就是说pdrop只支持flash-attn
    # train(attn_implementation="sdpa")   # 这个不行
    # train(attn_implementation="eager")    # 这个不行