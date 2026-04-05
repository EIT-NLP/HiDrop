import traceback

try:
    from .model import LlavaLlamaForCausalLM
except Exception as e:
    print(">>>>>>>>>>>>>>>>>>> 定位 llava.__init__.py >>>>>>>>>>>>>>>>>>>")
    print(f"{type(e).__name__}: {e}")
    traceback.print_exc()  # 打印完整堆栈
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
