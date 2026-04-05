import traceback

try:
    from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
    from .language_model.llava_mpt import LlavaMptForCausalLM, LlavaMptConfig
    from .language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig

    # Topk
    from .language_model.llava_llama_pdrop import LlavaLlamaPDropForCausalLM, LlavaPDropConfig
    from .language_model.llava_llama_topk import LlavaLlamaTopkForCausalLM, LlavaTopkConfig

    # Exit
    from .language_model.llava_llama_exit import LlavaLlamaExitForCausalLM, LlavaExitConfig # 不连续编码
    # from .language_model.llava_llama_exit_sep import LlavaLlamaExitForCausalLM, LlavaExitConfig   # 两套编码

    # Exit + Topk
    from .language_model.llava_llama_exit_topk import LlavaLlamaExitTopkForCausalLM, LlavaExitTopkConfig                # 不连续编码
    from .language_model.llava_llama_exit_topk_cdp import LlavaLlamaExitTopkCdpForCausalLM, LlavaExitTopkCdpConfig      # 连续编码,不重新编码
    # from .language_model.llava_llama_exit_topk_crp import LlavaLlamaExitTopkCrpForCausalLM, LlavaExitTopkCrpConfig      # 连续编码，重新编码
    from .language_model.llava_llama_exit_sep_topk import LlavaLlamaExitSepTopkForCausalLM, LlavaExitSepTopkConfig      # 两套编码
except Exception as e:
    print(">>>>>>>>>>>>>>>>>>> 定位 llava.model.__init__.py >>>>>>>>>>>>>>>>>>>")
    print(f"{type(e).__name__}: {e}")
    traceback.print_exc()  # 打印完整堆栈
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
