import torchacc as ta

from flashmodels.accelerators.accelerator import (Accelerator,
                                                  AcceleratorFactory)
from flashmodels.logger import logger


class ACCQwenAccelerator(Accelerator):

    def accelerate(self, model, loader):
        if self.args.lora:
            from peft import LoraConfig, TaskType, get_peft_model

            target_modules = ["c_attn"]
            if self.args.lora_target_modules == "ALL":
                target_modules.extend(["c_proj", "w1", "w2"])

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
            )
            model = get_peft_model(model, peft_config)
            if self.args.local_rank == 0:
                logger.info("Model after lora: \n %s " % model)

        model, loader = self.accelerate_internal(model, loader)

        return model, loader

    def accelerate_internal(self, model, loader):
        if not (self.args.tp_num > 1 or self.args.sp_num > 1):
            if self.args.resume_from_checkpoint:
                raise NotImplementedError("resume_from_checkpoint.")

            config = self.get_config(model)
            model = ta.accelerate(model, config)
            return model, loader

    def get_config(self, model):
        config = ta.Config()
        config.compute.fp16 = self.args.fp16
        config.compute.bf16 = self.args.bf16

        config.memory.gc = self.args.gc
        if self.args.gc:
            config.memory.gc_cls = {"QWenBlock"}

        config.dist.fsdp.size = self.args.fsdp_num
        config.dist.fsdp.wrap_layer_cls = {"QWenBlock"}
        config.dist.fsdp.flatten_parameters = not self.args.lora

        return config


AcceleratorFactory.regist("acc-qwen", ACCQwenAccelerator)
