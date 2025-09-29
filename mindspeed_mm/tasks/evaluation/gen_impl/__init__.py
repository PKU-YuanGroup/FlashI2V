from mindspeed_mm.tasks.evaluation.gen_impl.vbench_gen import VbenchGenEvalImpl

from mindspeed_mm.tasks.inference.pipeline import sora_pipeline_dict

eval_impl_dict = {"vbench_eval": VbenchGenEvalImpl}

eval_pipeline_dict = {"cogvideox-1.5": sora_pipeline_dict["CogVideoXPipeline"],
                      "OpenSoraPlan-1.3": sora_pipeline_dict["OpenSoraPlanPipeline"]}
