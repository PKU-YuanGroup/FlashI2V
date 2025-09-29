from .datasets_mmmu import MMMUEvalDataset
from .datasets_vqa import VQAEvalDataset
from .datasets_ai2d import AI2DEvalDataset
from .datasets_vbench import BaseGenEvalDataset, VbenchGenEvalDataset, VbenchI2VGenEvalDataset

eval_dataset_dict = {"mmmu_dev_val": MMMUEvalDataset, "ai2d_test": AI2DEvalDataset, "chartqa_test": VQAEvalDataset,
                     "docvqa_val": VQAEvalDataset, "gen_eval": BaseGenEvalDataset, "vbench_eval": VbenchGenEvalDataset,
                     "vbench_i2v": VbenchI2VGenEvalDataset}
