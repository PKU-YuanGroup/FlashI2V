from mindspeed_mm.tasks.evaluation.eval_impl.impl_base import BaseEvalImpl


class MMMUEvalImpl(BaseEvalImpl):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    def __call__(self):
        super().__call__()
        self.gather_result()
        self.analyse_result()
