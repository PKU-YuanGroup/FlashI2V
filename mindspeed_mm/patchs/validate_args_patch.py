# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
import os
import torch

from mindspeed.patch_utils import MindSpeedPatchesManager as pm
from mindspeed.arguments import validate_args_wrapper
from megatron.training.arguments import load_retro_args, _check_arg_is_not_none, _print_args
from megatron.training import print_rank_0

from mindspeed_mm.configs.config import merge_mm_args
from mindspeed_mm.utils.utils import ensure_valid


def safe_getattr(mm_object, mm_name, mm_default_value):
    # 如果 mm_object.mm_name 不等于 mm_default_value， 则打印日志， 提示用户真实使用的值已被覆盖
    mm_value = getattr(mm_object, mm_name, mm_default_value)
    if mm_value != mm_default_value:
        print_rank_0(f'[INFO] the original value of {mm_name} is {mm_default_value}, now changed as {mm_value} which comes from model.json')
    return mm_value


def validate_args(args, defaults=None):
    if defaults is None:
        defaults = {}

    #merge mm config to args
    merge_mm_args(args)

    #use model.json to fill args
    if hasattr(args.mm.model, 'text_decoder'):
        args.num_layers = safe_getattr(args.mm.model.text_decoder, 'num_layers', args.num_layers)
        args.hidden_size = safe_getattr(args.mm.model.text_decoder, 'hidden_size', args.hidden_size)
        args.num_attention_heads = safe_getattr(args.mm.model.text_decoder, 'num_attention_heads', args.num_attention_heads)
        args.max_position_embeddings = safe_getattr(args.mm.model.text_decoder, 'max_position_embeddings', args.max_position_embeddings)
        args.ffn_hidden_size = safe_getattr(args.mm.model.text_decoder, 'ffn_hidden_size', args.ffn_hidden_size)

        # MOE
        if hasattr(args.mm.model.text_decoder, 'num_moe_experts'):
            args.num_experts = safe_getattr(args.mm.model.text_decoder, 'num_moe_experts', args.num_experts)
            args.n_shared_experts = safe_getattr(args.mm.model.text_decoder, 'n_shared_experts', args.n_shared_experts)
            args.mm.model.text_decoder.moe_token_dispatcher_type = safe_getattr(args.mm.model.text_decoder, 'moe_token_dispatcher_type', args.moe_token_dispatcher_type)
            args.mm.model.text_decoder.tensor_model_parallel_size = safe_getattr(args.mm.model.text_decoder, 'tensor_model_parallel_size', args.tensor_model_parallel_size)
            args.mm.model.text_decoder.sequence_parallel = safe_getattr(args.mm.model.text_decoder, 'sequence_parallel', args.sequence_parallel)
            args.mm.model.text_decoder.expert_model_parallel_size = safe_getattr(args.mm.model.text_decoder, 'expert_model_parallel_size', args.expert_model_parallel_size)

    # use model.json to fill predictor arg
    if hasattr(args.mm.model, 'predictor'):
        if hasattr(args.mm.model.predictor, 'mm_single_blocks_depth') and hasattr(args.mm.model.predictor, 'mm_double_blocks_depth'):
            mm_double_blocks_depth = getattr(args.mm.model.predictor, 'mm_double_blocks_depth', 0)
            if mm_double_blocks_depth <= 0:
                raise AssertionError(f"MindSpeed-MM Error: mm_double_blocks_depth must > 0, actually:{mm_double_blocks_depth}")
            args.num_layers = safe_getattr(args.mm.model.predictor, 'mm_single_blocks_depth', args.num_layers)
        args.num_layers = safe_getattr(args.mm.model.predictor, 'num_layers', args.num_layers)
        args.num_attention_heads = safe_getattr(args.mm.model.predictor, 'num_heads', args.num_attention_heads)
        head_dim = getattr(args.mm.model.predictor, 'head_dim', 1)
        if isinstance(head_dim, int):
            hidden_size = args.num_attention_heads * head_dim
            print_rank_0(f"[INFO] the original value of normalization is {args.hidden_size}, now changed as {hidden_size} which comes from model.json")
            args.hidden_size = hidden_size
        args.hidden_size = safe_getattr(args.mm.model.predictor, 'hidden_size', args.hidden_size)
        args.seq_length = safe_getattr(args.mm.model.predictor, 'max_seq_len', 24)
        args.attention_dropout = safe_getattr(args.mm.model.predictor, 'dropout', args.attention_dropout)
        args.hidden_dropout = safe_getattr(args.mm.model.predictor, 'hidden_dropout', args.hidden_dropout)
        args.swiglu = safe_getattr(args.mm.model.predictor, 'swiglu', args.swiglu)
        args.masked_softmax_fusion = safe_getattr(args.mm.model.predictor, 'masked_softmax_fusion', args.masked_softmax_fusion)
        norm_type = getattr(args.mm.model.predictor, 'qk_norm_type', "")
        if isinstance(norm_type, str) and norm_type == "rmsnorm":
            print_rank_0(f"[INFO] the original value of normalization is {args.normalization}, now changed as RMSNorm which comes from model.json")
            args.normalization = "RMSNorm"
            args.use_fused_rmsnorm = safe_getattr(args.mm.model.predictor, 'use_fused_rmsnorm', args.use_fused_rmsnorm)
        args.max_position_embeddings = safe_getattr(args.mm.model.predictor, 'max_position_embeddings', args.seq_length)
        args.position_embedding_type = safe_getattr(args.mm.model.predictor, 'position_embedding_type', 'rope')
        args.rotary_base = safe_getattr(args.mm.model.predictor, 'rotary_base', 500000)
        args.tokenizer_type = safe_getattr(args.mm.model.predictor, 'tokenizer_type', "NullTokenizer")
        args.vocab_size = safe_getattr(args.mm.model.predictor, "vocab_size", 0)

    # use default value to fill feature_extration arg
    elif hasattr(args.mm.model, 'ae'):
        args.num_layers = 1
        args.hidden_size = 3072
        args.num_attention_heads = 48
        args.seq_length = 24
        args.attention_dropout = 0.0
        args.hidden_dropout = 0.0
        args.swiglu = True
        args.masked_softmax_fusion = False
        args.max_position_embeddings = 24
        args.position_embedding_type = "rope"
        args.rotary_base = 500000
        args.tokenizer_type = 'NullTokenizer'
        args.vocab_size = 0

    # Load saved args from Retro (if applicable).
    load_retro_args(args)

    # Tensor model parallel size.
    args.tensor_model_parallel_size = min(
        args.tensor_model_parallel_size, args.world_size)
    ensure_valid(args.world_size % args.tensor_model_parallel_size == 0, 'world size'\
        ' ({}) is not divisible by tensor model parallel size ({})'.format(
            args.world_size, args.tensor_model_parallel_size))

    # Pipeline model parallel size.
    args.pipeline_model_parallel_size = min(
        args.pipeline_model_parallel_size,
        (args.world_size // args.tensor_model_parallel_size))
    args.transformer_pipeline_model_parallel_size = (
        args.pipeline_model_parallel_size - 1
        if args.standalone_embedding_stage else
        args.pipeline_model_parallel_size
    )

    # Checks.
    model_parallel_size = args.pipeline_model_parallel_size * \
                          args.tensor_model_parallel_size
    ensure_valid(args.world_size % (model_parallel_size * args.context_parallel_size) == 0, \
        'world size ({}) is not divisible by tensor parallel size ({}) times ' \
        'pipeline parallel size ({}) times context parallel size ({})'.format(
        args.world_size, args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size, args.context_parallel_size))
    args.data_parallel_size = args.world_size // (model_parallel_size * args.context_parallel_size)
    if args.rank == 0:
        print('using world size: {}, data-parallel size: {}, '
              'context-parallel size: {} '
              'tensor-model-parallel size: {}, '
              'pipeline-model-parallel size: {} '.format(
                  args.world_size, args.data_parallel_size,
                  args.context_parallel_size,
                  args.tensor_model_parallel_size,
                  args.pipeline_model_parallel_size), flush=True)
    if args.pipeline_model_parallel_size > 1:
        if args.pipeline_model_parallel_split_rank is not None:
            ensure_valid(args.pipeline_model_parallel_split_rank < \
                    args.pipeline_model_parallel_size, 'split rank needs'\
                    ' to be less than pipeline model parallel size ({})'.format(
                            args.pipeline_model_parallel_size))

    if args.tp_comm_overlap:
        ensure_valid(args.sequence_parallel, 'Tensor parallel communication/GEMM overlap can happen only when sequence parallelism is enabled')

    # Deprecated arguments
    ensure_valid(args.batch_size is None, '--batch-size argument is no longer ' \
        'valid, use --micro-batch-size instead')
    del args.batch_size
    ensure_valid(args.warmup is None, '--warmup argument is no longer valid, use ' \
        '--lr-warmup-fraction instead')
    del args.warmup
    ensure_valid(args.model_parallel_size is None, '--model-parallel-size is no ' \
        'longer valid, use --tensor-model-parallel-size instead')
    del args.model_parallel_size

    if args.checkpoint_activations:
        if args.rank == 0:
            print('--checkpoint-activations is no longer valid, use --recompute-activations, '
                  'or, for more control, --recompute-granularity and --recompute-method.')
        exit()
    del args.checkpoint_activations

    if args.recompute_activations:
        args.recompute_granularity = 'selective'
    del args.recompute_activations

    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key, None) is not None:
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                                               flush=True)
        else:
            setattr(args, key, defaults[key])

    if args.data_path is not None and args.split is None:
        legacy_default_split_value = '969, 30, 1'
        if args.rank == 0:
            print('WARNING: Please specify --split when using --data-path. Using legacy default value '
                  f'of "{legacy_default_split_value}"')
        args.split = legacy_default_split_value

    # Batch size.
    ensure_valid(args.micro_batch_size is not None, 'args.micro_batch_size can not be None')
    ensure_valid(args.micro_batch_size > 0, 'args.micro_batch_size must be greater than 0')
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        if args.rank == 0:
            print('setting global batch size to {}'.format(
                args.global_batch_size), flush=True)
    ensure_valid(args.global_batch_size > 0, 'args.global_batch_size must be greater than 0')
    if args.num_layers_per_virtual_pipeline_stage is not None:
        raise AssertionError('MindSpeed-MM Error: --num-layers-per-virtual-pipeline-stage is deprecated, please use --virtual-pipeline-model-parallel-size instead')
    if hasattr(args, 'virtual_pipeline_model_parallel_size') and args.virtual_pipeline_model_parallel_size is not None and args.virtual_pipeline_model_parallel_size > 1:
        if args.overlap_p2p_comm:
            ensure_valid(args.pipeline_model_parallel_size > 1, \
                'when interleaved schedule is used, pipeline-model-parallel size '\
                'should be greater than 1')
        else:
            ensure_valid(args.pipeline_model_parallel_size > 2, \
                'when interleaved schedule is used and p2p communication overlap is disabled, '\
                'pipeline-model-parallel size should be greater than 2 to avoid having multiple '\
                'p2p sends and recvs between same 2 ranks per communication batch')
        if hasattr(args.mm.model, 'text_decoder'):
            _pipeline_num_layers = getattr(args.mm.model.text_decoder, 'pipeline_num_layers', None)
            if _pipeline_num_layers is None or len(_pipeline_num_layers) != args.virtual_pipeline_model_parallel_size:
                raise AssertionError('MindSpeed-MM Error: vpp must enabled by --virtual-pipeline-model-parallel-size in shell and pipeline_num_layers in model.json, \
                    and virtual-pipeline-model-parallel-size must equal the length of pipeline_num_layers')
        elif hasattr(args.mm.model, 'predictor'):
            _pipeline_num_layers = getattr(args.mm.model.predictor, 'pipeline_num_layers', None)
            if _pipeline_num_layers is None or len(_pipeline_num_layers) != args.virtual_pipeline_model_parallel_size:
                raise AssertionError('MindSpeed-MM Error: vpp must enabled by --virtual-pipeline-model-parallel-size in shell and pipeline_num_layers in model.json, \
                    and virtual-pipeline-model-parallel-size must equal the length of pipeline_num_layers')
        else:
            raise AssertionError('MindSpeed-MM Error: vpp must enabled by --virtual-pipeline-model-parallel-size in shell and pipeline_num_layers in model.json')
    else:
        args.virtual_pipeline_model_parallel_size = None
        # Overlap P2P communication is disabled if not using the interleaved schedule.
        args.overlap_p2p_comm = False
        if args.rank == 0:
            print('WARNING: Setting args.overlap_p2p_comm to False since non-interleaved '
                  'schedule does not support overlapping p2p communication')

    if args.overlap_param_gather:
        ensure_valid(args.use_distributed_optimizer, \
            '--overlap-param-gather only supported with distributed optimizer')
        ensure_valid(args.overlap_grad_reduce, \
            '--overlap-grad-reduce should be turned on when using --overlap-param-gather')
        ensure_valid(not args.use_legacy_models, \
            '--overlap-param-gather only supported with MCore models')

    # Parameters dtype.
    args.params_dtype = torch.float
    if args.fp16:
        ensure_valid(not args.bf16, 'args.bf16 must be false when args.fp16 is true')
        args.params_dtype = torch.half
        # Turn off checking for NaNs in loss and grads if using dynamic loss scaling,
        # where NaNs in grads / loss are signal to the loss scaler.
        if not args.loss_scale:
            args.check_for_nan_in_loss_and_grad = False
            if args.rank == 0:
                print('WARNING: Setting args.check_for_nan_in_loss_and_grad to False since '
                      'dynamic loss scaling is being used')
    if args.bf16:
        ensure_valid(not args.fp16, 'args.fp16 must be false when args.bf16 is true')
        args.params_dtype = torch.bfloat16
        # bfloat16 requires gradient accumulation and all-reduce to
        # be done in fp32.
        if not args.accumulate_allreduce_grads_in_fp32:
            args.accumulate_allreduce_grads_in_fp32 = True
            if args.rank == 0:
                print('accumulate and all-reduce gradients in fp32 for '
                      'bfloat16 data type.', flush=True)

    if args.rank == 0:
        print('using {} for parameters ...'.format(args.params_dtype),
              flush=True)

    if args.dataloader_type is None:
        args.dataloader_type = 'single'

    # data
    ensure_valid(args.num_dataset_builder_threads > 0, 'args.num_dataset_builder_threads should > 0')

    # Consumed tokens.
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0

    # Support for variable sequence lengths across batches/microbatches.
    # set it if the dataloader supports generation of variable sequence lengths
    # across batches/microbatches. Due to additional communication overhead
    # during pipeline parallelism, it should not be set if sequence length
    # is constant during training.
    args.variable_seq_lengths = False

    # Iteration-based training.
    if args.train_iters:
        # If we use iteration-based training, make sure the
        # sample-based options are off.
        ensure_valid(args.train_samples is None, \
            'expected iteration-based training')
        ensure_valid(args.lr_decay_samples is None, \
            'expected iteration-based learning rate decay')
        ensure_valid(args.lr_warmup_samples == 0, \
            'expected iteration-based learning rate warmup')
        ensure_valid(args.rampup_batch_size is None, \
            'expected no batch-size rampup for iteration-based training')
        if args.lr_warmup_fraction is not None:
            ensure_valid(args.lr_warmup_iters == 0, \
                'can only specify one of lr-warmup-fraction and lr-warmup-iters')

    # Sample-based training.
    if args.train_samples:
        # If we use sample-based training, make sure the
        # iteration-based options are off.
        ensure_valid(args.train_iters is None, \
            'expected sample-based training')
        ensure_valid(args.lr_decay_iters is None, \
            'expected sample-based learning rate decay')
        ensure_valid(args.lr_warmup_iters == 0, \
            'expected sample-based learnig rate warmup')
        if args.lr_warmup_fraction is not None:
            ensure_valid(args.lr_warmup_samples == 0, \
                'can only specify one of lr-warmup-fraction ' \
                'and lr-warmup-samples')

    if args.num_layers is not None:
        ensure_valid(args.encoder_num_layers is None, \
            'cannot have both num-layers and encoder-num-layers specified')
        args.encoder_num_layers = args.num_layers
    else:
        ensure_valid(args.encoder_num_layers is not None, \
            'either num-layers or encoder-num-layers should be specified')
        args.num_layers = args.encoder_num_layers

    # Check required arguments.
    required_args = ['num_layers', 'hidden_size', 'num_attention_heads',
                     'max_position_embeddings']
    for req_arg in required_args:
        _check_arg_is_not_none(args, req_arg)

    # Checks.
    if args.ffn_hidden_size is None:
        if args.swiglu:
            # reduce the dimnesion for MLP since projections happens on
            # two linear layers. this keeps the number of paramters in
            # the same ballpark as the counterpart with 4*h size
            # we keep it a multiple of 64, which means the actual tensor size
            # will be a multiple of 64 / tp_size
            args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64
        else:
            args.ffn_hidden_size = 4 * args.hidden_size

    if args.kv_channels is None:
        ensure_valid(args.hidden_size % args.num_attention_heads == 0, 'args.hidden_size % args.num_attention_heads != 0 is not allowed')
        args.kv_channels = args.hidden_size // args.num_attention_heads

    if args.seq_length is not None:
        ensure_valid(args.encoder_seq_length is None, 'args.encoder_seq_length must be None if args.seq_length is not None')
        args.encoder_seq_length = args.seq_length
    else:
        ensure_valid(args.encoder_seq_length is not None, 'args.encoder_seq_length must be not None when args.seq_length is None')
        args.seq_length = args.encoder_seq_length

    if args.seq_length is not None:
        ensure_valid(args.max_position_embeddings >= args.seq_length, 'args.max_position_embeddings should >= args.seq_length')
    if args.decoder_seq_length is not None:
        ensure_valid(args.max_position_embeddings >= args.decoder_seq_length, 'args.max_position_embeddings should >= args.decoder_seq_length')
    if args.lr is not None:
        ensure_valid(args.min_lr <= args.lr, 'args.min_lr should <= args.lr')
    if args.save is not None:
        ensure_valid(args.save_interval is not None, 'args.save_interval must is not None, when args.save is not None')
    # Mixed precision checks.
    if args.fp16_lm_cross_entropy:
        ensure_valid(args.fp16, 'lm cross entropy in fp16 only support in fp16 mode.')
    if args.fp32_residual_connection:
        ensure_valid(args.fp16 or args.bf16, \
            'residual connection in fp32 only supported when using fp16 or bf16.')

    if args.moe_grouped_gemm:
        ensure_valid(args.bf16, 'Currently GroupedGEMM for MoE only supports bf16 dtype.')
        dc = torch.cuda.get_device_capability()
        ensure_valid(dc[0] >= 8, "Unsupported compute capability for GroupedGEMM kernels.")

    if args.weight_decay_incr_style == 'constant':
        ensure_valid(args.start_weight_decay is None, 'args.start_weight_decay should is None when args.weight_decay_incr_style == constant')
        ensure_valid(args.end_weight_decay is None, 'args.end_weight_decay should is None when args.weight_decay_incr_style == constant')
        args.start_weight_decay = args.weight_decay
        args.end_weight_decay = args.weight_decay
    else:
        ensure_valid(args.start_weight_decay is not None, 'args.start_weight_decay should is not None, when args.weight_decay_incr_style != constant')
        ensure_valid(args.end_weight_decay is not None, 'args.end_weight_decay should is not None, when args.weight_decay_incr_style != constant')

    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])
    # Persistent fused layer norm.
    if TORCH_MAJOR < 1 or (TORCH_MAJOR == 1 and TORCH_MINOR < 11):
        args.no_persist_layer_norm = True
        if args.rank == 0:
            print('Persistent fused layer norm kernel is supported from '
                  'pytorch v1.11 (nvidia pytorch container paired with v1.11). '
                  'Defaulting to no_persist_layer_norm=True')

    # Activation recomputing.
    if args.distribute_saved_activations:
        ensure_valid(args.tensor_model_parallel_size > 1, 'can distribute ' \
            'recomputed activations only across tensor model ' \
            'parallel groups')
        ensure_valid(args.recompute_granularity == 'full', \
            'distributed recompute activations is only '\
            'application to full recompute granularity')
        ensure_valid(args.recompute_method is not None, \
            'for distributed recompute activations to work you '\
            'need to use a recompute method ')
        ensure_valid((TORCH_MAJOR, TORCH_MINOR) >= (1, 10), \
            'distributed recompute activations are supported for pytorch ' \
            'v1.10 and above (Nvidia Pytorch container >= 21.07). Current ' \
            'pytorch version is v%s.%s.' % (TORCH_MAJOR, TORCH_MINOR))

    if args.recompute_granularity == 'selective':
        ensure_valid(args.recompute_method is None, \
            'recompute method is not yet supported for ' \
            'selective recomputing granularity')

    # disable sequence parallelism when tp=1
    # to avoid change in numerics when
    # sequence_parallelism is enabled.
    if args.tensor_model_parallel_size == 1:
        args.sequence_parallel = False

    # disable async_tensor_model_parallel_allreduce when
    # model parallel memory optimization is enabled
    if args.sequence_parallel:
        args.async_tensor_model_parallel_allreduce = False

    if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
        if args.sequence_parallel:
            raise RuntimeError(
                "Using sequence parallelism requires setting the environment variable "
                "CUDA_DEVICE_MAX_CONNECTIONS to 1")
        if args.async_tensor_model_parallel_allreduce:
            raise RuntimeError(
                "Using async gradient all reduce requires setting the environment "
                "variable CUDA_DEVICE_MAX_CONNECTIONS to 1")

    # Disable bias gelu fusion if we are disabling bias altogether
    if not args.add_bias_linear:
        args.bias_gelu_fusion = False

    # Retro checks.
    if args.retro_add_retriever:

        # Train samples should be auto-loaded.
        ensure_valid(args.train_samples is not None, \
            "args.train_samples should be auto-loaded from the retro config.")

        # Sequence parallelism unsupported.
        ensure_valid(not args.sequence_parallel, \
            "retro currently does not support sequence parallelism.")

        # Pipeline parallelism unsupported.
        ensure_valid(args.pipeline_model_parallel_size == 1, \
            "retro currently does not support pipeline parallelism.")

    if args.decoupled_lr is not None or args.decoupled_min_lr is not None:
        ensure_valid(not args.use_legacy_models, \
            '--decoupled-lr and --decoupled-min-lr is not supported in legacy models.')
        ensure_valid(not args.use_dist_ckpt, "Distributed checkpointing does not work with decoupled LR yet.")

    # Legacy RoPE arguments
    if args.use_rotary_position_embeddings:
        args.position_embedding_type = 'rope'
    if args.rotary_interleaved and args.apply_rope_fusion:
        raise RuntimeError('--rotary-interleaved does not work with rope_fusion.')
    if args.rotary_interleaved and args.use_legacy_models:
        raise RuntimeError('--rotary-interleaved is not supported in legacy models.')

    # Would just need to add 'NoPE' as a position_embedding_type to support this, but for now
    # don't allow it to keep things simple
    if not args.add_position_embedding and args.position_embedding_type != 'rope':
        raise RuntimeError('--no-position-embedding is deprecated, use --position-embedding-type')

    # MoE Spec check
    if args.num_experts == 0:
        args.num_experts = None
    if args.num_experts is not None:
        ensure_valid(args.spec is None, "Model Spec must be None when using MoEs")

    # Context parallel
    if args.context_parallel_size > 1:
        ensure_valid(not args.use_legacy_models, "Context parallelism is not supported in legacy models.")

    # Expert parallelism check
    if args.expert_model_parallel_size > 1:
        ensure_valid(args.num_experts is not None, "num_experts must be non None to use expert model parallelism")
        ensure_valid(args.num_experts % args.expert_model_parallel_size == 0, \
            "Number of experts should be a multiple of expert model parallel_size.")
        ensure_valid(not args.fp16, \
            "Expert parallelism is not supported with fp16 training.")

    # Distributed checkpointing checks
    if args.use_dist_ckpt and args.use_legacy_models:
        raise RuntimeError('--use-dist-ckpt is not supported in legacy models.')
    
    # Data blend checks
    ensure_valid(args.mock_data + \
           bool(args.data_path) + \
           any([args.train_data_path, args.valid_data_path, args.test_data_path]) \
           <= 1, "A single data source must be provided in training mode, else None")

    if args.use_tp_pp_dp_mapping:
        ensure_valid(args.context_parallel_size * args.expert_model_parallel_size <= 1, \
            "context_parallel and expert_model_parallel can't be used with tp-pp-dp mapping.")

    # Deterministic mode
    if args.deterministic_mode:
        ensure_valid(not args.use_flash_attn, 'Flash attention can not be used in deterministic mode.')

        all_reduce_choices = ["Tree", "Ring", "CollnetDirect", "CollnetChain", "^NVLS"]
        ensure_valid(os.getenv("NCCL_ALGO", -1) != -1 and os.getenv("NCCL_ALGO") in all_reduce_choices, \
            f"NCCL_ALGO must be one of {all_reduce_choices}.")

    # Update the printed args to reflect that `apply_query_key_layer_scaling` also controls `attention_softmax_in_fp32`
    if args.apply_query_key_layer_scaling:
        args.attention_softmax_in_fp32 = True

    # Checkpointing
    if args.ckpt_fully_parallel_save_deprecated and args.rank == 0:
        print('--ckpt-fully-parallel-save flag is deprecated and has no effect.'
              ' Use --no-ckpt-fully-parallel-save to disable parallel save.')
    use_dist_ckpt_and_not_ckpt_fully_parallel_save = args.use_dist_ckpt and not args.ckpt_fully_parallel_save
    use_distributed_optimizer_and_rank = args.use_distributed_optimizer and args.rank == 0
    if use_dist_ckpt_and_not_ckpt_fully_parallel_save and use_distributed_optimizer_and_rank:
        print('Warning: With non-parallel ckpt save and DistributedOptimizer,'
              ' it will be impossible to resume training with different parallelism.'
              ' Consider removing flag --no-ckpt-fully-parallel-save.')

    # Print arguments.
    _print_args("arguments", args)

    return args


pm.register_patch("megatron.training.arguments.validate_args", validate_args, force_patch=True)
pm.register_patch("mindspeed_mm.patchs.validate_args_patch.validate_args", validate_args_wrapper)
pm.apply_patches()
