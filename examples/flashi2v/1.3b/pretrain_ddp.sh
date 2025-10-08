#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 该变量只用于规避megatron对其校验，对npu无效
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=1
VP=1
CP=1
MBS=1
GRAD_ACC_STEP=1
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

MM_DATA="examples/flashi2v/1.3b/data.json"
MM_MODEL="examples/flashi2v/1.3b/pretrain_model_flashi2v.json"
MM_TOOL="./mindspeed_mm/tools/tools.json"
SAVE_PATH="./save"
PROJECT_NAME="test_new_mm_script"
PROJECT_EXP_NAME="test_new_mm_script"
PROJECT_DIR=$SAVE_PATH

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --virtual-pipeline-model-parallel-size ${VP} \
    --context-parallel-size ${CP} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 2e-5 \
    --min-lr 2e-5 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-15 \
    --lr-decay-style constant \
    --weight-decay 1e-2 \
    --lr-warmup-init 0 \
    --lr-warmup-iters 0 \
    --clip-grad 1.0 \
    --train-iters 1000000 \
    --no-gradient-accumulation-fusion \
    --bf16 \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 0 \
    --recompute-skip-core-attention \
    --recompute-num-layers-skip-core-attention 22 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --optimizer-selection fused_ema_adamw \
    --seed 1024 \
    --data-parallel-random-init \
    --use-ema \
    --fp32-residual-connection \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32 \
"

MM_ARGS="
    --mm-data $MM_DATA \
    --mm-model $MM_MODEL \
    --mm-tool $MM_TOOL \
    --clip_grad_ema_decay 0.99
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10 \
    --eval-interval 5 \
    --eval-iters 10 \
    --load $SAVE_PATH \
    --save $SAVE_PATH \
"

WANDB_ARGS="
    --wandb-project $PROJECT_NAME \
    --wandb-exp-name $PROJECT_EXP_NAME \
    --wandb-save-dir $PROJECT_DIR \
    --tensorboard-log-interval 1 \
"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
torchrun $DISTRIBUTED_ARGS pretrain_flashi2v.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    $WANDB_ARGS \
    --distributed-backend nccl \
    2>&1 | tee logs/train_${logfile}.log

chmod 440 logs/train_${logfile}.log
chmod -R 640 $SAVE_PATH
STEP_TIME=`grep "elapsed time per iteration" logs/train_${logfile}.log | awk -F ':' '{print$5}' | awk -F '|' '{print$1}' | head -n 200 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`
SPS=`awk 'BEGIN{printf "%.3f\n", '${GBS}'*1000/'${STEP_TIME}'}'`
echo "Elapsed Time Per iteration: $STEP_TIME, Average Samples per Second: $SPS"