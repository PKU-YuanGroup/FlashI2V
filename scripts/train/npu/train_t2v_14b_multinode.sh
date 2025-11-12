pkill -9 pt_main_thread
pkill -9 python
pkill -9 -f train.py
ps aux | grep '[p]ython' | awk '{print $2}' | xargs -r kill -9
tmux new -s clash -d "cd /work/share/projects/clash && ./clash -f 723.yaml"
source /work/share/projects/clash/export.sh
sleep 5s
echo "start process..."
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export WANDB_MODE="online"
export WANDB_API_KEY="720d886d8c437c2142c88056a1eab8ef78d64a1f"
wandb login --relogin $WANDB_API_KEY

export TOKENIZERS_PARALLELISM=false

export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export MULTI_STREAM_MEMORY_REUSE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=0
export ACL_DEVICE_SYNC_TIMEOUT=3600

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29501}
NPRC_PER_NODE=${NPRC_PER_NODE:-8}
NNODES=${PET_NNODES:-1}
NODE_RANK=${RANK:-0}
WORLD_SIZE=$(($NNODES * $NPRC_PER_NODE))

torchrun \
  --nproc_per_node=${NPRC_PER_NODE} \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train/train.py \
  --config configs/train/npu/t2v_14b.yaml