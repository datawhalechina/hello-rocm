#!/bin/bash
# 第六章：预训练执行脚本
# 
# 使用方式：
#   bash pretrain.sh                    # 单卡训练
#   bash pretrain.sh -n 4               # 4 卡 DDP 训练
#   bash pretrain.sh -d ds_config.json  # 使用 DeepSpeed

set -e  # 错误时退出

# ============================================================================
# 配置参数
# ============================================================================

# 模型和数据
MODEL_NAME="Qwen/Qwen2.5-1.5B"
DATASET_NAME="wikitext"
OUTPUT_DIR="./outputs/pretrain"

# 训练参数
EPOCHS=1
BATCH_SIZE=4
LEARNING_RATE=2e-4

# 分布式参数
NUM_GPUS=1
DEEPSPEED_CONFIG=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -d|--deepspeed)
            DEEPSPEED_CONFIG="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# 环境检查
# ============================================================================

echo "========================================="
echo "预训练脚本 - AMD ROCm 版本"
echo "========================================="

# 检查 ROCm
echo "检查 ROCm 环境..."
rocm-smi > /dev/null || { echo "错误: 未安装 ROCm"; exit 1; }

# 检查 GPU
echo "检查 GPU..."
GPU_COUNT=$(rocm-smi --json | grep '"gpu_id"' | wc -l)
echo "可用 GPU 数量: $GPU_COUNT"

if [ $NUM_GPUS -gt $GPU_COUNT ]; then
    echo "警告: 指定 $NUM_GPUS 个 GPU，但只有 $GPU_COUNT 个可用"
    NUM_GPUS=$GPU_COUNT
fi

# ============================================================================
# 准备输出目录
# ============================================================================

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

echo "输出目录: $OUTPUT_DIR"

# ============================================================================
# 构建训练命令
# ============================================================================

PYTHON_SCRIPT="pretrain.py"

# 基础参数
TRAIN_ARGS="--model_name_or_path $MODEL_NAME"
TRAIN_ARGS="$TRAIN_ARGS --dataset_name $DATASET_NAME"
TRAIN_ARGS="$TRAIN_ARGS --output_dir $OUTPUT_DIR"
TRAIN_ARGS="$TRAIN_ARGS --num_train_epochs $EPOCHS"
TRAIN_ARGS="$TRAIN_ARGS --per_device_train_batch_size $BATCH_SIZE"
TRAIN_ARGS="$TRAIN_ARGS --learning_rate $LEARNING_RATE"
TRAIN_ARGS="$TRAIN_ARGS --bf16"
TRAIN_ARGS="$TRAIN_ARGS --save_strategy steps"
TRAIN_ARGS="$TRAIN_ARGS --save_steps 500"
TRAIN_ARGS="$TRAIN_ARGS --logging_steps 100"

# ============================================================================
# 执行训练
# ============================================================================

echo ""
echo "========================================="
echo "开始训练"
echo "========================================="
echo "模型: $MODEL_NAME"
echo "GPU 数量: $NUM_GPUS"
echo "批大小: $BATCH_SIZE"
echo "学习率: $LEARNING_RATE"

if [ ! -z "$DEEPSPEED_CONFIG" ]; then
    echo "使用 DeepSpeed 配置: $DEEPSPEED_CONFIG"
    echo ""
    
    # 使用 DeepSpeed
    deepspeed --num_gpus=$NUM_GPUS \
        $PYTHON_SCRIPT \
        $TRAIN_ARGS \
        --deepspeed $DEEPSPEED_CONFIG
else
    echo "使用 DDP 多卡训练"
    echo ""
    
    # 使用 DDP
    if [ $NUM_GPUS -gt 1 ]; then
        torchrun --nproc_per_node $NUM_GPUS \
            $PYTHON_SCRIPT \
            $TRAIN_ARGS
    else
        python $PYTHON_SCRIPT $TRAIN_ARGS
    fi
fi

# ============================================================================
# 完成
# ============================================================================

echo ""
echo "========================================="
echo "训练完成！"
echo "========================================="
echo "模型位置: $OUTPUT_DIR/final"
echo "日志位置: $OUTPUT_DIR/logs"
echo ""
echo "查看训练日志:"
echo "  tensorboard --logdir $OUTPUT_DIR/logs"
echo ""
echo "继续进行微调:"
echo "  python finetune.py --model_name_or_path $OUTPUT_DIR/final"
