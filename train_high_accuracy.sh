#!/bin/bash
# 天池新闻分类 - 95%+ 准确率优化训练脚本
#
# 模型优化:
# ✅ d_model=768 (BERT-base级别)
# ✅ num_layers=8 (更深网络)
# ✅ num_heads=12 (更多注意力)
# ✅ max_length=1024 (覆盖95%样本)
# ✅ Warmup + Cosine LR调度
# ✅ 标签平滑 (0.1)
# ✅ 梯度累积 (2步)
# ✅ 更优dropout (0.15)

echo "========================================"
echo "  天池新闻分类 - 高精度BERT训练"
echo "  目标: 验证集准确率 95%+"
echo "========================================"
echo ""

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

echo "模型配置:"
echo "- 模型维度: 768"
echo "- 编码器层数: 8"
echo "- 注意力头数: 12"
echo "- 最大序列长度: 1024"
echo "- 学习率调度: Warmup(10%) + Cosine Decay"
echo "- 标签平滑: 0.1"
echo "- 梯度累积: 2步"
echo ""

# ============================================
# 预期: 95-97% 准确率
# ============================================
read -p "训练完整版？(最佳效果，8-12小时) [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "完整版 - 全部样本"
    echo "----------------------------------------"
    echo "预期时间: 8-12小时"
    echo "预期准确率: 95-97%"
    echo ""

    python main.py train \
        --model-spec bert \
        --train-csv data/train_set.csv \
        --model-out models/model_bert_ultimate.pt \
        --epochs 15 \
        --batch-size 16 \
        --learning-rate 2e-5

    echo ""
    echo "训练完成！"
    echo ""
fi

echo ""
echo "========================================"
echo "  训练完成！"
echo "========================================"
echo ""
echo "推理示例:"
echo ""
echo "# 使用终极版模型 (最佳)"
echo "python main.py infer --model models/model_bert_ultimate.pt --model-type bert --input-csv data/test_a.csv --output-csv predictions_ultimate.csv"
echo ""
