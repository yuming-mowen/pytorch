import os
import math
import argparse

import csv
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


from tools.my_dataset import build_vit_dataloaders
from model import vit_model as vit_models
from tools.utils import read_split_data, train_one_epoch, evaluate, ConsolePrinter  # 数据划分、单epoch训练、验证评估函数
from tools.create_exp_folder import create_exp_folder
from tools.plot_metrics import plot_from_metrics_csv, plot_val_prf_curves, save_confusion_matrices


# 用于“权重-模型不匹配”时给出更明确的提示（按vit_model 里的工厂函数命名）
MODEL_SIGS = {
    "vit_base_patch16_224_in21k":  {"patch_size": 16, "embed_dim": 768,  "depth": 12},
    "vit_base_patch32_224_in21k":  {"patch_size": 32, "embed_dim": 768,  "depth": 12},
    "vit_large_patch16_224_in21k": {"patch_size": 16, "embed_dim": 1024, "depth": 24},
    "vit_large_patch32_224_in21k": {"patch_size": 32, "embed_dim": 1024, "depth": 24},
    "vit_huge_patch14_224_in21k":  {"patch_size": 14, "embed_dim": 1280, "depth": 32},
}

def _strip_module_prefix(state_dict):
    # 兼容 DataParallel / DDP 保存的 "module.xxx"
    if not isinstance(state_dict, dict):
        return state_dict
    if not state_dict:
        return state_dict
    has_module = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module:
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def _infer_vit_sig_from_weights(state_dict):
    """
    从权重里尽量推断出：patch_size / embed_dim / depth
    用于当用户选错模型时给更友好的提示
    """
    sig = {"patch_size": None, "embed_dim": None, "depth": None}

    w = state_dict.get("patch_embed.proj.weight", None)
    if w is not None and hasattr(w, "shape") and len(w.shape) == 4:
        # [embed_dim, in_c, patch, patch]
        sig["embed_dim"] = int(w.shape[0])
        sig["patch_size"] = int(w.shape[2])

    # depth：看 blocks.{i}.xxx 最大 i
    max_idx = -1
    for k in state_dict.keys():
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                max_idx = max(max_idx, int(parts[1]))
    if max_idx >= 0:
        sig["depth"] = max_idx + 1

    return sig


def _suggest_models_by_sig(sig):
    """
    根据推断的 (patch_size, embed_dim, depth) 给出可能匹配的模型名
    """
    ps, ed, dp = sig.get("patch_size"), sig.get("embed_dim"), sig.get("depth")
    if ps is None or ed is None or dp is None:
        return []

    cands = []
    for name, s in MODEL_SIGS.items():
        if s["patch_size"] == ps and s["embed_dim"] == ed and s["depth"] == dp:
            cands.append(name)
    return cands


def _smart_load_weights(model, ckpt, args, device):
    # 兼容两种格式：
    # 1) 纯 state_dict（直接就是参数字典）
    # 2) checkpoint（含 model_state/optimizer_state/...）
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    state_dict = _strip_module_prefix(state_dict)

    # 如果是“训练保存的 checkpoint”，可强校验 model 是否一致（避免你说的：B 权重配 L 模型）
    if isinstance(ckpt, dict) and "args" in ckpt and isinstance(ckpt["args"], dict):
        old_model = ckpt["args"].get("model", None)
        if old_model is not None and hasattr(args, "model") and args.model != old_model:
            raise RuntimeError(
                f"Checkpoint was trained with model={old_model}, "
                f"but now you selected --model={args.model}. Please make them一致。"
            )

    # 只删除分类头（类别数一定不匹配）
    for k in ["head.weight", "head.bias"]:
        state_dict.pop(k, None)

    # 自动处理：过滤掉 shape 不匹配的 key，并统计匹配比例
    model_sd = model.state_dict()
    expected_keys = [k for k in model_sd.keys() if not k.startswith("head.")]
    filtered = {}
    shape_mismatch = []
    unexpected = []

    for k, v in state_dict.items():
        if k in model_sd:
            if model_sd[k].shape == v.shape:
                filtered[k] = v
            else:
                shape_mismatch.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        else:
            unexpected.append(k)

    matched = sum(1 for k in expected_keys if k in filtered)
    keep_ratio = matched / max(1, len(expected_keys))

    # 如果匹配比例过低，基本就是“模型选错了”，直接报错并给提示
    # （否则 strict=False 可能让你误以为加载成功，但其实没加载多少）
    MIN_KEEP_RATIO = 0.85
    if keep_ratio < MIN_KEEP_RATIO:
        w_sig = _infer_vit_sig_from_weights(state_dict)
        suggestions = _suggest_models_by_sig(w_sig)

        msg = []
        msg.append(f"权重与当前模型不匹配（keep_ratio={keep_ratio:.2%} < {MIN_KEEP_RATIO:.0%}）")
        msg.append(f"当前选择 --model={getattr(args, 'model', None)}")
        msg.append(f"从权重推断到的结构特征：patch_size={w_sig.get('patch_size')}, embed_dim={w_sig.get('embed_dim')}, depth={w_sig.get('depth')}")
        if suggestions:
            msg.append("你更可能应该使用：")
            for s in suggestions:
                msg.append(f"   --model {s}")
        else:
            msg.append("建议：确认你选择的 --model 是否与权重对应（Base/Large/Huge、patch_size、embed_dim、depth 必须一致）。")

        # 额外给出几个最关键的 shape mismatch 例子，方便你定位
        if shape_mismatch:
            msg.append("部分 shape mismatch 示例（只显示前 5 个）：")
            for k, wsh, msh in shape_mismatch[:5]:
                msg.append(f"  - {k}: weight{wsh} vs model{msh}")

        raise RuntimeError("\n".join(msg))

    # 走到这里说明“基本匹配”，允许 strict=False 加载（并把不匹配的部分留给随机初始化）
    msg = model.load_state_dict(filtered, strict=False)
    print(msg)

    # 额外打印：哪些 key 因 shape 不匹配被跳过（少量时很正常，比如你改了分辨率/pos_embed 等）
    if shape_mismatch:
        print(f"skipped {len(shape_mismatch)} keys due to shape mismatch (showing first 10):")
        for k, wsh, msh in shape_mismatch[:10]:
            print(f"  - {k}: weight{wsh} vs model{msh}")

    return model


def build_model_and_prepare(args, device, num_classes: int):
    create_model = getattr(vit_models, args.model, None)
    if create_model is None or not callable(create_model):
        # 给出可选项：只列出 vit_model 里“看起来像 ViT 工厂函数”的名字
        candidates = [n for n in MODEL_SIGS.keys() if hasattr(vit_models, n)]
        raise ValueError(
            f"Unknown model: {args.model}\n"
            f"Available candidates: {candidates}"
        )

    model = create_model(num_classes=num_classes).to(device)

    if args.weights:
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
        ckpt = torch.load(args.weights, map_location=device)

        model = _smart_load_weights(model, ckpt, args, device)

    # freeze：只训练 pre_logits + head
    if args.freeze_layers:
        for name, p in model.named_parameters():
            if ("head" not in name) and ("pre_logits" not in name):
                p.requires_grad_(False)
            else:
                print(f"training {name}")

    return model


def main(args):
    # 设备选择：优先使用 args.device（例如 cuda:0），若无 GPU 则回退到 CPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 调用函数获取新的exp文件夹和weights文件夹路径
    exp_folder, weights_folder = create_exp_folder()

    # 在 main() 开头 exp_folder 创建之后，加：
    metrics_path = os.path.join(exp_folder, "metrics.csv")

    # 写表头（只写一次）
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_p", "val_r", "val_f1", "lr"])

    # 读取并划分数据集：返回训练/验证集图片路径与标签
    train_images_path, train_images_label, val_images_path, val_images_label, num_classes = read_split_data(
        args.data_path,
        val_rate=0.2,
        exp_folder=exp_folder,
        seed=0)

    # 构建训练/验证DataLoader
    # 输入：训练/验证集的图片路径列表 + 标签列表
    # 输出：两个可迭代对象 train_loader / val_loader，用于训练循环 for images, labels in loader
    train_loader, val_loader = build_vit_dataloaders(
        train_images_path, train_images_label,  # 训练集：路径list + label list
        val_images_path, val_images_label,  # 验证集：路径list + label list
        batch_size=args.batch_size  # 每个 batch 样本数
    )

    # 构建并准备模型（迁移学习入口）
    # - create_model(num_classes=K)：创建ViT，并把分类头改成下游任务的K类
    # - 若 args.weights非空：加载预训练权重（通常只加载backbone，删除head/pre_logits避免形状不匹配）
    # - 若 args.freeze_layers=True：冻结除head/pre_logits外的参数，只微调分类头（适合小数据集）
    model = build_model_and_prepare(args, device, num_classes)

    # ===================== 优化器与学习率调度器 =====================
    # 构建优化器：只优化requires_grad=True的参数
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # 学习率调度器：余弦退火（Cosine LR）
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # ===================== 权重保存相关 =====================
    os.makedirs(weights_folder, exist_ok=True)
    last_ckpt_path = os.path.join(weights_folder, "last.pth")
    best_ckpt_path = os.path.join(weights_folder, "best.pth")
    best_val_acc = -1.0
    best_epoch = -1

    # ===================== 训练 =====================
    # 训练循环：按epoch迭代
    printer = ConsolePrinter()
    for epoch in range(args.epochs):
        # ===== train header（蓝色）=====
        print()
        print(printer.train_header(colored=True))
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            epochs=args.epochs
        )

        scheduler.step()

        # ===== val header（黄色）=====
        print(printer.val_header(colored=True))
        val_loss, val_acc, val_p, val_r, val_f1 = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch,
            epochs=args.epochs,
            num_classes=num_classes,
            indent_spaces=16
        )

        # 读取当前学习率（scheduler.step() 之后 optimizer 里的 lr 已更新）
        # param_groups 是 PyTorch 优化器的“参数组列表”
        # 这里取第 0 组的学习率（常见情况只有一组）
        lr_now = optimizer.param_groups[0]["lr"]

        # 统一转成 Python float，方便后续：
        # - 写入 metrics.csv
        val_acc_value = float(val_acc.item()) if hasattr(val_acc, "item") else float(val_acc)
        train_acc_value = float(train_acc.item()) if hasattr(train_acc, "item") else float(train_acc)

        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc_value, val_loss, val_acc_value, val_p, val_r, val_f1, lr_now])

        # ===================== 新增：保存 last / best =====================
        # val_acc 兼容 float / tensor
        val_acc_value = float(val_acc.item()) if hasattr(val_acc, "item") else float(val_acc)

        # 保存 last：每个 epoch 覆盖写，最终得到最后一轮模型
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "args": vars(args),
        }, last_ckpt_path)

        # 保存 best：若 val_acc 更好则更新
        if val_acc_value > best_val_acc:
            best_val_acc = val_acc_value
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "args": vars(args),
            }, best_ckpt_path)

    # 训练结束后自动绘图
    metrics_path = os.path.join(exp_folder, "metrics.csv")
    plot_from_metrics_csv(metrics_path, out_dir=exp_folder, smooth=3)

    # 训练全部结束后：画 PRF 曲线 + 混淆矩阵
    plot_val_prf_curves(metrics_path, exp_folder)  # 保存：exp_folder/val_prf_curve.png

    # ===== 训练结束后：用 best 权重来画混淆矩阵 =====
    # 训练结束后：加载 best 权重
    best_path = os.path.join(weights_folder, "best.pth")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # 画混淆矩阵（best）
    save_confusion_matrices(model, val_loader, device, num_classes, exp_folder)

    print(f"curves saved to: {exp_folder} (loss_curve.*, acc_curve.*)")
    print(f"Training done. Best val_acc={best_val_acc:.4f} at epoch={best_epoch}")
    print(f"Last checkpoint: {last_ckpt_path}")
    print(f"Best checkpoint: {best_ckpt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 训练与任务相关参数
    parser.add_argument('--epochs', type=int, default=50)        # 训练轮数
    parser.add_argument('--batch-size', type=int, default=128)     # batch size（注意属性名是 opt.batch_size）
    parser.add_argument('--lr', type=float, default=0.001)        # 初始学习率
    parser.add_argument('--lrf', type=float, default=0.01)        # 最终学习率比例（cosine schedule 末端比例）
    # 数据路径
    parser.add_argument('--data-path', type=str, default="Actress")  # 数据集根目录
    parser.add_argument('--model', type=str, default="vit_base_patch16_224_in21k",
                        help='选择模型工厂函数名，例如 vit_base_patch16_224_in21k / vit_large_patch16_224_in21k')
    # 迁移学习相关
    # --weights：预训练权重路径，不想加载就传空字符串
    parser.add_argument('--weights', type=str, default='weights/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
                        help='initial weights path')
    # 是否冻结 backbone（常用于小数据集微调：只训练 head / pre_logits）
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # 设备选择
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    # 解析命令行参数
    # 例如：python train.py --epochs 50 --batch-size 32 --device cpu
    opt = parser.parse_args()
    # 调用主训练函数 main(opt)
    main(opt)


