# predict.py
import os
import json
import argparse
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
# Optional HEIC support: install `pillow-heif` to enable HEIC/HEIF image opening
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    # pillow-heif not installed -> HEIC images will not be openable; keep working for other formats
    pass

from torchvision import transforms

from tools.create_exp_folder import create_val_exp_folder
import model.vit_model as vit_models

# ===== 固定推理配置（与训练保持一致）=====
IMG_SIZE = 224
MEAN = (0.5, 0.5, 0.5)
STD  = (0.5, 0.5, 0.5)

# Allowed extensions (lowercase). Use .lower() in checks to be case-insensitive.
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}

def is_image_file(p: str) -> bool:
    return os.path.splitext(p)[-1].lower() in IMG_EXTS


def collect_images(input_path: str) -> List[str]:
    """支持：单张图片 / 文件夹（递归）"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"input path not found: {input_path}")

    if os.path.isfile(input_path):
        if not is_image_file(input_path):
            raise ValueError(f"input is file but not an image: {input_path}")
        return [input_path]

    imgs = []
    for root, _, files in os.walk(input_path):
        for fn in files:
            fp = os.path.join(root, fn)
            if is_image_file(fp):
                imgs.append(fp)

    imgs.sort()
    if len(imgs) == 0:
        raise ValueError(f"no images found under: {input_path}")
    return imgs


def load_class_indices(json_path: str) -> Optional[Dict[int, str]]:
    """
    class_indices.json 格式通常是：
      {"0":"daisy", "1":"roses", ...}
    返回：
      {0:"daisy", 1:"roses", ...}
    若 json_path 为空或不存在 -> None
    """
    if not json_path:
        return None
    if not os.path.exists(json_path):
        print(f"[WARN] class_indices.json not found: {json_path} -> will output class id only")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    out = {}
    for k, v in m.items():
        try:
            out[int(k)] = v
        except Exception:
            pass
    return out if out else None


def load_checkpoint(weights_path: str, device: torch.device) -> Tuple[Dict, Dict]:
    """
    返回：
      state_dict: 纯模型参数 dict
      raw_ckpt:   原始 ckpt（可能包含 epoch/optimizer_state/args 等）
    兼容两种保存格式：
    1) 纯 state_dict：直接是参数字典
    2) checkpoint：dict 里包含 model_state 或 state_dict
    """
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"], ckpt
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"], ckpt
    # 纯 state_dict
    return ckpt, {"state_dict": ckpt}


def infer_num_classes_from_state_dict(state_dict: Dict) -> Optional[int]:
    """尽量从 head.weight 推断类别数 K"""
    w = state_dict.get("head.weight", None)
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return int(w.shape[0])
    return None


def get_model_factory(model_name: str):
    """根据字符串拿到 vit_model.py 里的工厂函数"""
    if not hasattr(vit_models, model_name):
        candidates = [n for n in dir(vit_models) if n.startswith("vit_")]
        raise ValueError(f"Unknown model_name='{model_name}'. Example candidates: {candidates[:15]} ...")
    fn = getattr(vit_models, model_name)
    if not callable(fn):
        raise ValueError(f"model_name='{model_name}' exists but is not callable.")
    return fn


def build_val_transform():
    """固定输入尺寸为 IMG_SIZE，保证与 PatchEmbed 的 assert 对齐"""
    resize_size = int(IMG_SIZE / 224 * 256)  # 224 -> 256
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def _is_allowed_mismatch_key(k: str) -> bool:
    """
    这里定义“允许忽略”的 key（比如某些 checkpoint 可能带额外字段，但我们已经做了 model_state 提取）
    目前保守：不额外放行。
    """
    return False


def safe_load_state_dict(
    model: torch.nn.Module,
    state_dict: Dict,
    allow_partial: bool = False,
) -> None:
    """
    安全加载：
    - 默认：要求权重与模型结构严格匹配（shape / key 都匹配），否则报错，防止选错模型版本
    - allow_partial=True：只加载 shape 匹配的部分，并打印跳过信息（不建议用于正式推理）
    """
    model_sd = model.state_dict()

    unexpected = []
    missing = []
    mismatched = []

    # 只收集 shape 匹配的 key（用于 partial load）
    filtered = {}

    for k, v in state_dict.items():
        if _is_allowed_mismatch_key(k):
            continue

        if k not in model_sd:
            unexpected.append(k)
            continue

        if model_sd[k].shape != v.shape:
            mismatched.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue

        filtered[k] = v

    for k in model_sd.keys():
        if k not in state_dict:
            missing.append(k)

    # 默认严格：任何 mismatched / missing / unexpected 都直接报错（防止模型选错）
    if (not allow_partial) and (len(unexpected) > 0 or len(missing) > 0 or len(mismatched) > 0):
        # 打印更友好的错误信息（只展示前若干条）
        msg_lines = []
        msg_lines.append("Weights do NOT match the selected model architecture.")
        msg_lines.append("This usually means you selected the wrong --model-name for the given --weights.")
        msg_lines.append("")
        if mismatched:
            msg_lines.append(f"[Shape mismatch] count={len(mismatched)} (show up to 20):")
            for k, s_w, s_m in mismatched[:20]:
                msg_lines.append(f"  - {k}: weight={s_w}, model={s_m}")
            msg_lines.append("")
        if unexpected:
            msg_lines.append(f"[Unexpected keys in weights] count={len(unexpected)} (show up to 20):")
            for k in unexpected[:20]:
                msg_lines.append(f"  - {k}")
            msg_lines.append("")
        if missing:
            msg_lines.append(f"[Missing keys in weights] count={len(missing)} (show up to 20):")
            for k in missing[:20]:
                msg_lines.append(f"  - {k}")
            msg_lines.append("")
        msg_lines.append("Fix suggestions:")
        msg_lines.append("  1) Check that --model-name matches the checkpoint variant (B/16 vs B/32 vs L/16 etc).")
        msg_lines.append("  2) If you *really* want to partially load (NOT recommended), add --allow-partial-load.")
        raise RuntimeError("\n".join(msg_lines))

    # partial 或严格都用 filtered（严格时 filtered==全部匹配；不匹配早就抛错）
    msg = model.load_state_dict(filtered, strict=False)
    print("[INFO] load_state_dict (filtered, strict=False):", msg)

    # allow_partial 时额外提示
    if allow_partial and (unexpected or missing or mismatched):
        print(f"[WARN] Partial load enabled: loaded={len(filtered)} keys, "
              f"mismatched={len(mismatched)}, unexpected={len(unexpected)}, missing={len(missing)}")


@torch.no_grad()
def predict_one(model, img_pil: Image.Image, tfm, device: torch.device):
    model.eval()
    img = img_pil.convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)      # [1,3,224,224]
    logits = model(x)                         # [1,K]
    prob = torch.softmax(logits, dim=1)       # [1,K]
    pred_idx = int(prob.argmax(dim=1).item())
    pred_prob = float(prob[0, pred_idx].item())
    return pred_idx, pred_prob


def draw_text_on_image(img: Image.Image, text: str) -> Image.Image:
    """把预测结果写到图上"""
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    pad = 4
    x0, y0 = 6, 6
    bbox = draw.textbbox((x0, y0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([x0 - pad, y0 - pad, x0 + w + pad, y0 + h + pad], fill=(0, 0, 0))
    draw.text((x0, y0), text, fill=(255, 255, 255), font=font)
    return img


def main(args):
    # 1) device
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    # 2) create exp folder for prediction results
    exp_folder = create_val_exp_folder()
    os.makedirs(exp_folder, exist_ok=True)

    # 3) collect images
    img_paths = collect_images(args.data)
    print(f"[INFO] Found {len(img_paths)} images.")

    # 4) load class indices (optional)
    class_map = load_class_indices(args.class_indices)

    # 5) load weights
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"weights not found: {args.weights}")
    state_dict, raw_ckpt = load_checkpoint(args.weights, device)

    # 6) decide num_classes（优先从权重 head 推断）
    inferred_k = infer_num_classes_from_state_dict(state_dict)
    if inferred_k is None:
        if class_map is not None:
            inferred_k = len(class_map)
            print(f"[WARN] Cannot infer num_classes from weights; fallback to class_indices length={inferred_k}")
        elif args.num_classes is not None:
            inferred_k = int(args.num_classes)
            print(f"[WARN] Cannot infer num_classes from weights; fallback to --num-classes={inferred_k}")
        else:
            raise RuntimeError(
                "Cannot infer num_classes from weights (missing head.weight) and no class_indices / --num-classes provided."
            )
    num_classes = inferred_k

    # 7) build model（由 --model-name 决定）
    factory = get_model_factory(args.model_name)
    model = factory(num_classes=num_classes).to(device)

    # 8) safe load weights（默认严格，防止选错模型/权重）
    safe_load_state_dict(model, state_dict, allow_partial=False)

    # 9) transform
    tfm = build_val_transform()

    # 10) 如果 class_indices 存在且长度与 head 输出一致 -> 输出类别名，否则输出 id
    use_class_name = (class_map is not None) and (len(class_map) == num_classes)

    # 11) output txt
    txt_path = os.path.join(exp_folder, "predictions.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("image_path\tpred_id\tpred_name\tprob\n")

        for p in img_paths:
            img = Image.open(p).convert("RGB")
            pred_id, pred_prob = predict_one(model, img, tfm, device)

            pred_name = class_map.get(pred_id, str(pred_id)) if use_class_name else str(pred_id)
            f.write(f"{p}\t{pred_id}\t{pred_name}\t{pred_prob:.6f}\n")

            # 12) draw prediction on image (optional)
            if args.draw:
                text = f"{pred_name} ({pred_prob:.3f})"
                out_img = draw_text_on_image(img.copy(), text)
                out_name = os.path.basename(p)
                out_img.save(os.path.join(exp_folder, out_name))

    print(f"[INFO] Saved predictions to: {txt_path}")
    if args.draw:
        print(f"[INFO] Saved drawn images to: {exp_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="E:\PyTorch\VisionTransformer/vision_transformer\Actress")
    parser.add_argument("--weights", type=str, default="run/train/exp7/weights/best.pth")
    parser.add_argument("--class-indices", type=str, default="run/train/exp7/class_indices.json",
                        help="class_indices.json 路径；可为空/不匹配时只输出类别id")
    # 关键：必须显式指定你要用的模型工厂（防止选错）
    parser.add_argument("--model-name", type=str, default="vit_base_patch16_224_in21k",
                        help="模型工厂函数名（必须在 model/vit_model.py 里存在）")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="cuda:0 / cpu")
    # 正确的 bool 参数写法（默认 True，可用 --no-draw 关闭）
    parser.add_argument("--draw", action="store_true", default=True,
                        help="把预测类别写到图片上并保存（默认开启）")
    # 当权重里无法推断类别数时，用该参数指定 K
    parser.add_argument("--num-classes", type=int, default=None,
                        help="可选：当权重里无法推断类别数时，用该参数指定 K")
    args = parser.parse_args()

    main(args)
