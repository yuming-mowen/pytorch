import torch
from dataset import HorseZebraDataset
import os
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator

# 单个 epoch 的训练函数
def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    # tqdm 用于显示训练进度条
    loop = tqdm(loader, leave=True)
    for idx, (zebra, horse) in enumerate(loop):
        # 将数据移动到指定设备（CPU / GPU）
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # 1. 训练判别器 Discriminator（H 和 Z）
        # 使用混合精度训练以加速并减少显存占用
        with torch.amp.autocast("cuda"):
            # 生成假马图像（Zebra → Horse）
            fake_horse = gen_H(zebra)
            # 判别器对真实马图像的判断
            D_H_real = disc_H(horse)
            # 判别器对假马图像的判断（detach 防止梯度回传到生成器）
            D_H_fake = disc_H(fake_horse.detach())
            # Horse 判别器的真实/虚假损失
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            # 生成假斑马图像（Horse → Zebra）
            fake_zebra = gen_Z(horse)
            # 判别器对真实斑马图像的判断
            D_Z_real = disc_Z(zebra)
            # 判别器对假斑马图像的判断
            D_Z_fake = disc_Z(fake_zebra.detach())
            # Zebra 判别器的真实/虚假损失
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # 两个判别器的总损失取平均
            D_loss = (D_H_loss + D_Z_loss) / 2

        # 判别器反向传播与参数更新
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # 2. 训练生成器 Generator（H 和 Z）
        with torch.amp.autocast("cuda"):
            # 生成器希望“骗过”判别器，因此标签为 1
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # Cycle Consistency Loss
            # Zebra → Horse → Zebra
            cycle_zebra = gen_Z(fake_horse)
            # Horse → Zebra → Horse
            cycle_horse = gen_H(fake_zebra)
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)

            # Identity Loss
            # 将斑马输入 Zebra→Zebra 生成器，输出应接近原图
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            identity_zebra_loss = l1(zebra, identity_zebra)
            identity_horse_loss = l1(horse, identity_horse)

            # 生成器总损失
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_zebra_loss * config.LAMBDA_CYCLE
                + cycle_horse_loss * config.LAMBDA_CYCLE
                + identity_horse_loss * config.LAMBDA_IDENTITY
                + identity_zebra_loss * config.LAMBDA_IDENTITY
            )

        # 生成器反向传播与参数更新
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # 每隔一定步数保存生成结果，便于观察训练效果
        if idx % 200 == 0:
            # 将 [-1,1] 映射回 [0,1] 后保存图像
            save_image(fake_horse * 0.5 + 0.5, f"saved_images/horse_{idx}.png")
            save_image(fake_zebra * 0.5 + 0.5, f"saved_images/zebra_{idx}.png")


# 主训练入口
def main():
    os.makedirs("saved_images", exist_ok=True)
    # 实例化判别器（Horse 和 Zebra）
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)

    # 实例化生成器（Zebra→Horse 和 Horse→Zebra）
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    # 判别器优化器
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    # 生成器优化器
    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    # 定义损失函数
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    # 是否加载已有模型权重
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE,
        )

    # 构建数据集
    dataset = HorseZebraDataset(
        root_horse=config.TRAIN_DIR + "/trainA",
        root_zebra=config.TRAIN_DIR + "/trainB",
        transform=config.transforms,
    )

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 混合精度的梯度缩放器
    g_scaler = torch.amp.GradScaler("cuda")
    d_scaler = torch.amp.GradScaler("cuda")

    # 训练多个 epoch
    for epoch in range(config.NUM_EPOCHS):
        print("[Epoch]:", epoch + 1, "/", config.NUM_EPOCHS, "/n")
        train_fn(
            disc_H, disc_Z,
            gen_Z, gen_H,
            loader,
            opt_disc, opt_gen,
            L1, mse,
            d_scaler, g_scaler
        )

        # 是否保存模型
        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)


if __name__ == "__main__":
    main()
