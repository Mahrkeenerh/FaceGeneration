import os
from tqdm import tqdm

import cv2
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader

import dataset_tools
import loss
import modules


# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

size_levels = 5
epochs_per_size = 40
easing_steps = 15

batch_sizes = [128, 128, 128, 128, 128, 128]
latent_dim = 512

dataset_root = "/home/xbuban1/celebahq_out"


generator = modules.Generator(latent_dim, easing_steps).to(device)
discriminator = modules.Discriminator(latent_dim, easing_steps).to(device)

lr = 1e-4

gen_optim = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
disc_optim = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

if not os.path.exists("previews"):
    os.mkdir("previews")

if not os.path.exists("models"):
    os.mkdir("models")

latent_vector_preview = dataset_tools.get_latent_vector(16, latent_dim).to(device)

for size_level in range(size_levels + 1):
    print("Loading dataset...")
    dataset_name = f"r{size_level + 2:02d}"
    dataset = dataset_tools.FaceDataset(dataset_root, f"celebahq_out-{dataset_name}.tfrecords")
    dataloader = DataLoader(dataset, batch_size=batch_sizes[size_level], shuffle=True)

    if size_level > 0:
        generator.increase_size()
        discriminator.increase_size()

    if not os.path.exists(f"previews/previews_{size_level}"):
        os.mkdir(f"previews/previews_{size_level}")

    for epoch in range(epochs_per_size):
        train_tqdm = tqdm(dataloader, desc=f"Size: {size_level} Epoch {epoch + 1}/{epochs_per_size}")

        real_mean, fake_mean = 0, 0

        for real_images in train_tqdm:
            batch_size = real_images.shape[0]

            real_images = real_images.to(device)
            latent_vector = dataset_tools.get_latent_vector(batch_size, latent_dim).to(device)
            fake_images = generator(latent_vector)

            real_results = discriminator(real_images)
            fake_results = discriminator(fake_images)

            disc_loss = loss.D_mean(fake_results, real_results)

            disc_optim.zero_grad()
            disc_loss.backward()
            disc_optim.step()

            # Freeze discriminator
            for param in discriminator.parameters():
                param.requires_grad = False

            latent_vector1 = dataset_tools.get_latent_vector(batch_size, latent_dim).to(device)
            latent_vector2 = dataset_tools.get_latent_vector(batch_size, latent_dim).to(device)
            fake_images1 = generator(latent_vector1)
            fake_images2 = generator(latent_vector2)
            fake_results1 = discriminator(fake_images1)
            fake_results2 = discriminator(fake_images2)
            # gen_loss = loss.G_wgan_acgan(generator, discriminator, latent_vector, device)
            gen_loss = loss.G_mean(fake_results1, fake_results2)

            gen_optim.zero_grad()
            gen_loss.backward()
            gen_optim.step()

            # Unfreeze discriminator
            for param in discriminator.parameters():
                param.requires_grad = True

            real_mean = real_mean * 0.9 + torch.mean(real_results).item() * 0.1
            fake_mean = fake_mean * 0.9 + torch.mean(fake_results).item() * 0.1


            train_tqdm.set_postfix(real=real_mean, fake=fake_mean)

        generator.step()
        discriminator.step()

        # Generate preview images
        fake_images = generator(latent_vector_preview)
        fake_images = fake_images.detach().cpu().numpy()
        fake_images = np.transpose(fake_images, (0, 2, 3, 1))
        fake_images = (fake_images + 1) / 2

        # Concat images into 4x4 image
        resolution = 2 ** (size_level + 2)
        out_image = np.zeros((4 * resolution, 4 * resolution, 3))
        for i in range(4):
            for j in range(4):
                out_image[i * resolution : (i + 1) * resolution, j * resolution : (j + 1) * resolution, :] = fake_images[i * 4 + j]

        # Upscale to 1024x1024
        out_image = cv2.resize(out_image, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        out_image = cv2.cvtColor(out_image.astype('float32'), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"previews/previews_{size_level}/{epoch + 1}.png", out_image * 255)

        # Clear cuda memory
        del fake_images
        torch.cuda.empty_cache()

    torch.save(generator.state_dict(), f"models/generator_{size_level}.pt")
    torch.save(discriminator.state_dict(), f"models/discriminator_{size_level}.pt")
