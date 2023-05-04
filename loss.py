import torch
import torch.nn as nn


def G_wgan_acgan(G, D, latent_vector, device, cond_weight = 1.0):

    fake_images = G(latent_vector)
    fake_results = D(fake_images)
    loss = torch.nn.functional.softplus(-fake_results)

    batch_size = fake_results.shape[0]
    label_penalty_fakes = nn.CrossEntropyLoss()(fake_results, torch.ones(batch_size, 1).to(device))

    loss += label_penalty_fakes * cond_weight

    return torch.mean(loss)


def D_wgangp_acgan(
        D,
        fake_images, real_images,
        fake_results, real_results,
        device,
        wgan_lambda     = 10.0,
        wgan_epsilon    = 0.001,
        wgan_target     = 1.0,
        cond_weight     = 1.0
    ):

    batch_size = real_images.shape[0]

    loss = fake_results - real_results

    mixing_factors = torch.rand(batch_size, 1, 1, 1).to(device)
    mixed_images = torch.lerp(real_images, fake_images, mixing_factors)
    mixed_results = D(mixed_images)
    mixed_loss = torch.sum(mixed_results)
    mixed_grads = torch.autograd.grad(mixed_loss, mixed_images, create_graph=True)[0]
    mixed_norms = torch.sqrt(torch.sum(torch.square(mixed_grads), axis=[1, 2, 3]))
    gradient_penalty = torch.square(mixed_norms - wgan_target).unsqueeze(1)
    loss += gradient_penalty * (wgan_lambda / (wgan_target ** 2))

    epsilon_penalty = torch.square(real_results)
    loss += epsilon_penalty * wgan_epsilon

    real_labels = real_results.squeeze() > 0
    real_labels = real_labels.float()
    fake_labels = fake_results.squeeze() > 0
    fake_labels = fake_labels.float()

    label_penalty_reals = nn.CrossEntropyLoss()(real_labels, torch.ones(batch_size).to(device))
    label_penalty_fakes = nn.CrossEntropyLoss()(fake_labels, torch.zeros(batch_size).to(device))
    loss += (label_penalty_reals + label_penalty_fakes) * cond_weight

    return torch.mean(loss)


def D_wgangp_acgan_simplified(fake_results, real_results, device):
    
    batch_size = real_results.shape[0]

    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    loss = torch.nn.functional.softplus(real_results) + torch.nn.functional.softplus(fake_results)
    loss += nn.CrossEntropyLoss()(torch.cat([real_results, fake_results]), torch.cat([real_labels, fake_labels]))

    return torch.mean(loss)


def G_mean(fake_results1, fake_results2):

    loss = torch.mean(torch.nn.functional.softplus(-fake_results1)) + torch.mean(torch.nn.functional.softplus(-fake_results2))

    return loss


def D_mean(fake_results, real_results):

    real_loss = torch.mean(torch.nn.functional.softplus(-real_results))
    fake_loss = torch.mean(torch.nn.functional.softplus(fake_results))

    return real_loss + fake_loss
