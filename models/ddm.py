import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import utils
from models.unet import DiffusionUNet
from models.decom import CTDN
import torchvision
import matplotlib.pyplot as plt 

class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas




class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        self.device = config.device

        self.Unet = DiffusionUNet(config)
        if self.args.mode == 'training':
            self.decom = self.load_stage1(CTDN(), 'ckpt/stage1')
        else:
            self.decom = CTDN()

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    @staticmethod
    def load_stage1(model, model_dir):
        checkpoint = utils.logging.load_checkpoint(os.path.join(model_dir, 'stage1_weight.pth.tar'), 'cuda')
        model.load_state_dict(checkpoint['model'], strict=True)
        return model

    def sample_training(self, x_cond, b, eta=0.):
        skip = self.config.diffusion.num_diffusion_timesteps // self.config.diffusion.num_sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = x_cond.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = self.Unet(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        return xs[-1]

    def forward(self, inputs):
        data_dict = {}

        b = self.betas.to(inputs.device)

        if self.training:
            output = self.decom(inputs, pred_fea=None)
            low_R, low_L, low_fea, high_L = output["low_R"], output["low_L"], \
                output["low_fea"], output["high_L"]
            low_condition_norm = utils.data_transform(low_fea)

            t = torch.randint(low=0, high=self.num_timesteps, size=(low_condition_norm.shape[0] // 2 + 1,)).to(
                self.device)
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:low_condition_norm.shape[0]].to(inputs.device)
            a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

            e = torch.randn_like(low_condition_norm)

            high_input_norm = utils.data_transform(low_R * high_L)

            x = high_input_norm * a.sqrt() + e * (1.0 - a).sqrt()
            noise_output = self.Unet(torch.cat([low_condition_norm, x], dim=1), t.float())

            pred_fea = self.sample_training(low_condition_norm, b)
            pred_fea = utils.inverse_data_transform(pred_fea)
            reference_fea = low_R * torch.pow(low_L, 0.2)

            data_dict["noise_output"] = noise_output
            data_dict["e"] = e

            data_dict["pred_fea"] = pred_fea
            data_dict["reference_fea"] = reference_fea

        else:
            output = self.decom(inputs, pred_fea=None)
            low_fea = output["low_fea"]
            low_condition_norm = utils.data_transform(low_fea)

            pred_fea = self.sample_training(low_condition_norm, b)
            pred_fea = utils.inverse_data_transform(pred_fea)
            pred_x = self.decom(inputs, pred_fea=pred_fea)["pred_img"]
            data_dict["pred_x"] = pred_x

        return data_dict


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint {} step {}".format(load_path, self.step))

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        self.epoch_losses = []            # 전체 loss
        self.epoch_noise_losses = []     # noise loss
        self.epoch_scc_losses = []       # scc loss
        self.epoch_grad_losses = []
        self.epoch_tv_losses = []
        for name, param in self.model.named_parameters():
            if "decom" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            epoch_loss = []
            epoch_noise_loss = []
            epoch_scc_loss = []
            epoch_grad_loss = []
            epoch_tv_loss = []

            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                self.model.train()
                self.step += 1

                x = x.to(self.device)

                output = self.model(x)

                noise_loss, scc_loss, grad_loss, tv_loss = self.noise_estimation_loss(output, epoch)
                loss = noise_loss + scc_loss + grad_loss + tv_loss
                epoch_loss.append(loss.item())
                epoch_noise_loss.append(noise_loss.item())
                epoch_scc_loss.append(scc_loss.item())
                epoch_grad_loss.append(grad_loss.item())
                epoch_tv_loss.append(tv_loss.item())
                data_time += time.time() - data_start

                if self.step % 10 == 0:
                    print("step:{}, noise_loss:{:.5f} scc_loss:{:.5f} grad_loss:{:.5f} tv_loss:{:.5f} time:{:.5f}".
                          format(self.step, noise_loss.item(),
                                 scc_loss.item(), grad_loss.item(), tv_loss.item(), data_time / (i + 1)))
                    save_dir = os.path.join(self.args.image_folder, "debug_feats", f"step_{self.step}")
                    os.makedirs(save_dir, exist_ok=True)

                    pred_fea = output["pred_fea"]  # F̂_low
                    reference_fea = output["reference_fea"]  # F̃_low

                    # Clamp and normalize if needed
                    pred_img = torch.clamp(pred_fea, 0, 1)
                    ref_img = torch.clamp(reference_fea, 0, 1)

                    for idx in range(min(4, pred_img.shape[0])):  # Save up to 4 samples
                        torchvision.utils.save_image(
                            pred_img[idx],
                            os.path.join(save_dir, f"pred_{idx}.png")
                        )
                        torchvision.utils.save_image(
                            ref_img[idx],
                            os.path.join(save_dir, f"ref_{idx}.png")
                        )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0 and self.step != 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)

                    utils.logging.save_checkpoint({'step': self.step,
                                                   'epoch': epoch + 1,
                                                   'state_dict': self.model.state_dict(),
                                                   'optimizer': self.optimizer.state_dict(),
                                                   'ema_helper': self.ema_helper.state_dict(),
                                                   'params': self.args,
                                                   'config': self.config},
                                                  filename=os.path.join(self.config.data.ckpt_dir, 'model_ours_4500_adaptive2'))
                

                        # ---- 에폭 종료 후 평균 loss 저장 ----
            # ---- 에폭 종료 후 평균 loss 저장 ----
            avg_loss = np.mean(epoch_loss)
            avg_noise = np.mean(epoch_noise_loss)
            avg_scc = np.mean(epoch_scc_loss)
            avg_grad = np.mean(epoch_grad_loss)
            avg_tv = np.mean(epoch_tv_loss)

            self.epoch_losses.append(avg_loss)
            self.epoch_noise_losses.append(avg_noise)
            self.epoch_scc_losses.append(avg_scc)
            self.epoch_grad_losses.append(avg_grad)
            self.epoch_tv_losses.append(avg_tv)
            
            print(f"Epoch {epoch} finished. Avg loss: {avg_loss:.5f} | noise: {avg_noise:.5f} | scc: {avg_scc:.5f} | tv: {avg_tv:.5f}  | grad: {avg_grad:.5f}")

        self.save_loss_curve()


    def save_loss_curve(self):
        save_path = self.args.image_folder
        os.makedirs(save_path, exist_ok=True)

        # 전체 loss
        plt.figure()
        plt.plot(range(len(self.epoch_losses)), self.epoch_losses, label='Total Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Total Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_path, "loss_curve_total.png"))
        plt.close()

        # Noise Loss
        plt.figure()
        plt.plot(range(len(self.epoch_noise_losses)), self.epoch_noise_losses, label='Noise Loss', color='orange')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Noise Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_path, "loss_curve_noise.png"))
        plt.close()

        # SCC Loss
        plt.figure()
        plt.plot(range(len(self.epoch_scc_losses)), self.epoch_scc_losses, label='SCC Loss', color='green')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SCC Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_path, "loss_curve_scc.png"))
        plt.close()
        
        # grad Loss
        plt.figure()
        plt.plot(range(len(self.epoch_grad_losses)), self.epoch_grad_losses, label='grad Loss', color='blue')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("grad Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_path, "loss_curve_grad.png"))
        plt.close()
        
        # tv Loss
        plt.figure()
        plt.plot(range(len(self.epoch_tv_losses)), self.epoch_tv_losses, label='tv Loss', color='red')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("tv Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_path, "loss_curve_tv.png"))
        plt.close()
        
        print("All loss curves saved in:", save_path)
        
    def gradient_loss(self, pred, target):
        def _gradient(x):
            dx = x[:, :, 1:, :] - x[:, :, :-1, :]
            dy = x[:, :, :, 1:] - x[:, :, :, :-1]
            return dx, dy

        dx_pred, dy_pred = _gradient(pred)
        dx_target, dy_target = _gradient(target)
        return self.l1_loss(dx_pred, dx_target) + self.l1_loss(dy_pred, dy_target)

    def tv_loss(self, img):
        batch_size = img.size(0)
        h_x = img.size(2)
        w_x = img.size(3)
        count_h = self.l1_loss(img[:, :, 1:, :], img[:, :, :-1, :])
        count_w = self.l1_loss(img[:, :, :, 1:], img[:, :, :, :-1])
        return (count_h + count_w) / batch_size
    '''
    def noise_estimation_loss(self, output):
        pred_fea, reference_fea = output["pred_fea"], output["reference_fea"]
        noise_output, e = output["noise_output"], output["e"]

        noise_loss = self.l2_loss(noise_output, e)
        scc_loss = 0.005 * self.l1_loss(pred_fea, reference_fea)

        #grad_loss = 0.05 * self.gradient_loss(pred_fea, reference_fea)
        tv = 0.05 * self.tv_loss(pred_fea)

        return noise_loss, scc_loss, tv #, grad_loss
    '''
    def noise_estimation_loss(self, output, epoch):
        pred_fea, reference_fea = output["pred_fea"], output["reference_fea"]
        noise_output, e = output["noise_output"], output["e"]

        noise_loss = self.l2_loss(noise_output, e)
        scc_loss = 0.005 * self.l1_loss(pred_fea, reference_fea)

        # adaptive weight
        max_epochs = self.config.training.n_epochs
        progress = epoch / max_epochs
        grad_weight = 0.05
        tv_weight = 0.005 * progress

        grad_loss = grad_weight * self.gradient_loss(pred_fea, reference_fea)
        tv = tv_weight * self.tv_loss(pred_fea)

        return noise_loss, scc_loss, grad_loss, tv
    

    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder,
                                    self.config.data.type + str(self.config.data.patch_size))
        self.model.eval()

        with torch.no_grad():
            print('Performing validation at step: {}'.format(step))
            for i, (x, y) in enumerate(val_loader):
                b, _, img_h, img_w = x.shape

                img_h_64 = int(64 * np.ceil(img_h / 64.0))
                img_w_64 = int(64 * np.ceil(img_w / 64.0))
                x = F.pad(x, (0, img_w_64 - img_w, 0, img_h_64 - img_h), 'reflect')
                pred_x = self.model(x.to(self.device))["pred_x"][:, :, :img_h, :img_w]
                utils.logging.save_image(pred_x, os.path.join(image_folder, str(step), '{}'.format(y[0])))