import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.gridspec import GridSpec
from configs import Config
from utils import *
from nets import NetS
from scipy.io import savemat, loadmat


class NSSR:
    def __init__(self, indata, conf=Config()):
        self.conf = conf
        self.cuda = conf.cuda
        self.dev = conf.dev if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.dev)
        self.input = indata
        self.init_parameters()
        self.model = NetS().to(self.device)
        self.hr_fathers_sources = [self.input]

    def init_parameters(self):
        self.loss = [None] * self.conf.max_iters
        self.mse_steps, self.mse_rec, self.interp_rec_mse = [], [], []
        self.learning_rate = self.conf.learning_rate
        self.learning_rate_change_iter_nums = [0]
        self.base_sf, self.base_ind, self.iter = 1.0, 0, 0

    def run(self):
        for self.sf_ind, sf in enumerate(self.conf.scale_factors):
            print('** Scale Factor =', sf, ' **')
            sf = [sf, sf] if np.isscalar(sf) else sf
            self.sf = np.array(sf) / np.array(self.base_sf)
            print('Input Shape = ',self.input.shape)
            self.init_parameters()
            self.train()
            post_processed_output = self.final_test()
            self.hr_fathers_sources.append(post_processed_output)
            self.base_change()
            if self.conf.save_results:
                print('Output Shape = ', post_processed_output.shape)
            print('** Finish Training for ', sf, ' **\n')
        return post_processed_output

    def train(self):
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for self.iter in range(self.conf.max_iters):
            self.hr_father = random_augment(ims=self.hr_fathers_sources,
                                            base_scales=[1.0] + self.conf.scale_factors,
                                            leave_as_is_probability=self.conf.augment_leave_as_is_probability,
                                            no_interpolate_probability=self.conf.augment_no_interpolate_probability,
                                            min_scale=self.conf.augment_min_scale,
                                            max_scale=([1.0] + self.conf.scale_factors)[len(self.hr_fathers_sources)-1],
                                            allow_rotation=self.conf.augment_allow_rotation,
                                            scale_diff_sigma=self.conf.augment_scale_diff_sigma,
                                            shear_sigma=self.conf.augment_shear_sigma,
                                            crop_size=self.conf.crop_size)
            self.lr_son = self.father_to_son(self.hr_father)            
            self.train_output = self.forward_backward_pass(self.lr_son, self.hr_father, criterion, optimizer)
            if not self.iter % self.conf.display_every:
                print('Scale Factor =', self.sf*self.base_sf, ', Iteration =', self.iter, ', Train Loss =', self.loss[self.iter])
            if self.conf.run_test and (not self.iter % self.conf.run_test_every):
                if self.quick_test(): break
            self.learning_rate_policy()
            if self.learning_rate < self.conf.min_learning_rate:
                break

    def father_to_son(self, hr_father):
        lr_son = imresize(hr_father, 1.0 / self.sf)
        return np.clip(lr_son + np.random.randn(*lr_son.shape) * self.conf.noise_std, 0, 1)

    def forward_backward_pass(self, lr_son, hr_father, criterion, optimizer):
        interpolated_lr_son = imresize(lr_son, self.sf, hr_father.shape, self.conf.upscale_method)
        lr_son_input = torch.Tensor(interpolated_lr_son).permute(2,0,1).unsqueeze_(0).unsqueeze_(0).requires_grad_() # N,C,D,H,W
        hr_father = torch.Tensor(hr_father).permute(2,0,1).unsqueeze_(0).unsqueeze_(0).to(self.device)
        lr_son_input = lr_son_input.to(self.device)
        train_output = self.model(lr_son_input)
        loss = criterion(hr_father, train_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.loss[self.iter] = loss.item()
        return np.clip(np.squeeze(train_output.cpu().detach().numpy()), 0, 1)

    def quick_test(self):
        reconstruct_output = self.forward_pass(self.father_to_son(self.input), self.input.shape)
        self.mse_rec.append(np.mean(np.ndarray.flatten(np.square(self.input - reconstruct_output))))
        interp_rec = imresize(self.father_to_son(self.input), self.sf, self.input.shape[0:2], self.conf.upscale_method)
        self.interp_rec_mse.append(np.mean(np.ndarray.flatten(np.square(self.input - interp_rec))))
        self.mse_steps.append(self.iter)
        print('Iteration =', self.iter, ', Re MSE =', self.mse_rec[-1], ', In MSE =', self.interp_rec_mse[-1])
        return self.mse_rec[-1] <= self.interp_rec_mse[-1]

    def forward_pass(self, lr_son, hr_father_shape=None):
        interpolated_lr_son = imresize(lr_son, self.sf, hr_father_shape, self.conf.upscale_method)
        interpolated_lr_son = (torch.Tensor(interpolated_lr_son).permute(2,0,1)).unsqueeze_(0).unsqueeze_(0)
        if self.cuda: interpolated_lr_son = interpolated_lr_son.to(self.device)
        return np.clip(np.squeeze(self.model(interpolated_lr_son).cpu().detach().permute(0,3,4,2,1).numpy()), 0, 1)

    def learning_rate_policy(self):
        if (not (1 + self.iter) % self.conf.learning_rate_policy_check_every
                and self.iter - self.learning_rate_change_iter_nums[-1] > self.conf.min_iters):
            [slope, _], [[var, _], _] = np.polyfit(self.mse_steps[-(self.conf.learning_rate_slope_range //
                                                                    self.conf.run_test_every):],
                                                   self.mse_rec[-(self.conf.learning_rate_slope_range //
                                                                  self.conf.run_test_every):],
                                                   1, cov=True)
            if -self.conf.learning_rate_change_ratio * slope < np.sqrt(var):
                self.learning_rate /= 10
                print("Learning Rate Updated =", self.learning_rate)
                self.learning_rate_change_iter_nums.append(self.iter)

    def final_test(self):
        outputs = []
        for k in range(0, 1 + 7 * self.conf.output_flip, 1 + int(self.sf[0] != self.sf[1])):
            test_input = np.rot90(self.input, k) if k < 4 else np.fliplr(np.rot90(self.input, k))
            tmp_output = self.forward_pass(test_input)
            tmp_output = np.rot90(tmp_output, -k) if k < 4 else np.rot90(np.fliplr(tmp_output), -k)
            for _ in range(self.conf.back_projection_iters[self.sf_ind]):
                tmp_output = back_projection(tmp_output, self.input, down_kernel=None,
                                             up_kernel=self.conf.upscale_method, sf=self.sf)
            outputs.append(tmp_output)
        almost_final_sr = np.median(outputs, 0)
        for _ in range(self.conf.back_projection_iters[self.sf_ind]):
            almost_final_sr = back_projection(almost_final_sr, self.input, down_kernel=None,
                                              up_kernel=self.conf.upscale_method, sf=self.sf)
        self.final_sr = almost_final_sr
        return self.final_sr

    def base_change(self):
        if len(self.conf.base_change_sfs) <= self.base_ind: return
        if abs(self.conf.scale_factors[self.sf_ind] - self.conf.base_change_sfs[self.base_ind]) < 0.001:
            self.input = self.final_sr
            self.base_sf = self.conf.base_change_sfs[self.base_ind]
            self.base_ind += 1
            print('Base Changed = %.1f' % self.base_sf)
