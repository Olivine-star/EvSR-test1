import os

class Config:
    def __init__(self):
        self.initialize_parameter()
        self.initialize_method()
        self.initialize_learning_rate_policy()
        self.initialize_data_augmentation()
        self.initialize_input()
        self.initialize_output()
        self.initialize_info_display()

    def initialize_parameter(self):
        self.seed = 0
        self.cuda = True
        self.dev = "cuda"
        self.dvs_size = [129, 129]
        self.scale_factors = [[4, 4]]
        self.back_projection_iters = [10]
        self.init_net_for_each_sf = False
        self.crop_size = 128
        self.max_iters = 3000
        self.min_iters = 256
        self.noise_std = 0 
        self.batch_size_t = -1
        self.cutoff_low = 0
        self.cutoff_high = 1

    def initialize_method(self):
        self.downscale_method = 'cubic'
        self.upscale_method = 'cubic'
        self.downscale_gt_method = 'cubic'
        self.output_flip = True
        self.base_change_sfs = []

    def initialize_learning_rate_policy(self):
        self.learning_rate = 0.001
        self.learning_rate_t = 0.001
        self.min_learning_rate = 9e-6  
        self.learning_rate_change_ratio = 1.5
        self.learning_rate_policy_check_every = 60
        self.learning_rate_slope_range = 256
        self.adjust_period = 4000
        self.adjust_ratio = 10
        self.epochs_t = 10000

    def initialize_data_augmentation(self):
        self.augment_leave_as_is_probability = 0.05
        self.augment_no_interpolate_probability = 0.45
        self.augment_min_scale = 0.5
        self.augment_scale_diff_sigma = 0.25
        self.augment_shear_sigma = 0.1
        self.augment_allow_rotation = True

    def initialize_input(self):
        self.file_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.file_name = 'sample129.npy'

    def initialize_output(self):
        self.create_result_dir = True
        self.result_path = os.path.join(os.path.dirname(__file__), 'result')
        self.sample = 'sample'
        self.create_code_copy = True
        self.save_results = True

    def initialize_info_display(self):
        self.display_every = 20
        self.show_every = 400
        self.run_test = True
        self.run_test_every = 50