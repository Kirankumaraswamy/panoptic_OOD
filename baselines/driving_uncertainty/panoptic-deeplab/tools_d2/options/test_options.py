from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='/home/kumarasw/OOD_dataset/bdd/synboost_bdd/results/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')

        # General
        parser.add_argument('--demo-folder', type=str, default='/home/kumarasw/OOD_dataset/bdd/bdd/leftImg8bit/train/city', help='path to the folder containing demo images',
                            required=False)
        parser.add_argument('--no-segmentation', action='store_true', help='if specified, do *not* segment images since they are already created')

        # Segmentation
        parser.add_argument('--snapshot', type=str, default='../models/image-segmentation/bdd_best_0124999.pth',
                            help='pre-trained Segmentation checkpoint', required=False)
        parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus',
                            help='Network architecture used for Segmentation inference')

        parser.add_argument('--config-file', type=str, default='../configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_X_65_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml',
                            help='config file to use to load the model')
        
        parser.set_defaults(preprocess_mode='fixed', crop_size=512, load_size=256, display_winsize=256, aspect_ratio=2.0)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        parser.set_defaults(use_vae=True)
        parser.set_defaults(gpu=0)
        self.isTrain = False
        return parser
