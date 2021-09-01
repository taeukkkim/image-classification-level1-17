import argparse
import os
from dotenv import load_dotenv
from train import train, train_multi

parser = argparse.ArgumentParser()
load_dotenv(verbose=True)

# Data and model checkpoints directories
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDatasetMulti', help='dataset augmentation type (default: MaskBaseDataset)')
parser.add_argument('--augmentation', type=str, default='get_transforms', help='data augmentation type (default: BaseAugmentation)')
parser.add_argument("--resize", nargs="+", type=int, default=[512, 384], help='resize size for image when training')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 64)')
parser.add_argument('--model', type=str, default='Resnet18', help='model type (default: BaseModel)')
parser.add_argument('--optimizer', type=str, default='focal_smoothing', help='optimizer type (default: SGD)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
parser.add_argument('--lr_gamma', type=float, default=0.5, help='learning rate scheduler gamma (default: 0.5)')
parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
parser.add_argument('--model_version', type=str, default='b0', help='model version (default: b0)')
parser.add_argument('--multi', type=bool, default=False, help='multi output label option, class num - mask:3, gender:2, age:3 (default: False)')

# Container environment
parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model/T2214'))

args = parser.parse_args()
print(args)

data_dir = args.data_dir
model_dir = args.model_dir

if args.multi:
    train_multi(data_dir, model_dir, args)
else:
    train(data_dir, model_dir, args)
