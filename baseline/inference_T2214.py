import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm

from dataset import TestDataset, MaskBaseDataset
from inference import load_model
from PIL import Image
from importlib import import_module
import shutil

@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls()
    
    model_path = os.path.join(model_dir, args.name, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for _, images in enumerate(tqdm(loader,leave=True)):
            images = images['image'].to(device)
            if args.multi:
                mask_out, gender_out, age_out = model(images)
                mask_out = mask_out.argmax(dim=-1).cpu().numpy()
                gender_out = gender_out.argmax(dim=-1).cpu().numpy()
                age_out = age_out.argmax(dim=-1).cpu().numpy()
                pred = MaskBaseDataset.encode_multi_class(mask_out, gender_out, age_out)
            else:
                pred = model(images)
                pred = pred.argmax(dim=-1)
                pred = pred.cpu().numpy()
            preds.extend(pred)

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'{args.output_name}.csv'), index=False)
    print(f'Inference Done!')
    return info

def split_save_img(info):
    save_dir = 'eval_data'
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    print('making save dir')
    for i in range(18):
        idirpath = os.path.join(save_dir, str(i))
        # 
        os.makedirs(idirpath)
    img_root = os.path.join(data_dir, 'images')
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    preds = info['ans'].values
    print('save img to other dir...')
    for img_path, pred in zip(img_paths, preds):
        im = Image.open(img_path)
        img_sav_path = os.path.join(save_dir, str(pred))
        id = img_path.split('/')[-1]
        im.save(f'{img_sav_path}/{id}')
    print('done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for validing (default: 64)')
    parser.add_argument('--resize', nargs="+", type=int,  default=[512, 384], help='resize size for image when you trained (default: [512, 384])')
    parser.add_argument('--model', type=str, default='Resnet18_multi', help='model type (default: BaseModel)')
    parser.add_argument('--model_version', type=str, default='b0', help='model version (default: b0)')
    parser.add_argument('--multi', type=bool, default=True, help='multi label (default: True)')
    parser.add_argument('--name', default='resnet18_multi2', help='model saved dir: {SM_CHANNEL_MODEL}/{name}')
    parser.add_argument('--output_name', type=str, default='output', help='submission saved name: {name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/T2214'))
    # parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './model/T2214/resnet18_multi2'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = f'{args.model_dir}/{args.name}'

    os.makedirs(output_dir, exist_ok=True)

    info = inference(data_dir, model_dir, output_dir, args)
    info_path = os.path.join(output_dir, f'{args.output_name}.csv')
    info = pd.read_csv(info_path)
    print(info.groupby('ans').size())
    split_save_img(info)
