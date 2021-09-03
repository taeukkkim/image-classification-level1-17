import argparse
import os
import pandas as pd


def make_result(data_dir, output_dir, args):

    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    mask = pd.read_csv(os.path.join(output_dir, f'{args.output_mask}.csv'))
    gender = pd.read_csv(os.path.join(output_dir, f'{args.output_gender}.csv'))
    age = pd.read_csv(os.path.join(output_dir, f'{args.output_age}.csv'))

    preds = mask['ans'] * 6 + gender['ans'] * 3 + age['ans']


    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'{args.output_name}.csv'), index=False)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--output_name', type=str, default='output', help='submission saved name: {output_name}')
    parser.add_argument('--output_mask', type=str, default='EfficientNet_b7_0902_mask', help='mask model output: {output_mask}')
    parser.add_argument('--output_gender', type=str, default='EfficientNet_b7_0902_gender', help='gender model output: {output_gender}')
    parser.add_argument('--output_age', type=str, default='EfficientNet_b7_0902_age', help='age model output: {output_age}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    make_result(data_dir, output_dir, args)