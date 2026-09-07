import os
import torch
import json
from collections import defaultdict
from models import create_model
import argparse
from utils.logger import get_logger
import numpy as np
import pandas as pd
import time
from torch.utils.data import DataLoader
from dataset import *

class Opt:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)


def unpack_checkpoint(payload, fallback_model):
    """Return model name, state dict and saved config for old or new checkpoints."""
    if isinstance(payload, dict) and 'model_state_dict' in payload:
        return (
            payload.get('model_name', fallback_model),
            payload['model_state_dict'],
            payload.get('config', {}),
        )
    return fallback_model, payload, {}


def validate_checkpoint_features(payload, args):
    """Reject accidental evaluation with features unlike those used to train a new checkpoint."""
    if not isinstance(payload, dict) or not payload.get('feature_config'):
        return
    expected = payload['feature_config']
    actual = {
        'audio_feature': args.audiofeature_method,
        'video_feature': args.videofeature_method,
        'split_window': args.splitwindow_time,
    }
    for key, value in actual.items():
        if key in expected and str(expected[key]) != str(value):
            raise ValueError(
                f"Checkpoint feature mismatch for {key}: saved={expected[key]!r}, requested={value!r}"
            )


def normalize_model_weights(weights_text, model_count):
    if not weights_text:
        return [1.0 / model_count] * model_count
    weights = [float(value) for value in weights_text.split(',')]
    if len(weights) != model_count:
        raise ValueError(f"The number of model weights ({len(weights)}) must equal models ({model_count})")
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("The sum of model weights must be positive")
    return [weight / weight_sum for weight in weights]

def infer_probs(model, data_loader, device):
    model.eval()
    probs_list = []
    with torch.no_grad():
        for data in data_loader:
            for k, v in data.items():
                data[k] = v.to(device)
            model.set_input(data)
            if hasattr(model, 'test'):
                model.test()
            else:
                model.forward()
            probs_list.append(model.emo_pred.detach().cpu().numpy())
    return np.concatenate(probs_list, axis=0)

def build_subject_ids(audio_paths):
    subject_ids = []
    for path in audio_paths:
        name = os.path.basename(path)
        if '_' in name:
            subject_ids.append(name.split('_')[0])
        else:
            subject_ids.append(name.replace('.npy', ''))
    return subject_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Bi-CFNet Adapted Model")
    parser.add_argument('--labelcount', type=int, default=2, help="Number of data categories (2, 3, or 5).")
    parser.add_argument('--track_option', type=str, required=True, help="Track1 or Track2")
    parser.add_argument('--feature_max_len', type=int, required=True, help="Max length of feature.")
    parser.add_argument('--data_rootpath', type=str, required=True, help="Root path to the program dataset")
    parser.add_argument('--train_model', action='append', required=True, help="Path to a training model")
    parser.add_argument('--model_weights', type=str, default=None, help="Comma-separated weights")
    parser.add_argument('--test_json', type=str, required=False, help="File name of the testing JSON file")
    parser.add_argument('--personalized_features_file', type=str, help="File name of the personalized features file")
    parser.add_argument('--audiofeature_method', type=str, default='wav2vec', choices=['mfccs', 'opensmile', 'wav2vec'])
    parser.add_argument('--videofeature_method', type=str, default='openface', choices=['openface', 'resnet', 'densenet'])
    parser.add_argument('--splitwindow_time', type=str, default='1s')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model', type=str, default='our', help="Fallback model name for legacy checkpoints")

    args = parser.parse_args()

    # 自动检测数据目录结构：兼容有/无 Testing/ 子目录两种情况
    if os.path.isdir(os.path.join(args.data_rootpath, 'Testing')):
        data_prefix = os.path.join(args.data_rootpath, 'Testing')
    else:
        data_prefix = args.data_rootpath

    args.test_json = os.path.join(data_prefix, 'labels', 'Testing_files.json')
    args.personalized_features_file = os.path.join(data_prefix, 'individualEmbedding', 'descriptions_embeddings_with_ids.npy')

    config = load_config('config.json')
    opt = Opt(config)

    opt.emo_output_dim = args.labelcount
    opt.feature_max_len = args.feature_max_len
    opt.lr = args.lr
    opt.model = args.model

    audio_path = os.path.join(data_prefix, f"{args.splitwindow_time}", 'Audio', f"{args.audiofeature_method}") + '/'
    video_path = os.path.join(data_prefix, f"{args.splitwindow_time}", 'Visual', f"{args.videofeature_method}") + '/'

    # 自动获取输入维度
    for filename in os.listdir(audio_path):
        if filename.endswith('.npy'):
            opt.input_dim_a = np.load(audio_path + filename).shape[1]
            break

    for filename in os.listdir(video_path):
        if filename.endswith('.npy'):
            opt.input_dim_v = np.load(video_path + filename).shape[1]
            break

    opt.name = f'BiCFNet_{args.splitwindow_time}_{args.labelcount}labels_{args.audiofeature_method}+{args.videofeature_method}'
    logger_path = os.path.join(opt.log_dir, opt.name)
    if not os.path.exists(opt.log_dir):
        os.mkdir(opt.log_dir)
    if not os.path.exists(logger_path):
        os.mkdir(logger_path)
    logger = get_logger(logger_path, 'result')

    test_data = json.load(open(args.test_json, 'r'))
    logger.info('The number of testing samples = %d' % len(test_data))

    def build_loader():
        return DataLoader(
            AudioVisualDataset(test_data, args.labelcount, args.personalized_features_file, opt.feature_max_len,
                               batch_size=args.batch_size, audio_path=audio_path, video_path=video_path, isTest=True),
            batch_size=args.batch_size, shuffle=False,
        )

    model_paths = args.train_model
    weights = normalize_model_weights(args.model_weights, len(model_paths))

    logger.info(f"Using {len(model_paths)} model(s) for ensemble with weights: {weights}")

    probs_list = []
    for model_path in model_paths:
        payload = torch.load(model_path, map_location=torch.device(args.device))
        validate_checkpoint_features(payload, args)
        model_name, state_dict, saved_config = unpack_checkpoint(payload, args.model)
        for key, value in saved_config.items():
            setattr(opt, key, value)
        opt.model = model_name
        opt.isTrain = False
        model = create_model(opt)
        model.load_state_dict(state_dict)
        model.to(args.device)
        test_loader = build_loader()
        probs_list.append(infer_probs(model, test_loader, args.device))

    ensemble_probs = np.zeros_like(probs_list[0])
    for weight, probs in zip(weights, probs_list):
        ensemble_probs += weight * probs

    pred = ensemble_probs.argmax(axis=1)
    filenames = [item["audio_feature_path"] for item in test_data if "audio_feature_path" in item]
    subject_ids = build_subject_ids(filenames)

    if args.labelcount == 2:
        label = "bin"
    elif args.labelcount == 3:
        label = "tri"
    elif args.labelcount == 5:
        label = "pen"

    pred_col_name = f"{args.splitwindow_time}_{label}"
    result_dir = f"./answer_{args.track_option}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    csv_file = f"{result_dir}/submission.csv"

    # Subject-level majority voting
    subject_to_indices = defaultdict(list)
    for idx, subject_id in enumerate(subject_ids):
        subject_to_indices[subject_id].append(idx)

    voted_pred = pred.copy()
    for subject_id, indices in subject_to_indices.items():
        labels = pred[indices]
        counts = np.bincount(labels, minlength=args.labelcount)
        top = counts.max()
        winners = np.where(counts == top)[0]
        if len(winners) == 1:
            vote = int(winners[0])
        else:
            mean_probs = ensemble_probs[indices].mean(axis=0)
            vote = int(mean_probs.argmax())
        voted_pred[indices] = vote

    pred = voted_pred
    id_list = [item["audio_feature_path"].replace(".npy", "") for item in test_data]

    result_df = pd.DataFrame({
        "ID": id_list,
        pred_col_name: pred
    })

    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file)
        if pred_col_name in existing_df.columns:
            existing_df = existing_df.drop(columns=[pred_col_name])
        merged_df = pd.merge(existing_df, result_df, on="ID", how="outer")
    else:
        merged_df = result_df

    merged_df.to_csv(csv_file, index=False)
    logger.info(f"Testing complete. Results saved to: {csv_file}. Shape={merged_df.shape}")

    # ------------------ 自动评估：检测同目录下是否有真实标签 ------------------
    import glob as _glob
    from sklearn.metrics import f1_score, accuracy_score
    track_suffix = args.track_option.replace('Track1', 'Elderly').replace('Track2', 'Young')
    parent_dir = os.path.dirname(args.data_rootpath)
    gt_candidates = _glob.glob(os.path.join(parent_dir, f'MM2025_{args.track_option}_{track_suffix}.json'))
    if not gt_candidates:
        gt_candidates = _glob.glob(os.path.join(parent_dir, f'*{args.track_option}*{track_suffix}*.json'))
    if gt_candidates:
        gt_path = gt_candidates[0]
        logger.info(f"Found ground truth file: {gt_path}")
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        gt_df = pd.DataFrame(gt_data)
        merged = gt_df.merge(result_df, left_on='test_id', right_on='ID', how='inner')
        if len(merged) > 0:
            y_true = merged[f'label_{label}']
            y_pred = merged[pred_col_name]
            class_counts = np.bincount(y_true)
            sample_weights_arr = 1.0 / (class_counts[y_true] + 1e-6)
            acc_w = accuracy_score(y_true, y_pred, sample_weight=sample_weights_arr)
            acc_u = accuracy_score(y_true, y_pred)
            f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            f1_u = f1_score(y_true, y_pred, average='macro', zero_division=0)
            cm = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Pred'])
            logger.info(f"===== Test Set Evaluation ({pred_col_name}) =====")
            logger.info(f"Acc(W)={acc_w:.4f}  Acc(U)={acc_u:.4f}  F1(W)={f1_w:.4f}  F1(U)={f1_u:.4f}")
            logger.info(f"Confusion Matrix:\n{cm}")
        else:
            logger.warning("Ground truth found but could not be aligned with predictions.")
    else:
        logger.info("No ground truth file found; skipping automatic evaluation.")
