import os
import argparse
import numpy as np
from tqdm import tqdm
import time
import math
import cv2
import platform

import torch
from torch.utils.data import DataLoader

from dataset import Shuttlecock_Trajectory_Dataset, Video_IterableDataset
from utils.general import *
# Note: Avoid top-level MoviePy import to reduce dependency surface
# from testing_3 import testing

import pickle

def change_fps(video_file):
    """Convert video to 30 FPS and save to a temporary file to avoid corruption."""
    try:
        # Lazy import MoviePy only when conversion is requested
        from moviepy.video.io.VideoFileClip import VideoFileClip
        # Use MoviePy for more reliable FPS conversion
        clip = VideoFileClip(video_file)
        
        # If already 30 FPS, no need to convert
        if abs(clip.fps - 30.0) < 0.1:
            clip.close()
            return video_file
            
        # Create temporary filename
        base_name, ext = os.path.splitext(video_file)
        temp_file = f"{base_name}_30fps{ext}"
        
        # Write video with 30 FPS
        clip.write_videofile(temp_file, fps=30, codec='libx264', audio=False, 
                            temp_audiofile=None, verbose=False, logger=None)
        clip.close()
        
        print(f"Converted {video_file} to 30 FPS and saved as {temp_file}")
        return temp_file
    except Exception as e:
        print(f"Error converting video to 30 FPS: {e}")
        return video_file

def predict(indices, y_pred=None, c_pred=None, img_scaler=(1, 1)):
    """ Predict coordinates from heatmap or inpainted coordinates. 

        Args:
            indices (torch.Tensor): indices of input sequence with shape (N, L, 2)
            y_pred (torch.Tensor, optional): predicted heatmap sequence with shape (N, L, H, W)
            c_pred (torch.Tensor, optional): predicted inpainted coordinates sequence with shape (N, L, 2)
            img_scaler (Tuple): image scaler (w_scaler, h_scaler)

        Returns:
            pred_dict (Dict): dictionary of predicted coordinates
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
    """

    pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

    batch_size, seq_len = indices.shape[0], indices.shape[1]
    indices = indices.detach().cpu().numpy()if torch.is_tensor(indices) else indices.numpy()
    
    # Transform input for heatmap prediction
    if y_pred is not None:
        y_pred = y_pred > 0.5
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_pred = to_img_format(y_pred) # (N, L, H, W)
    
    # Transform input for coordinate prediction
    if c_pred is not None:
        c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred

    prev_f_i = -1
    for n in range(batch_size):
        for f in range(seq_len):
            f_i = indices[n][f][1]
            if f_i != prev_f_i:
                if c_pred is not None:
                    # Predict from coordinate
                    c_p = c_pred[n][f]
                    cx_pred, cy_pred = int(c_p[0] * WIDTH * img_scaler[0]), int(c_p[1] * HEIGHT* img_scaler[1]) 
                elif y_pred is not None:
                    # Predict from heatmap
                    y_p = y_pred[n][f]
                    bbox_pred = predict_location(to_img(y_p))
                    cx_pred, cy_pred = int(bbox_pred[0]+bbox_pred[2]/2), int(bbox_pred[1]+bbox_pred[3]/2)
                    cx_pred, cy_pred = int(cx_pred*img_scaler[0]), int(cy_pred*img_scaler[1])
                    # print(f"Frame {f_i}: Heatmap dimensions (H, W): {y_p.shape[-2:]}, Bounding box (x, y, width, height): {bbox_pred}")
                else:
                    raise ValueError('Invalid input')
                vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                pred_dict['Frame'].append(int(f_i))
                pred_dict['X'].append(cx_pred)
                pred_dict['Y'].append(cy_pred)
                pred_dict['Visibility'].append(vis_pred)
                prev_f_i = f_i
            else:
                break
     
    return pred_dict    


# ------------------------------
# Model cache and perf settings
# ------------------------------
_MODEL_CACHE = {}

def _get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def get_models(tracknet_file, inpaintnet_file=None, device=None):
    """Load and cache models to avoid re-loading per segment."""
    device = device or _get_device()
    key = (tracknet_file, inpaintnet_file or 'None', str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    tracknet_ckpt = torch.load(tracknet_file, map_location=device)
    tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
    tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).to(device)
    tracknet.load_state_dict(tracknet_ckpt['model'])
    tracknet.eval()

    inpaintnet, inpaintnet_seq_len = None, None
    if inpaintnet_file:
        inpaintnet_ckpt = torch.load(inpaintnet_file, map_location=device)
        inpaintnet_seq_len = inpaintnet_ckpt['param_dict']['seq_len']
        inpaintnet = get_model('InpaintNet').to(device)
        inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
        inpaintnet.eval()

    _MODEL_CACHE[key] = {
        'tracknet': tracknet,
        'tracknet_seq_len': tracknet_seq_len,
        'bg_mode': bg_mode,
        'inpaintnet': inpaintnet,
        'inpaintnet_seq_len': inpaintnet_seq_len,
        'device': device,
    }
    return _MODEL_CACHE[key]


# ------------------------------
# Local utilities (avoid importing test.py)
# ------------------------------
def get_ensemble_weight(seq_len, eval_mode):
    if eval_mode == 'average':
        weight = torch.ones(seq_len) / seq_len
    elif eval_mode == 'weight':
        weight = torch.ones(seq_len)
        for i in range(math.ceil(seq_len/2)):
            weight[i] = (i+1)
            weight[seq_len-i-1] = (i+1)
        weight = weight / weight.sum()
    else:
        raise ValueError('Invalid mode')
    return weight

def predict_location(heatmap):
    if np.amax(heatmap) == 0:
        return 0, 0, 0, 0
    else:
        (cnts, _) = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in cnts]
        max_area_idx = 0
        max_area = rects[0][2] * rects[0][3]
        for i in range(1, len(rects)):
            area = rects[i][2] * rects[i][3]
            if area > max_area:
                max_area_idx = i
                max_area = area
        x, y, w, h = rects[max_area_idx]
        return x, y, w, h

def generate_inpaint_mask(pred_dict, th_h=30):
    y = np.array(pred_dict['Y'])
    vis_pred = np.array(pred_dict['Visibility'])
    inpaint_mask = np.zeros_like(y)
    i = 0
    j = 0
    threshold = th_h
    while j < len(vis_pred):
        while i < len(vis_pred)-1 and vis_pred[i] == 1:
            i += 1
        j = i
        while j < len(vis_pred)-1 and vis_pred[j] == 0:
            j += 1
        if j == i:
            break
        elif i == 0 and y[j] > threshold:
            inpaint_mask[:j] = 1
        elif (i > 1 and y[i-1] > threshold) and (j < len(vis_pred) and y[j] > threshold):
            inpaint_mask[i:j] = 1
        else:
            pass
        i = j
    return inpaint_mask.tolist()


# ------------------------------
# Datasets (top-level for Windows pickling)
# ------------------------------
class SimpleFramesDataset(torch.utils.data.Dataset):
    def __init__(self, frames_chw_np: np.ndarray, seq_len: int, sliding_step: int, bg_mode: str = '', median_chw: np.ndarray | None = None):
        self.frames = frames_chw_np  # (N,3,H,W) float32
        self.seq_len = seq_len
        self.sliding_step = sliding_step
        self.bg_mode = bg_mode or ''
        self.median_chw = median_chw  # (3,H,W) float32 normalized
        if len(self.frames) < self.seq_len:
            self.length = 0
        else:
            self.length = (len(self.frames) - self.seq_len) // self.sliding_step + 1
    def __len__(self):
        return self.length
    def __getitem__(self, index: int):
        start = index * self.sliding_step
        end = start + self.seq_len
        window = self.frames[start:end]  # (L,3,H,W)
        x = window.reshape(self.seq_len * 3, HEIGHT, WIDTH)  # (L*3,H,W)
        # handle concat background mode: prepend median channels
        if self.bg_mode == 'concat' and self.median_chw is not None:
            x = np.concatenate([self.median_chw, x], axis=0)  # ((L+1)*3,H,W)
        data_idx = np.stack([(0, start + f) for f in range(self.seq_len)], axis=0).astype(np.int32)
        return torch.from_numpy(data_idx), torch.from_numpy(x)

class FramesWindowIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, frames_list, seq_len: int, sliding_step: int):
        super().__init__()
        self.frames_list = frames_list
        self.seq_len = seq_len
        self.sliding_step = sliding_step

    def _process_frame(self, f_bgr: np.ndarray) -> np.ndarray:
        if f_bgr.shape[0] != HEIGHT or f_bgr.shape[1] != WIDTH:
            f_bgr = cv2.resize(f_bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        f_rgb = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
        chw = np.transpose(f_rgb, (2, 0, 1)).astype(np.float32) / 255.0
        return chw

    def __iter__(self):
        total = len(self.frames_list)
        if total < self.seq_len:
            return

        # shard indices among workers
        worker_info = torch.utils.data.get_worker_info()
        starts = list(range(0, total - self.seq_len + 1, self.sliding_step))
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            # contiguous split
            per_worker = (len(starts) + num_workers - 1) // num_workers
            s = worker_id * per_worker
            e = min(s + per_worker, len(starts))
            starts = starts[s:e]

        for start in starts:
            end = start + self.seq_len
            window = [self._process_frame(self.frames_list[fi]) for fi in range(start, end)]
            x = np.stack(window, axis=0).reshape(self.seq_len * 3, HEIGHT, WIDTH)
            data_idx = np.stack([(0, start + f) for f in range(self.seq_len)], axis=0).astype(np.int32)
            yield torch.from_numpy(data_idx), torch.from_numpy(x)


def pred_main(frame_list, fps, w, h, tracknet_file = "ckpts/TrackNet_best.pt", inpaintnet_file = None, batch_size = 1, eval_mode = "weight", output_video = True, traj_len = 8, large_video = False):

    # Record start time
    start_time = time.time()
    
    # Perf knobs
    device = _get_device()
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

    cpu_count = os.cpu_count() or 4
    # Windows 避免 DataLoader 进程复制大量帧导致首批卡顿
    if platform.system() == 'Windows':
        num_workers = 0
    else:
        num_workers = min(max(1, cpu_count - 1), 8)
    # Heuristic batch size if not provided or too small
    if batch_size is None or batch_size <= 1:
        batch_size = 8 if device.type == 'cuda' else 2
    # out_csv_file = os.path.join(save_dir, f'{video_name}_ball.csv')
    # out_video_file_cap = os.path.join(save_dir, f'{video_name}.mp4')

    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    
    # Device and cached models
    print(f"Using device: {device}")
    model_pack = get_models(tracknet_file, inpaintnet_file, device)
    tracknet = model_pack['tracknet']
    tracknet_seq_len = model_pack['tracknet_seq_len']
    bg_mode = model_pack['bg_mode']
    inpaintnet = model_pack['inpaintnet']
    inpaintnet_seq_len = model_pack['inpaintnet_seq_len']

    # Sample all frames from video
    # frame_list, fps, (w, h) = generate_frames_from_cap(video_file_cap)
    # print(frame_list)
    print("FPS: ",fps)

    # if (fps != 30):
    #     print("Converting video to 30 fps")
    #     change_fps(video_file_cap)


    w_scaler, h_scaler = w / WIDTH, h / HEIGHT
    img_scaler = (w_scaler, h_scaler)
    print(f'Number of sampled frames: {len(frame_list)}')

    tracknet_pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Inpaint_Mask':[],
                        'Img_scaler': (w_scaler, h_scaler), 'Img_shape': (w, h)}

    # Test on TrackNet
    seq_len = tracknet_seq_len

    # Lazy preprocessing dataset moved to top-level class (Windows pickling safe)
    # Build frames_chw once
    frames_chw = np.empty((len(frame_list), 3, HEIGHT, WIDTH), dtype=np.float32)
    for idx, f in enumerate(frame_list):
        f = cv2.resize(f, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames_chw[idx] = np.transpose(f, (2, 0, 1)) / 255.0

    # Background median for concat
    median_chw = None
    if bg_mode == 'concat' and len(frame_list) > 0:
        sample_step = max(1, len(frame_list) // 50)
        sampled = frames_chw[::sample_step]  # (S,3,H,W)
        median = np.median(sampled, axis=0)  # (3,H,W)
        median_chw = median.astype(np.float32)

    if eval_mode == 'nonoverlap':
        # Create dataset with non-overlap sampling (lazy)
        dataset = SimpleFramesDataset(frames_chw, seq_len=seq_len, sliding_step=seq_len, bg_mode=bg_mode, median_chw=median_chw)
        dl_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(num_workers > 0),
        )
        if num_workers > 0:
            dl_kwargs['prefetch_factor'] = 2
        data_loader = DataLoader(dataset, **dl_kwargs)

        amp_enabled = (device.type == 'cuda')
        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().to(device, non_blocking=True)
            x = x.to(memory_format=torch.channels_last)
            with torch.inference_mode():
                if amp_enabled:
                    with torch.amp.autocast('cuda'):
                        y_pred = tracknet(x)
                else:
                    y_pred = tracknet(x)
                y_pred = y_pred.detach().cpu()
            
            # Predict
            tmp_pred = predict(i, y_pred=y_pred, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])
    else:


        if large_video:
            # dataset = Video_IterableDataset(video_file_cap, seq_len=seq_len, sliding_step=1, bg_mode=bg_mode,
            #                                 max_sample_num=1000, video_range=None)
            # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
            # video_len = dataset.video_len
            # print(f'Video length: {video_len}')
            pass
            
        else:
            # Sample all frames from video
            dataset = SimpleFramesDataset(frames_chw, seq_len=seq_len, sliding_step=1, bg_mode=bg_mode, median_chw=median_chw)
            dl_kwargs = dict(
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=(device.type == 'cuda'),
                persistent_workers=(num_workers > 0),
            )
            if num_workers > 0:
                dl_kwargs['prefetch_factor'] = 2
            data_loader = DataLoader(dataset, **dl_kwargs)
            video_len = len(frame_list)
        
        # Init prediction buffer params
        num_sample, sample_count = video_len-seq_len+1, 0
        buffer_size = seq_len - 1
        batch_i = torch.arange(seq_len) # [0, 1, 2, 3, 4, 5, 6, 7]
        frame_i = torch.arange(seq_len-1, -1, -1) # [7, 6, 5, 4, 3, 2, 1, 0]
        y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
        weight = get_ensemble_weight(seq_len, eval_mode)
        amp_enabled = (device.type == 'cuda')
        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().to(device, non_blocking=True)
            x = x.to(memory_format=torch.channels_last)
            b_size, seq_len = i.shape[0], i.shape[1]
            with torch.inference_mode():
                if amp_enabled:
                    with torch.amp.autocast('cuda'):
                        y_pred = tracknet(x)
                else:
                    y_pred = tracknet(x)
                y_pred = y_pred.detach().cpu()
            
            y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
            ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
            ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)
            for b in range(b_size):
                if sample_count < buffer_size:
                    # Imcomplete buffer
                    y_pred = y_pred_buffer[batch_i+b, frame_i].sum(0)
                    y_pred /= (sample_count+1)
                else:
                    # General case
                    y_pred = (y_pred_buffer[batch_i+b, frame_i] * weight[:, None, None]).sum(0)
                
                ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                sample_count += 1

                if sample_count == num_sample:
                    # Last batch
                    y_zero_pad = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                    y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)

                    for f in range(1, seq_len):
                        # Last input sequence
                        y_pred = y_pred_buffer[batch_i+b+f, frame_i].sum(0)
                        y_pred /= (seq_len-f)
                        ensemble_i = torch.cat((ensemble_i, i[-1][f].reshape(1, 1, 2)), dim=0)
                        ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)

            # Predict
            tmp_pred = predict(ensemble_i, y_pred=ensemble_y_pred, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])

            # Update buffer, keep last predictions for ensemble in next iteration
            y_pred_buffer = y_pred_buffer[-buffer_size:]

    # Test on TrackNetV3 (TrackNet + InpaintNet)
    if inpaintnet is not None:
        seq_len = inpaintnet_seq_len
        tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(tracknet_pred_dict, th_h=h*0.05)
        inpaint_pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

        if eval_mode == 'nonoverlap':
            # Create dataset with non-overlap sampling
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='coordinate', pred_dict=tracknet_pred_dict, padding=True)
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=(device.type == 'cuda'),
                persistent_workers=(num_workers > 0),
            )

            amp_enabled = (device.type == 'cuda')
            for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
                coor_pred = coor_pred.float().to(device, non_blocking=True)
                inpaint_mask = inpaint_mask.float().to(device, non_blocking=True)
                with torch.inference_mode():
                    if amp_enabled:
                        with torch.cuda.amp.autocast():
                            coor_inpaint = inpaintnet(coor_pred, inpaint_mask)
                    else:
                        coor_inpaint = inpaintnet(coor_pred, inpaint_mask)
                    coor_inpaint = coor_inpaint.detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask) # replace predicted coordinates with inpainted coordinates
                
                # Thresholding
                th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                coor_inpaint[th_mask] = 0.
                
                # Predict
                tmp_pred = predict(i, c_pred=coor_inpaint, img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])
                
        else:
            # Create dataset with overlap sampling for temporal ensemble
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1, data_mode='coordinate', pred_dict=tracknet_pred_dict)
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=(device.type == 'cuda'),
                persistent_workers=(num_workers > 0),
            )
            weight = get_ensemble_weight(seq_len, eval_mode)

            # Init buffer params
            num_sample, sample_count = len(dataset), 0
            buffer_size = seq_len - 1
            batch_i = torch.arange(seq_len) # [0, 1, 2, 3, 4, 5, 6, 7]
            frame_i = torch.arange(seq_len-1, -1, -1) # [7, 6, 5, 4, 3, 2, 1, 0]
            coor_inpaint_buffer = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
            
            amp_enabled = (device.type == 'cuda')
            for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
                coor_pred = coor_pred.float().to(device, non_blocking=True)
                inpaint_mask = inpaint_mask.float().to(device, non_blocking=True)
                b_size = i.shape[0]
                with torch.inference_mode():
                    if amp_enabled:
                        with torch.cuda.amp.autocast():
                            coor_inpaint = inpaintnet(coor_pred, inpaint_mask)
                    else:
                        coor_inpaint = inpaintnet(coor_pred, inpaint_mask)
                    coor_inpaint = coor_inpaint.detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask)
                
                # Thresholding
                th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                coor_inpaint[th_mask] = 0.

                coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_inpaint), dim=0)
                ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
                ensemble_coor_inpaint = torch.empty((0, 1, 2), dtype=torch.float32)
                
                for b in range(b_size):
                    if sample_count < buffer_size:
                        # Imcomplete buffer
                        coor_inpaint = coor_inpaint_buffer[batch_i+b, frame_i].sum(0)
                        coor_inpaint /= (sample_count+1)
                    else:
                        # General case
                        coor_inpaint = (coor_inpaint_buffer[batch_i+b, frame_i] * weight[:, None]).sum(0)
                    
                    ensemble_i = torch.cat((ensemble_i, i[b][0].view(1, 1, 2)), dim=0)
                    ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)
                    sample_count += 1

                    if sample_count == num_sample:
                        # Last input sequence
                        coor_zero_pad = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
                        coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0)
                        
                        for f in range(1, seq_len):
                            coor_inpaint = coor_inpaint_buffer[batch_i+b+f, frame_i].sum(0)
                            coor_inpaint /= (seq_len-f)
                            ensemble_i = torch.cat((ensemble_i, i[-1][f].view(1, 1, 2)), dim=0)
                            ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)

                # Thresholding
                th_mask = ((ensemble_coor_inpaint[:, :, 0] < COOR_TH) & (ensemble_coor_inpaint[:, :, 1] < COOR_TH))
                ensemble_coor_inpaint[th_mask] = 0.

                # Predict
                tmp_pred = predict(ensemble_i, c_pred=ensemble_coor_inpaint, img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])
                
                # Update buffer, keep last predictions for ensemble in next iteration
                coor_inpaint_buffer = coor_inpaint_buffer[-buffer_size:]
        

    # Write csv file
    pred_dict = inpaint_pred_dict if inpaintnet is not None else tracknet_pred_dict
    # write_pred_csv(pred_dict, save_file=out_csv_file)

    # # print(frame_list, pred_dict, out_video_file_cap)
    # # file1 = open('bin_subclips.bin','ab')

    # with open(f'predicted.bin','wb') as file:
    #     pickle.dump(pred_dict, file)
    #     pickle.dump(out_video_file, file)
    #     pickle.dump(video_file, file)
    
    # Record end time and calculate duration
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Model prediction completed in {processing_time:.2f} seconds")

    return pred_dict