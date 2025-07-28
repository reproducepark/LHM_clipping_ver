# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Peihao Li
# @Email         : liphao99@gmail.com
# @Time          : 2025-03-19 12:47:58
# @Function      : video motion process pipeline
import copy
import json
import os
import sys
from math import sqrt

current_dir_path = os.path.dirname(__file__)
sys.path.append(current_dir_path + "/../pose_estimation")
import argparse
import copy
import gc
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from blocks import SMPL_Layer
from blocks.detector import DetectionModel
from model import forward_model, load_model
from pose_utils.constants import KEYPOINT_THR
from pose_utils.image import img_center_padding, normalize_rgb_tensor
from pose_utils.inference_utils import get_camera_parameters
from pose_utils.postprocess import OneEuroFilter, smplx_gs_smooth
from pose_utils.render import render_video
from pose_utils.tracker import bbox_xyxy_to_cxcywh, track_by_area
from smplify import TemporalSMPLify

import shutil
import glob
import subprocess

import math


torch.cuda.empty_cache()

np.random.seed(seed=0)
random.seed(0)


def load_video(video_path, pad_ratio, max_resolution):
    frames = []
    for i in range(2):
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"fail to load video file {video_path}"
        fps = cap.get(cv2.CAP_PROP_FPS)
    
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        downsample_factor = -1
        if (height * width) > max_resolution:
            downsample_factor = sqrt(max_resolution / (height * width))
            height = int(height * downsample_factor)
            width = int(width * downsample_factor)

        
        offset_w, offset_h = 0, 0
        while cap.isOpened():
            flag, frame = cap.read()
            if not flag:
                break

            # since the tracker and detector receive BGR images as inputs
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if downsample_factor > 0:
                frame = cv2.resize(
                    frame,
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )
            if pad_ratio > 0:
                frame, offset_w, offset_h = img_center_padding(frame, pad_ratio)
            frames.append(frame)
        height, width, _ = frames[0].shape
    return frames, height, width, fps, offset_w, offset_h


def images_crop(images, bboxes, target_size, device=torch.device("cuda")):
    # bboxes: cx, cy, w, h
    crop_img_list = []
    crop_annotations = []
    i = 0
    raw_img_size = max(images[0].shape[:2])
    for img, bbox in zip(images, bboxes):

        left = max(0, int(bbox[0] - bbox[2] // 2))
        right = min(img.shape[1] - 1, int(bbox[0] + bbox[2] // 2))
        top = max(0, int(bbox[1] - bbox[3] // 2))
        bottom = min(img.shape[0] - 1, int(bbox[1] + bbox[3] // 2))
        crop_img = img[top:bottom, left:right]
        crop_img = torch.Tensor(crop_img).to(device).unsqueeze(0).permute(0, 3, 1, 2)

        _, _, h, w = crop_img.shape
        scale_factor = min(target_size / w, target_size / h)
        crop_img = F.interpolate(crop_img, scale_factor=scale_factor, mode="bilinear")

        _, _, h, w = crop_img.shape
        pad_left = (target_size - w) // 2
        pad_top = (target_size - h) // 2
        pad_right = target_size - w - pad_left
        pad_bottom = target_size - h - pad_top
        crop_img = F.pad(
            crop_img,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0,
        )

        resize_img = normalize_rgb_tensor(crop_img)

        crop_img_list.append(resize_img)
        crop_annotations.append(
            (
                left,
                top,
                pad_left,
                pad_top,
                scale_factor,
                target_size / scale_factor,
                raw_img_size,
            )
        )

    return crop_img_list, crop_annotations


def generate_pseudo_idx(keypoints, patch_size, n_patch, crop_annotation):

    device = keypoints.device
    anchors = torch.stack([keypoints[3], keypoints[4], keypoints[5], keypoints[6]])

    mask = anchors[..., -1] >= KEYPOINT_THR
    if mask.sum() < 2:

        return None, None
    anchors = anchors[mask, :2]  # N, 2

    radius = torch.norm(anchors.max(dim=0)[0] - anchors.min(dim=0)[0]) / 2

    head_pseudo_loc = anchors.mean(0)
    if crop_annotation is not None:
        left, top, pad_left, pad_top, scale_factor, crop_size, raw_size = (
            crop_annotation
        )
        head_pseudo_loc = (
            head_pseudo_loc - torch.tensor([left, top], device=device)
        ) * scale_factor + torch.tensor([pad_left, pad_top], device=device)
        radius = radius * scale_factor
    coarse_loc = (head_pseudo_loc // patch_size).int()  # (nhv,2)
    pseudo_idx = torch.clamp(coarse_loc, 0, n_patch - 1)  # (nhv,2)
    pseudo_idx = (
        torch.zeros((1,), dtype=torch.int32, device=device),
        pseudo_idx[1:2],
        pseudo_idx[0:1],
        torch.zeros((1,), dtype=torch.int32, device=device),
    )
    max_dist = (radius // patch_size).int()
    if max_dist < 2:
        max_dist = None
    return pseudo_idx, max_dist


def project2origin_img(target_human, crop_annotation):
    if target_human is None:
        return target_human
    left, top, pad_left, pad_top, scale_factor, crop_size, raw_size = crop_annotation
    device = target_human["loc"].device

    target_human["loc"] = (
        target_human["loc"] - torch.tensor([pad_left, pad_top], device=device)
    ) / scale_factor + torch.tensor([left, top], device=device)

    target_human["dist"] = target_human["dist"] / (crop_size / raw_size)
    return target_human


def empty_frame_pad(pose_results):
    if len(pose_results) == 1:
        return pose_results
    all_is_None = True
    for i in range(1, len(pose_results)):
        if pose_results[i] is None and pose_results[i - 1] is not None:
            pose_results[i] = copy.deepcopy(pose_results[i - 1])
        if pose_results[i] is not None:
            all_is_None = False
    if all_is_None:
        return []
    for i in range(len(pose_results) - 2, -1, -1):
        if pose_results[i] is None and pose_results[i + 1] is not None:
            pose_results[i] = copy.deepcopy(pose_results[i + 1])
    return pose_results


def parse_chunks(
    frame_ids,
    pose_results,
    k2d,
    bboxes,
    min_len=10,
):
    """If a track disappear in the middle,
    we separate it to different segments
    """
    data_chunks = []
    if isinstance(frame_ids, list):
        frame_ids = np.array(frame_ids)
    step = frame_ids[1:] - frame_ids[:-1]
    step = np.concatenate([[0], step])
    breaks = np.where(step != 1)[0]
    start = 0
    for bk in breaks[1:]:
        f_chunk = frame_ids[start:bk]

        if len(f_chunk) >= min_len:
            data_chunk = {
                "frame_id": f_chunk,
                "keypoints_2d": k2d[start:bk],
                "bbox": bboxes[start:bk],
                "rotvec": [],
                "beta": [],
                "loc": [],
                "dist": [],
            }
            padded_pose_results = empty_frame_pad(pose_results[start:bk])

            for pose_result in padded_pose_results:
                data_chunk["rotvec"].append(pose_result["rotvec"])
                data_chunk["beta"].append(pose_result["shape"])
                data_chunk["loc"].append(pose_result["loc"])
                data_chunk["dist"].append(pose_result["dist"])
            if len(padded_pose_results) > 0:
                data_chunks.append(data_chunk)
        start = bk

    start = breaks[-1]  # last chunk
    bk = len(frame_ids)
    f_chunk = frame_ids[start:bk]

    if len(f_chunk) >= min_len:
        data_chunk = {
            "frame_id": f_chunk,
            "keypoints_2d": k2d[start:bk].clone().detach(),
            "bbox": bboxes[start:bk].clone().detach(),
            "rotvec": [],
            "beta": [],
            "loc": [],
            "dist": [],
        }
        padded_pose_results = empty_frame_pad(pose_results[start:bk])
        for pose_result in padded_pose_results:
            data_chunk["rotvec"].append(pose_result["rotvec"])
            data_chunk["beta"].append(pose_result["shape"])
            data_chunk["loc"].append(pose_result["loc"])
            data_chunk["dist"].append(pose_result["dist"])

        if len(padded_pose_results) > 0:

            data_chunks.append(data_chunk)

    for data_chunk in data_chunks:
        for key in ["rotvec", "beta", "loc", "dist"]:
            try:
                data_chunk[key] = torch.stack(data_chunk[key])
            except:
                print(key)

    return data_chunks


def load_models(model_path, device):
    ckpt_path = os.path.join(model_path, "pose_estimate", "multiHMR_896_L.pt")
    pose_model = load_model(ckpt_path, model_path, device=device)
    print("load hmr")
    pose_model_ckpt = os.path.join(
        model_path, "pose_estimate", "vitpose-h-wholebody.pth"
    )
    keypoint_detector = DetectionModel(pose_model_ckpt, device)
    print("load detection")
    smplx_model = SMPL_Layer(
        model_path,
        type="smplx",
        gender="neutral",
        num_betas=10,
        kid=False,
        person_center="head",
    ).to(device)
    print("load smplx")
    return pose_model, keypoint_detector, smplx_model


class Video2MotionPipeline:
    def __init__(
        self,
        model_path,
        fitting_steps,
        device,
        kp_mode="vitpose",
        visualize=True,
        pad_ratio=0.2,
        fov=60,
    ):
        self.MAX_RESOLUTION = 1280 * 720
        self.device = device
        self.visualize = visualize
        self.kp_mode = kp_mode
        self.pad_ratio = pad_ratio
        self.fov = fov
        self.fps = None
        self.pose_model, self.keypoint_detector, self.smplx_model = load_models(
            model_path, self.device
        )
        self.smplx_model.to(self.device)
        self.smplify = TemporalSMPLify(
            smpl=self.smplx_model, device=self.device, num_steps=fitting_steps
        )

    def track(self, all_frames):
        self.keypoint_detector.initialize_tracking()
        for frame in all_frames:
            self.keypoint_detector.track(frame, self.fps, len(all_frames))
        tracking_results = self.keypoint_detector.process(self.fps)
        # note: only surpport pose estimation for one character
        main_character = None
        max_frame_length = -1
        for _id in tracking_results.keys():
            if len(tracking_results[_id]["frame_id"]) > max_frame_length:
                max_frame_length = len(tracking_results[_id]["frame_id"])
                main_character = _id

        if main_character is None or len(tracking_results[main_character]["bbox"]) == 0:
            print("WARNING: No person detected in this clip. Skipping...")
            return [], [], []  # 빈 값 반환

        bboxes = tracking_results[main_character]["bbox"]
        frame_ids = tracking_results[main_character]["frame_id"]
        frames = [all_frames[i] for i in frame_ids]
        assert not (bboxes[0][0] == 0 and bboxes[0][2] == 0)

        return bboxes, frame_ids, frames

    def detect_keypoint2d(self, bboxes, frames):
        if self.kp_mode == "vitpose":
            keypoints, bboxes = self.keypoint_detector.batch_detection(bboxes, frames)
        else:
            raise NotImplementedError
        return bboxes, keypoints

    def estimate_pose(self, frame_ids, frames, keypoints, bboxes, raw_K, video_length):
        target_img_size = self.pose_model.img_size
        patch_size = self.pose_model.patch_size

        K = get_camera_parameters(
            target_img_size, fov=self.fov, p_x=None, p_y=None, device=self.device
        )

        keypoints = torch.tensor(keypoints, device=self.device)
        bboxes = torch.tensor(bboxes, device=self.device)
        bboxes = bbox_xyxy_to_cxcywh(bboxes, scale=1.5)

        crop_images, crop_annotations = images_crop(
            frames, bboxes, target_size=target_img_size, device=self.device
        )

        all_frame_results = []
        # model inference
        for i, image in enumerate(crop_images):

            # Calculate the possible search area for the primary joint (head) based on 2D keypoints
            # pseudo_idx: The index of the search area center after patching
            # max_dist: The maximum radius of the search area
            pseudo_idx, max_dist = generate_pseudo_idx(
                keypoints[i],
                patch_size,
                int(target_img_size / patch_size),
                crop_annotations[i],
            )
            humans = forward_model(
                self.pose_model, image, K, pseudo_idx=pseudo_idx, max_dist=max_dist
            )
            target_human = track_by_area(humans, target_img_size)
            target_human = project2origin_img(target_human, crop_annotations[i])

            all_frame_results.append(target_human)

        # parse chunk & missed frame padding
        data_chunks = parse_chunks(
            frame_ids,
            all_frame_results,
            keypoints,
            bboxes,
            min_len=int(self.fps / 10),
        )

        trans_cam_fill = np.zeros((video_length, 3))
        smpl_poses_cam_fill = np.zeros((video_length, 55, 3))
        smpl_shapes_fill = np.zeros((video_length, 10))
        all_verts = [None] * video_length
        for data_chunk in data_chunks:
            # one_euro filter on 2d keypoints

            one_euro = OneEuroFilter(
                min_cutoff=1.2, beta=0.3, sampling_rate=self.fps, device=self.device
            )
            for i in range(len(data_chunk["keypoints_2d"])):
                data_chunk["keypoints_2d"][i, :, :2] = one_euro.filter(
                    data_chunk["keypoints_2d"][i, :, :2]
                )

            poses, betas, transl = self.smplify.fit(
                data_chunk["rotvec"],
                data_chunk["beta"],
                data_chunk["dist"],
                data_chunk["loc"],
                raw_K,
                data_chunk["keypoints_2d"],
                data_chunk["bbox"],
            )

            # gaussian filter
            with torch.no_grad():

                poses, betas, transl = smplx_gs_smooth(
                    poses, betas, transl, fps=self.fps
                )

                out = self.smplx_model(
                    poses,
                    betas,
                    None,
                    None,
                    transl=transl,
                    K=raw_K,
                    expression=None,
                    rot6d=False,
                )

                transl = out["transl_pelvis"].squeeze(1)
                poses_ = self.smplx_model.convert_standard_pose(poses)
                smpl_poses_cam_fill[data_chunk["frame_id"]] = poses_.cpu().numpy()
                smpl_shapes_fill[data_chunk["frame_id"]] = betas.cpu().numpy()
                trans_cam_fill[data_chunk["frame_id"]] = transl.cpu().numpy()

            for i, frame_id in enumerate(data_chunk["frame_id"]):
                try:
                    if all_verts[frame_id] is None:
                        all_verts[frame_id] = []
                    all_verts[frame_id].append(out["v3d"][i])
                except:
                    break

        return smpl_poses_cam_fill, smpl_shapes_fill, trans_cam_fill, all_verts

    def save_video(
        self, all_frames, frame_ids, bboxes, keypoints, verts, K, out_folder
    ):
        all_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in all_frames]
        save_name = os.path.join(out_folder, "pose_visualized.mp4")

        # 2d keypoints visualization
        for i, frame_id in enumerate(frame_ids):
            keypoint_results = [{"bbox": bboxes[i], "keypoints": keypoints[i]}]

            all_frames[frame_id] = self.keypoint_detector.visualize(
                all_frames[frame_id], keypoint_results
            )

        render_video(
            verts,
            self.pose_model.smpl_layer["neutral_10"].bm_x.faces,
            K,
            all_frames,
            self.fps,
            save_name,
            self.device,
            True,
        )

    def save_results(self, out_path, frame_ids, poses, betas, transl, K, img_wh):
        K = K[0].cpu().numpy()
        for i in frame_ids:

            smplx_param = {}
            smplx_param["betas"] = betas[i].tolist()
            smplx_param["root_pose"] = poses[i, 0].tolist()
            smplx_param["body_pose"] = poses[i, 1:22].tolist()
            smplx_param["jaw_pose"] = poses[i, 22].tolist()
            smplx_param["leye_pose"] = [0.0, 0.0, 0.0]
            smplx_param["reye_pose"] = [0.0, 0.0, 0.0]
            smplx_param["lhand_pose"] = poses[i, 25:40].tolist()
            smplx_param["rhand_pose"] = poses[i, 40:55].tolist()

            smplx_param["trans"] = transl[i].tolist()
            smplx_param["focal"] = [float(K[0, 0]), float(K[1, 1])]
            smplx_param["princpt"] = [float(K[0, 2]), float(K[1, 2])]
            smplx_param["img_size_wh"] = [img_wh[0], img_wh[1]]
            smplx_param["pad_ratio"] = self.pad_ratio
            with open(os.path.join(out_path, f"{(i+1):05}.json"), "w") as fp:
                json.dump(smplx_param, fp)

    def __call__(self, video_path, output_path, is_file_only=False):
        start = time.time()
        all_frames, raw_H, raw_W, fps, offset_w, offset_h = load_video(
            video_path, pad_ratio=self.pad_ratio, max_resolution=self.MAX_RESOLUTION
        )
        self.fps = fps
        video_length = len(all_frames)

        raw_K = get_camera_parameters(
            max(raw_H, raw_W), fov=self.fov, p_x=None, p_y=None, device=self.device
        )
        raw_K[..., 0, -1] = raw_W / 2
        raw_K[..., 1, -1] = raw_H / 2

        bboxes, frame_ids, frames = self.track(all_frames)
        if len(bboxes) == 0 or len(frame_ids) == 0 or len(frames) == 0:
            print(f"{video_path}: No person found/skipping clip.")
            return None  # 핵심! - 이후 detect_keypoint2d나 estimate_pose 실행 금지
        bboxes, keypoints = self.detect_keypoint2d(bboxes, frames)
        gc.collect()
        torch.cuda.empty_cache()

        poses, betas, transl, verts = self.estimate_pose(
            frame_ids, frames, keypoints, bboxes, raw_K, video_length
        )

        if is_file_only:
            output_folder = output_path
        else:
            output_folder = os.path.join(
                output_path, video_path.split("/")[-1].split(".")[0]
            )
        os.makedirs(output_folder, exist_ok=True)

        if self.visualize:
            self.save_video(
                all_frames, frame_ids, bboxes, keypoints, verts, raw_K, output_folder
            )

        smplx_output_folder = os.path.join(output_folder, "smplx_params")
        os.makedirs(smplx_output_folder, exist_ok=True)
        self.save_results(
            smplx_output_folder, frame_ids, poses, betas, transl, raw_K, (raw_W, raw_H)
        )
        duration = time.time() - start
        print(f"{video_path} processing completed, duration: {duration:.2f}s")

        return smplx_output_folder


def get_parse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./train_data/custom_motion")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./pretrained_models/human_model_files",
        help="model_path",
    )
    parser.add_argument(
        "--pad_ratio",
        type=float,
        default=0.2,
        help="padding images for more accurate estimation results",
    )
    parser.add_argument(
        "--kp_mode",
        type=str,
        default="vitpose",
        help="only ViTPose is supported currently",
    )
    parser.add_argument(
        "--fitting_steps",
        nargs="+",
        type=int,
        default=[30, 50],
        help="Number of iterations for the two-stage fitting in SMPLify",
    )

    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    return args


def split_video_ffmpeg(input_path, temp_dir, clip_sec=10):
    os.makedirs(temp_dir, exist_ok=True)
    out_pattern = os.path.join(temp_dir, 'part_%03d.mp4')
    command = [
        'ffmpeg', '-i', input_path, '-c', 'copy',
        '-map', '0', '-segment_time', str(clip_sec),
        '-f', 'segment', out_pattern, '-y'
    ]
    subprocess.run(command, check=True)

def split_video_ffmpeg_precise(input_path, temp_dir, clip_sec=10):
    os.makedirs(temp_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    n_clip = math.ceil(duration / clip_sec)
    cap.release()
    for idx in range(n_clip):
        ss = idx * clip_sec
        out = os.path.join(temp_dir, f'part_{idx:03d}.mp4')
        cmd = [
            'ffmpeg', '-y', '-ss', str(ss), '-i', input_path, '-t', str(clip_sec),
            '-c', 'copy', out
        ]
        subprocess.run(cmd, check=True)

def merge_jsons_to_one_folder(all_clip_json_dirs, final_dir):
    os.makedirs(final_dir, exist_ok=True)
    index = 1
    for part in sorted(all_clip_json_dirs):
        files = sorted(glob.glob(os.path.join(part, '*.json')))
        for f in files:
            new_fname = f"{index:05}.json"
            shutil.copy(f, os.path.join(final_dir, new_fname))
            index += 1

def merge_videos_ffmpeg(clip_video_paths, out_path):
    list_txt = os.path.join(os.path.dirname(out_path), 'merge_list.txt')
    with open(list_txt, 'w') as f:
        for v in clip_video_paths:
            f.write(f"file '{os.path.abspath(v)}'\n")
    command = [
        'ffmpeg', '-f', 'concat', '-safe', '0', '-i',
        list_txt, '-c', 'copy', out_path, '-y'
    ]
    subprocess.run(command, check=True)

def video2motion_full_auto(
        input_video, output_base, model_path, fitting_steps,
        pad_ratio=0.2, kp_mode="vitpose", visualize=True,
        fov=60, clip_sec=10):

    # 1. 영상제목 폴더 자동생성
    video_stem = os.path.splitext(os.path.basename(input_video))[0]
    target_base = os.path.join(output_base, video_stem)
    os.makedirs(target_base, exist_ok=True)

    temp_dir = os.path.join(target_base, '__temp_clips')
    split_video_ffmpeg_precise(input_video, temp_dir, clip_sec=clip_sec)
    video_parts = sorted(glob.glob(os.path.join(temp_dir, 'part_*.mp4')))
    print("병합 순서:", video_parts)
    clip_results = []
    vis_videos = []
    device = torch.device("cuda:0")

    for idx, part in enumerate(video_parts):
        out_part = os.path.join(target_base, f'clip_{idx:03d}')
        os.makedirs(out_part, exist_ok=True)
        pipeline = Video2MotionPipeline(
            model_path, fitting_steps, device,
            kp_mode=kp_mode, visualize=visualize,
            pad_ratio=pad_ratio, fov=fov,
        )
        smplx_folder = pipeline(part, out_part, is_file_only=True)
        if smplx_folder is None:
            continue
        clip_results.append(smplx_folder)
        vis_vid = os.path.join(out_part, 'pose_visualized.mp4')
        if os.path.exists(vis_vid):
            vis_videos.append(vis_vid)

    # 2. 병합: target_base/smplx_params, target_base/pose_visualized.mp4
    final_json_dir = os.path.join(target_base, 'smplx_params')
    merge_jsons_to_one_folder(clip_results, final_json_dir)
    final_video_path = os.path.join(target_base, 'pose_visualized.mp4')
    if vis_videos:
        merge_videos_ffmpeg(vis_videos, final_video_path)
    shutil.rmtree(temp_dir, ignore_errors=True)

    for out_part in glob.glob(os.path.join(target_base, 'clip_*')):
        shutil.rmtree(out_part, ignore_errors=True)

    print(f"처리 및 병합 완료! 결과: {final_json_dir}, {final_video_path}")

# 기존 모든 코드(클래스와 함수)는 동일하게 둔 채,
# 최종 진입점만 아래처럼 교체

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./train_data/custom_motion")
    parser.add_argument("--model_path", type=str, default="./pretrained_models/human_model_files")
    parser.add_argument("--pad_ratio", type=float, default=0.2)
    parser.add_argument("--kp_mode", type=str, default="vitpose")
    parser.add_argument("--fitting_steps", nargs="+", type=int, default=[30,50])
    parser.add_argument("--clip_sec", type=int, default=10)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    assert torch.cuda.is_available(), "CUDA is not available, please check your environment"
    assert os.path.exists(args.video_path), "The video does not exist"

    video2motion_full_auto(
        args.video_path, args.output_path, args.model_path,
        args.fitting_steps, pad_ratio=args.pad_ratio,
        kp_mode=args.kp_mode, visualize=args.visualize,
        fov=60, clip_sec=args.clip_sec
    )
