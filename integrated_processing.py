#!/usr/bin/env python3
"""
통합 처리 스크립트: 여러 함수들을 합쳐서 JSON 파일들을 처리합니다.
1. trans의 z값을 0.4로 스케일링
2. trans 값을 이동평균으로 안정화
3. 0값이 감지되면 root_pose는 interpolate, 나머지는 이동평균으로 안정화
"""

import json
import os
import numpy as np
import glob
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

def axis_angle_to_quaternion(axis_angle):
    """Axis-angle을 quaternion으로 변환"""
    angles = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = np.abs(angles) < eps
    
    sin_half_angles_over_angles = np.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        np.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    
    quaternions = np.concatenate([
        np.cos(half_angles), 
        axis_angle * sin_half_angles_over_angles
    ], axis=-1)
    return quaternions

def quaternion_to_axis_angle(quaternions):
    """Quaternion을 axis-angle로 변환"""
    norms = np.linalg.norm(quaternions[..., 1:], axis=-1, keepdims=True)
    half_angles = np.arctan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = np.abs(angles) < eps
    
    sin_half_angles_over_angles = np.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        np.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    
    return quaternions[..., 1:] / sin_half_angles_over_angles

def slerp(q1, q2, t):
    """Spherical Linear Interpolation (SLERP)"""
    # 정규화
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # 내적 계산
    dot = np.dot(q1, q2)
    
    # 짧은 경로 선택
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    # 각도 계산
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    
    if theta < 1e-6:
        return q1
    
    # SLERP 공식
    sin_theta = np.sin(theta)
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    return w1 * q1 + w2 * q2

def detect_zero_frames(data_list, threshold=1e-6):
    """0값 프레임들을 감지합니다."""
    zero_indices = []
    
    for i, data in enumerate(data_list):
        # trans, root_pose, body_pose가 모두 0에 가까운지 확인
        trans_zero = np.allclose(data['trans'], [0, 0, 0], atol=threshold)
        root_pose_zero = np.allclose(data['root_pose'], [0, 0, 0], atol=threshold)
        body_pose_zero = np.allclose(data['body_pose'], np.zeros_like(data['body_pose']), atol=threshold)
        
        # 추가 pose 데이터들도 확인
        jaw_pose_zero = np.allclose(data.get('jaw_pose', [0, 0, 0]), [0, 0, 0], atol=threshold)
        leye_pose_zero = np.allclose(data.get('leye_pose', [0, 0, 0]), [0, 0, 0], atol=threshold)
        reye_pose_zero = np.allclose(data.get('reye_pose', [0, 0, 0]), [0, 0, 0], atol=threshold)
        lhand_pose_zero = np.allclose(data.get('lhand_pose', np.zeros((15, 3))), np.zeros((15, 3)), atol=threshold)
        rhand_pose_zero = np.allclose(data.get('rhand_pose', np.zeros((15, 3))), np.zeros((15, 3)), atol=threshold)
        
        if trans_zero and root_pose_zero and body_pose_zero and jaw_pose_zero and leye_pose_zero and reye_pose_zero and lhand_pose_zero and rhand_pose_zero:
            zero_indices.append(i)
    
    return zero_indices

def interpolate_single_pose(pose_data, zero_indices):
    """단일 3D 회전 벡터를 보간합니다 (root_pose, jaw_pose, leye_pose, reye_pose용)."""
    if not zero_indices:
        return pose_data.copy()
    
    interpolated_data = pose_data.copy()
    
    # 연속된 0값 구간들을 찾습니다
    zero_ranges = []
    start_idx = zero_indices[0]
    prev_idx = zero_indices[0]
    
    for idx in zero_indices[1:]:
        if idx != prev_idx + 1:
            # 연속이 끊어짐
            zero_ranges.append((start_idx, prev_idx))
            start_idx = idx
        prev_idx = idx
    
    # 마지막 구간 추가
    zero_ranges.append((start_idx, prev_idx))
    
    # 각 구간에 대해 보간 수행
    for start, end in zero_ranges:
        # 시작점과 끝점 찾기
        start_pose = None
        end_pose = None
        
        # 시작점 이전의 유효한 pose 찾기
        for i in range(start - 1, -1, -1):
            if i not in zero_indices:
                start_pose = pose_data[i]
                break
        
        # 끝점 이후의 유효한 pose 찾기
        for i in range(end + 1, len(pose_data)):
            if i not in zero_indices:
                end_pose = pose_data[i]
                break
        
        # 보간 수행
        if start_pose is not None and end_pose is not None:
            # Quaternion으로 변환하여 보간
            start_quat = axis_angle_to_quaternion(start_pose.reshape(1, 3))[0]
            end_quat = axis_angle_to_quaternion(end_pose.reshape(1, 3))[0]
            
            for i in range(start, end + 1):
                t = (i - start) / (end - start + 1)
                interpolated_quat = slerp(start_quat, end_quat, t)
                interpolated_data[i] = quaternion_to_axis_angle(interpolated_quat.reshape(1, 4))[0]
        
        elif start_pose is not None:
            # 시작점만 있는 경우
            for i in range(start, end + 1):
                interpolated_data[i] = start_pose
        
        elif end_pose is not None:
            # 끝점만 있는 경우
            for i in range(start, end + 1):
                interpolated_data[i] = end_pose
    
    return interpolated_data

def interpolate_pose_data(pose_data, zero_indices, is_rotation=True):
    """여러 3D 회전 벡터를 보간합니다 (body_pose, lhand_pose, rhand_pose용)."""
    if not zero_indices:
        return pose_data.copy()
    
    interpolated_data = pose_data.copy()
    
    # 연속된 0값 구간들을 찾습니다
    zero_ranges = []
    start_idx = zero_indices[0]
    prev_idx = zero_indices[0]
    
    for idx in zero_indices[1:]:
        if idx != prev_idx + 1:
            # 연속이 끊어짐
            zero_ranges.append((start_idx, prev_idx))
            start_idx = idx
        prev_idx = idx
    
    # 마지막 구간 추가
    zero_ranges.append((start_idx, prev_idx))
    
    # 각 구간에 대해 보간 수행
    for start, end in zero_ranges:
        # 시작점과 끝점 찾기
        start_pose = None
        end_pose = None
        
        # 시작점 이전의 유효한 pose 찾기
        for i in range(start - 1, -1, -1):
            if i not in zero_indices:
                start_pose = pose_data[i]
                break
        
        # 끝점 이후의 유효한 pose 찾기
        for i in range(end + 1, len(pose_data)):
            if i not in zero_indices:
                end_pose = pose_data[i]
                break
        
        # 보간 수행
        if start_pose is not None and end_pose is not None:
            if is_rotation:
                # 회전 데이터는 각 3D 벡터를 개별적으로 Quaternion으로 변환하여 보간
                for i in range(start, end + 1):
                    t = (i - start) / (end - start + 1)
                    interpolated_pose = np.zeros_like(start_pose)
                    
                    # 각 3D 회전 벡터를 개별적으로 보간
                    for j in range(len(start_pose)):
                        start_quat = axis_angle_to_quaternion(start_pose[j:j+1].reshape(1, 3))[0]
                        end_quat = axis_angle_to_quaternion(end_pose[j:j+1].reshape(1, 3))[0]
                        interpolated_quat = slerp(start_quat, end_quat, t)
                        interpolated_pose[j] = quaternion_to_axis_angle(interpolated_quat.reshape(1, 4))[0]
                    
                    interpolated_data[i] = interpolated_pose
            else:
                # 위치 데이터는 선형 보간
                for i in range(start, end + 1):
                    t = (i - start) / (end - start + 1)
                    interpolated_data[i] = (1 - t) * start_pose + t * end_pose
        
        elif start_pose is not None:
            # 시작점만 있는 경우
            for i in range(start, end + 1):
                interpolated_data[i] = start_pose
        
        elif end_pose is not None:
            # 끝점만 있는 경우
            for i in range(start, end + 1):
                interpolated_data[i] = end_pose
    
    return interpolated_data

def interpolate_root_pose(root_poses, zero_indices):
    """0값 프레임들의 root_pose를 보간합니다."""
    return interpolate_pose_data(root_poses, zero_indices, is_rotation=True)

def interpolate_trans_data(trans_data, zero_indices):
    """trans 데이터를 선형 보간합니다."""
    if not zero_indices:
        return trans_data.copy()
    
    interpolated_data = trans_data.copy()
    
    # 연속된 0값 구간들을 찾습니다
    zero_ranges = []
    start_idx = zero_indices[0]
    prev_idx = zero_indices[0]
    
    for idx in zero_indices[1:]:
        if idx != prev_idx + 1:
            # 연속이 끊어짐
            zero_ranges.append((start_idx, prev_idx))
            start_idx = idx
        prev_idx = idx
    
    # 마지막 구간 추가
    zero_ranges.append((start_idx, prev_idx))
    
    # 각 구간에 대해 보간 수행
    for start, end in zero_ranges:
        # 시작점과 끝점 찾기
        start_trans = None
        end_trans = None
        
        # 시작점 이전의 유효한 trans 찾기
        for i in range(start - 1, -1, -1):
            if i not in zero_indices:
                start_trans = trans_data[i]
                break
        
        # 끝점 이후의 유효한 trans 찾기
        for i in range(end + 1, len(trans_data)):
            if i not in zero_indices:
                end_trans = trans_data[i]
                break
        
        # 보간 수행
        if start_trans is not None and end_trans is not None:
            # 선형 보간
            for i in range(start, end + 1):
                t = (i - start) / (end - start + 1)
                interpolated_data[i] = (1 - t) * start_trans + t * end_trans
        
        elif start_trans is not None:
            # 시작점만 있는 경우
            for i in range(start, end + 1):
                interpolated_data[i] = start_trans
        
        elif end_trans is not None:
            # 끝점만 있는 경우
            for i in range(start, end + 1):
                interpolated_data[i] = end_trans
    
    return interpolated_data

def moving_average_smoothing(data, window_size=5):
    """이동평균을 이용한 스무딩"""
    smoothed_data = np.zeros_like(data)
    
    for i in range(len(data)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2 + 1)
        smoothed_data[i] = np.mean(data[start_idx:end_idx], axis=0)
    
    return smoothed_data

def scale_trans_z(trans_data, scale_factor=0.4):
    """trans의 z값을 스케일링합니다."""
    scaled_trans = trans_data.copy()
    scaled_trans[:, 2] = scaled_trans[:, 2] * scale_factor
    return scaled_trans

def stabilize_trans(trans_data, method='moving_avg', strength=0.7):
    """trans 값을 안정화합니다."""
    if method == 'moving_avg':
        window_size = int(5 + strength * 10)  # 5~15
        smoothed_trans = moving_average_smoothing(trans_data, window_size)
    elif method == 'gaussian':
        sigma = strength * 2.0
        smoothed_trans = np.zeros_like(trans_data)
        for dim in range(3):
            smoothed_trans[:, dim] = gaussian_filter1d(trans_data[:, dim], sigma=sigma)
    else:
        raise ValueError(f"지원하지 않는 방법: {method}")
    
    # 원본과 스무딩된 데이터를 혼합
    final_trans = (1 - strength) * trans_data + strength * smoothed_trans
    return final_trans

def stabilize_pose_data(pose_data, method='moving_avg', strength=0.3, window_size=3):
    """회전 데이터를 안정화합니다 (quaternion 기반)."""
    if method == 'moving_avg':
        smoothed_pose = moving_average_smoothing(pose_data, window_size)
    elif method == 'gaussian':
        sigma = strength * 1.0
        smoothed_pose = np.zeros_like(pose_data)
        for dim in range(pose_data.shape[1]):
            smoothed_pose[:, dim] = gaussian_filter1d(pose_data[:, dim], sigma=sigma)
    else:
        raise ValueError(f"지원하지 않는 방법: {method}")
    
    # 원본과 스무딩된 데이터를 혼합
    final_pose = (1 - strength) * pose_data + strength * smoothed_pose
    return final_pose

def stabilize_rotation_data(rotation_data, method='quaternion_smooth', strength=0.3, window_size=3):
    """3D 회전 벡터를 quaternion 기반으로 안정화합니다."""
    if method == 'quaternion_smooth':
        # Axis-angle을 quaternion으로 변환
        quaternions = axis_angle_to_quaternion(rotation_data)
        
        # Quaternion을 안정화
        smoothed_quaternions = moving_average_smoothing(quaternions, window_size)
        
        # 정규화
        norms = np.linalg.norm(smoothed_quaternions, axis=-1, keepdims=True)
        smoothed_quaternions = smoothed_quaternions / norms
        
        # Quaternion을 다시 axis-angle로 변환
        smoothed_rotation = quaternion_to_axis_angle(smoothed_quaternions)
        
        # 원본과 스무딩된 데이터를 혼합
        final_rotation = (1 - strength) * rotation_data + strength * smoothed_rotation
        return final_rotation
    
    elif method == 'moving_avg':
        return stabilize_pose_data(rotation_data, method='moving_avg', strength=strength, window_size=window_size)
    
    else:
        raise ValueError(f"지원하지 않는 방법: {method}")

def stabilize_multi_rotation_data(rotation_data, method='quaternion_smooth', strength=0.3, window_size=3):
    """여러 3D 회전 벡터를 개별적으로 안정화합니다 (body_pose, hand_pose용)."""
    if method == 'quaternion_smooth':
        # 각 3D 회전 벡터를 개별적으로 처리
        num_rotations = rotation_data.shape[1] // 3
        stabilized_data = rotation_data.copy()
        
        for i in range(num_rotations):
            start_idx = i * 3
            end_idx = (i + 1) * 3
            single_rotation = rotation_data[:, start_idx:end_idx]
            
            # 개별 회전을 안정화
            stabilized_single = stabilize_rotation_data(
                single_rotation, 
                method='quaternion_smooth', 
                strength=strength, 
                window_size=window_size
            )
            
            stabilized_data[:, start_idx:end_idx] = stabilized_single
        
        return stabilized_data
    
    elif method == 'moving_avg':
        return stabilize_pose_data(rotation_data, method='moving_avg', strength=strength, window_size=window_size)
    
    else:
        raise ValueError(f"지원하지 않는 방법: {method}")

def process_json_files(input_dir, output_dir, trans_z_scale=0.4, stabilize_strength=0.7, 
                      stabilize_pose=True, pose_window_size=3, pose_strength=0.3):
    """
    JSON 파일들을 통합 처리합니다.
    
    Args:
        input_dir: 입력 디렉토리
        output_dir: 출력 디렉토리
        trans_z_scale: trans z값 스케일링 팩터
        stabilize_strength: trans 안정화 강도
        stabilize_pose: 회전 데이터 안정화 여부
        pose_window_size: 회전 데이터 안정화 윈도우 크기
        pose_strength: 회전 데이터 안정화 강도
    """
    
    # 파일들을 숫자 순서로 정렬
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.json')], 
                   key=lambda x: int(x.split('.')[0]))
    
    if not files:
        print(f"'{input_dir}' 디렉토리에서 JSON 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(files)}개의 JSON 파일을 처리합니다...")
    
    # 모든 데이터 로드
    all_data = []
    for filename in files:
        filepath = os.path.join(input_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.append(data)
    
    # 0값 프레임 감지
    zero_indices = detect_zero_frames(all_data)
    print(f"0값 프레임 {len(zero_indices)}개 감지됨")
    
    # 데이터 추출
    trans_data = np.array([data['trans'] for data in all_data])
    root_poses = np.array([data['root_pose'] for data in all_data])
    body_poses = np.array([data['body_pose'] for data in all_data])
    
    # 추가 pose 데이터들 추출 (존재하지 않으면 기본값 사용)
    jaw_poses = np.array([data.get('jaw_pose', [0, 0, 0]) for data in all_data])
    leye_poses = np.array([data.get('leye_pose', [0, 0, 0]) for data in all_data])
    reye_poses = np.array([data.get('reye_pose', [0, 0, 0]) for data in all_data])
    lhand_poses = np.array([data.get('lhand_pose', np.zeros((15, 3))) for data in all_data])
    rhand_poses = np.array([data.get('rhand_pose', np.zeros((15, 3))) for data in all_data])
    
    # 1. 0값 프레임 처리 (먼저 보간)
    if zero_indices:
        print("1. 0값 프레임 처리 중...")
        # 단일 3D 회전 벡터들 (quaternion 보간)
        interpolated_root_poses = interpolate_single_pose(root_poses, zero_indices)
        interpolated_jaw_poses = interpolate_single_pose(jaw_poses, zero_indices)
        interpolated_leye_poses = interpolate_single_pose(leye_poses, zero_indices)
        interpolated_reye_poses = interpolate_single_pose(reye_poses, zero_indices)
        
        # 여러 3D 회전 벡터들 (quaternion 보간)
        interpolated_body_poses = interpolate_pose_data(body_poses, zero_indices, is_rotation=True)
        interpolated_lhand_poses = interpolate_pose_data(lhand_poses, zero_indices, is_rotation=True)
        interpolated_rhand_poses = interpolate_pose_data(rhand_poses, zero_indices, is_rotation=True)
        
        # trans 데이터 보간 (선형 보간)
        interpolated_trans_data = interpolate_trans_data(trans_data, zero_indices)
        
        # 0값 프레임들에 대해 보간된 값 적용
        for idx in zero_indices:
            root_poses[idx] = interpolated_root_poses[idx]
            body_poses[idx] = interpolated_body_poses[idx]
            jaw_poses[idx] = interpolated_jaw_poses[idx]
            leye_poses[idx] = interpolated_leye_poses[idx]
            reye_poses[idx] = interpolated_reye_poses[idx]
            lhand_poses[idx] = interpolated_lhand_poses[idx]
            rhand_poses[idx] = interpolated_rhand_poses[idx]
            trans_data[idx] = interpolated_trans_data[idx] # trans 데이터도 보간
    else:
        print("1. 0값 프레임 없음 - 모든 데이터 그대로 유지")
    
    # 2. trans z값 스케일링
    print("2. trans z값 스케일링 중...")
    scaled_trans = scale_trans_z(trans_data, trans_z_scale)
    
    # 3. trans 안정화
    print("3. trans 안정화 중...")
    stabilized_trans = stabilize_trans(scaled_trans, method='moving_avg', strength=stabilize_strength)
    
    # 4. 회전 데이터 안정화 (선택적)
    if stabilize_pose:
        print("4. 회전 데이터 안정화 중...")
        # 개별 회전 데이터를 안정화
        stabilized_body_poses = stabilize_multi_rotation_data(body_poses, method='quaternion_smooth', strength=pose_strength, window_size=pose_window_size)
        stabilized_lhand_poses = stabilize_multi_rotation_data(lhand_poses, method='quaternion_smooth', strength=pose_strength, window_size=pose_window_size)
        stabilized_rhand_poses = stabilize_multi_rotation_data(rhand_poses, method='quaternion_smooth', strength=pose_strength, window_size=pose_window_size)
    else:
        print("4. 회전 데이터 안정화 건너뜀")
        stabilized_body_poses = body_poses
        stabilized_lhand_poses = lhand_poses
        stabilized_rhand_poses = rhand_poses
    
    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    
    print("5. 결과 저장 중...")
    for i, (filename, data) in enumerate(zip(files, all_data)):
        # 값 업데이트
        data['trans'] = stabilized_trans[i].tolist() # 최종 안정화된 trans 데이터 사용
        data['root_pose'] = root_poses[i].tolist()
        data['body_pose'] = stabilized_body_poses[i].tolist() # 안정화된 body_pose 사용
        
        # 추가 pose 데이터들 업데이트
        data['jaw_pose'] = jaw_poses[i].tolist()
        data['leye_pose'] = leye_poses[i].tolist()
        data['reye_pose'] = reye_poses[i].tolist()
        data['lhand_pose'] = stabilized_lhand_poses[i].tolist() # 안정화된 lhand_pose 사용
        data['rhand_pose'] = stabilized_rhand_poses[i].tolist() # 안정화된 rhand_pose 사용
        
        # 새로운 파일로 저장
        output_filepath = os.path.join(output_dir, filename)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 처리 완료!")
    print(f"입력 디렉토리: {input_dir}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"trans z 스케일링: {trans_z_scale}")
    print(f"trans 안정화 강도: {stabilize_strength}")
    print(f"회전 데이터 안정화: {'활성화' if stabilize_pose else '비활성화'}")
    if stabilize_pose:
        print(f"  - 윈도우 크기: {pose_window_size}")
        print(f"  - 안정화 강도: {pose_strength}")
    print(f"0값 프레임 처리: {len(zero_indices)}개")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='JSON 파일들을 통합 처리합니다.')
    parser.add_argument('input_dir', nargs='?', default='datas/test_video', 
                       help='입력 디렉토리 경로 (기본값: datas/test_video)')
    parser.add_argument('output_dir', nargs='?', default='datas/processed_output', 
                       help='출력 디렉토리 경로 (기본값: datas/processed_output)')
    parser.add_argument('--trans-z-scale', '-z', type=float, default=0.4,
                       help='trans z값 스케일링 팩터 (기본값: 0.4)')
    parser.add_argument('--stabilize-strength', '-s', type=float, default=0.7,
                       help='trans 안정화 강도 (기본값: 0.7)')
    parser.add_argument('--stabilize-pose', action='store_true', default=True,
                       help='회전 데이터 안정화 여부 (기본값: True)')
    parser.add_argument('--no-stabilize-pose', dest='stabilize_pose', action='store_false',
                       help='회전 데이터 안정화 비활성화')
    parser.add_argument('--pose-window-size', type=int, default=3,
                       help='회전 데이터 안정화 윈도우 크기 (기본값: 3)')
    parser.add_argument('--pose-strength', type=float, default=0.3,
                       help='회전 데이터 안정화 강도 (기본값: 0.3)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: 입력 디렉토리 '{args.input_dir}'가 존재하지 않습니다.")
        return
    
    # 처리 실행
    process_json_files(
        args.input_dir, 
        args.output_dir, 
        args.trans_z_scale, 
        args.stabilize_strength,
        args.stabilize_pose,
        args.pose_window_size,
        args.pose_strength
    )

if __name__ == "__main__":
    main() 