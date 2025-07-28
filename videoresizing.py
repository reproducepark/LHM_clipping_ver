import cv2

def crop_and_resize_video(input_path, output_path, start_frame, end_frame, out_w, out_h):
    # 원본 영상 열기
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Cannot open file:", input_path)
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx > end_frame:
            break
        if frame_idx >= start_frame:
            resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            out.write(resized)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved cropped/downsized video as {output_path}")

# 사용 예시 (프레임 300~1200, 해상도 256x144로 변환)
crop_and_resize_video(
    "myvideos/attention.mp4",
    "myvideos/attention_re.mp4",
    300, 1200,
    256, 144
)
