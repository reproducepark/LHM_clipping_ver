import subprocess

def composite_background_with_foreground(
    background_path, foreground_path, output_path,
    color_hex='0xFFFFFF', similarity='0.01', blend='0.02'
):
    # 1. foreground 영상 해상도 자동감지
    cmd_probe = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=s=x:p=0',
        foreground_path
    ]
    result = subprocess.run(cmd_probe, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")
    width, height = result.stdout.strip().split('x')

    # 2. ffmpeg로 배경크기 맞춤+colorkey+합성 모두 실행
    cmd_ffmpeg = [
        'ffmpeg', '-y',
        '-i', background_path,
        '-i', foreground_path,
        '-filter_complex',
        f"[0:v]scale={width}:{height},pad=ceil(iw/2)*2:ceil(ih/2)*2[bg];"
        f"[1:v]colorkey={color_hex}:{similarity}:{blend}[fg];"
        f"[bg][fg]overlay=0:0",
        '-shortest',
        '-c:v', 'libx264',
        '-c:a', 'copy',
        output_path
    ]
    process = subprocess.run(cmd_ffmpeg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {process.stderr}")
    print(f"Output saved to {output_path}")

# 사용 예시
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print("Usage: python composite_bg.py <background.jpg> <foreground.mp4> <output.mp4>")
        sys.exit(1)
    background = sys.argv[1]
    foreground = sys.argv[2]
    output = sys.argv[3]
    composite_background_with_foreground(background, foreground, output)
