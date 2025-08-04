import subprocess
import os
import sys

def check_ffmpeg_available():
    """ffmpeg와 ffprobe가 사용 가능한지 확인"""
    try:
        # ffmpeg 확인
        result = subprocess.run(['ffmpeg', '-version'], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              text=True, shell=True)
        if result.returncode != 0:
            return False, "ffmpeg"
        
        # ffprobe 확인
        result = subprocess.run(['ffprobe', '-version'], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              text=True, shell=True)
        if result.returncode != 0:
            return False, "ffprobe"
        
        return True, None
    except FileNotFoundError:
        return False, "ffmpeg/ffprobe"

def find_ffmpeg_path():
    """ffmpeg 설치 경로 찾기"""
    possible_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        os.path.join(os.getcwd(), "ffmpeg", "bin", "ffmpeg.exe"),
        os.path.join(os.getcwd(), "ffmpeg.exe")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def composite_background_with_foreground(
    background_path, foreground_path, output_path,
    color_hex='0xFFFFFF', similarity='0.01', blend='0.02'
):
    # ffmpeg 사용 가능성 확인
    is_available, missing_tool = check_ffmpeg_available()
    
    if not is_available:
        print(f"오류: {missing_tool}을 찾을 수 없습니다.")
        print("해결 방법:")
        print("1. ffmpeg를 다운로드하여 설치하세요: https://ffmpeg.org/download.html")
        print("2. 또는 ffmpeg 폴더를 현재 디렉토리에 압축 해제하세요")
        print("3. 또는 ffmpeg를 시스템 PATH에 추가하세요")
        
        # ffmpeg 경로 찾기 시도
        ffmpeg_path = find_ffmpeg_path()
        if ffmpeg_path:
            print(f"발견된 ffmpeg 경로: {ffmpeg_path}")
            print("이 경로를 사용하려면 코드를 수정해야 합니다.")
        return False

    # 1. foreground 영상 해상도 자동감지
    cmd_probe = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=s=x:p=0',
        foreground_path
    ]
    
    # 윈도우에서는 shell=True 추가
    result = subprocess.run(cmd_probe, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                          text=True, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 오류: {result.stderr}")
    
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
    
    # 윈도우에서는 shell=True 추가
    process = subprocess.run(cmd_ffmpeg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                           text=True, shell=True)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg 오류: {process.stderr}")
    
    print(f"출력 파일이 저장되었습니다: {output_path}")
    return True

# 사용 예시
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("사용법: python merge_background.py <background.jpg> <foreground.mp4> <output.mp4>")
        print("예시: python merge_background.py background.jpg video.mp4 output.mp4")
        sys.exit(1)
    
    background = sys.argv[1]
    foreground = sys.argv[2]
    output = sys.argv[3]
    
    # 파일 존재 확인
    if not os.path.exists(background):
        print(f"오류: 배경 파일을 찾을 수 없습니다: {background}")
        sys.exit(1)
    
    if not os.path.exists(foreground):
        print(f"오류: 전경 파일을 찾을 수 없습니다: {foreground}")
        sys.exit(1)
    
    success = composite_background_with_foreground(background, foreground, output)
    if not success:
        sys.exit(1)
