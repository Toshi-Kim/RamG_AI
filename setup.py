import sys
from cx_Freeze import setup, Executable
sys.setrecursionlimit(5000)
# 추가 패키지가 있는 경우, build_exe_options에 추가
build_exe_options = {
    "includes": ["scipy","lap","pygame","cython_bbox","PIL", "cv2", "onnxruntime", "collections","datetime", "sys","threading","numpy","requests"],
    "include_files": ["lib/best640_640.onnx",
                      "lib/warningsound.wav",
                      "lib/NanumGothic-Bold.ttf"]}

# Windows에서 콘솔 창을 숨기려면 'base'를 설정합니다.
base = None
if sys.platform == "win32":
    base = "Win32GUI"

# setup 함수 호출
setup(
    name="Video Predictor",
    version="1.0",
    description="Detection 기반 이미지 예측 애플리케이션",
    options={"build_exe": build_exe_options},
    executables=[Executable("animal_pred.py", base=base)],  # your_tkinter_script.py를 실제 파일 이름으로 바꾸세요
)
