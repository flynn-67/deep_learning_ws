## 사용 방법 요약

1. 이 GitHub 저장소에서 `demo.py`, `go2_physics_teleop.py` 두 파일을 다운로드합니다.
2. 기존 컴퓨터의 `~/dl/IAmGoodNavigator` 폴더에 두 파일을 넣고 기존 파일을 덮어씌웁니다.
3. 터미널에서 아래 명령어를 실행합니다.

```bash
cd ~/dl/IAmGoodNavigator

#(가상환경 들어가기)
conda activate goodnav 

#현재 사용자 계정에 따로 설치된 Python 패키지들을 사용하지 않도록 막는 역할을 합니다.
#IsaacLab은 패키지 버전 충돌이 자주 발생할 수 있기 때문에, conda 가상환경 안에 설치된 패키지만 사용하도록 설정하는 것입니다.
export PYTHONNOUSERSITE=1

#시뮬 실행 명령어
python demo.py --agent go2 --task fine --index 1 --work_dir ./myresults
