cd ~/dl/IAmGoodNavigator

#가상환경 들어가기
conda activate goodnav

export PYTHONNOUSERSITE=1

python demo.py --agent go2 --task fine --index 1 --work_dir ./myresults
