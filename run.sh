source my_python3/bin/activate
clear

#python -m manage.config
#python -m manage.src.preprocess
#python -m manage.src.my_model

#=======================================================================

python -m manage.src.train
cp manage/checkpoint/model.h5 manage/output/

#========================================================================

python -m manage.src.test
./manage/output/rename_file.sh
./manage/submission/rename_file.sh


#kill -9 PID
