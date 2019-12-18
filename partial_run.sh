source my_python3/bin/activate
#clear

#python -m manage.config
#python -m manage.src.preprocess

#============================= model ==========================================
#python -m manage.src.my_model
#python -m manage.src.test_model

python -m manage.src.my_model_baseline
python -m manage.src.my_model_90_des_861
python -m manage.src.my_model_90_des-x_873
python -m manage.src.my_model_marge_net
python -m manage.src.my_model_x_full_lrelu
#============================= train ==========================================

#python -m manage.src.train
#cp manage/checkpoint/model.h5 manage/output/
#./manage/output/rename_file.sh

#============================= test ============================================

#python -m manage.src.test
#python -m manage.src.test_mode
#./manage/submission/rename_file.sh

#============================= optional ============================================
#python -m manage.src.marge_submissions
#python -m manage.src.marge_models

