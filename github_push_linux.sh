#!/bin/bash
echo "===================>>> secure git - passwd hint: she <<<==================="
./manage/submission/secure-zip_submission.sh
./manage/dataset/secure-zip_dataset.sh
./manage/output/secure-zip_output.sh
./manage/src/secure-zip_src.sh


git status
git add *
echo ""
echo "===================>>> *.pyc will be removed <<<==================="
git rm *.pyc -f
echo "===================>>> *.pyc removed <<<==================="
echo ""
git add -u
git status

echo ""
echo "==================>>> Enter the commit <<<==================="
read x
git commit -m "$x"

git push origin master
