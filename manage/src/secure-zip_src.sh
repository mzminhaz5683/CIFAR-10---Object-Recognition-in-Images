cd ./manage/src
echo ">>>>>>>>>>>>>>>>>>> passwd of src >>>>>>>>>>>>>>>>>>>"
echo " >>> Enter name of zip : "
read x
zip -P 0000 -r $x.zip *.py
mv *.zip ../zip_src
cd ..
cd ..
