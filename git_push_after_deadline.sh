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
