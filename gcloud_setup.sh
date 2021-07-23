git clone https://github.com/MaekTec/dl-lab-project.git
cd dl-lab-project/
git checkout --track origin/markus
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 test_imports.py
chmod +x run_all.sh
./run_all.sh &> run_all_log.txt &
disown -h # or -a
# 30536
# 2055
zip -r dl-lab-project-glcoud.zip dl-lab-project