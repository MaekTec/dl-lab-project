git clone https://github.com/MaekTec/dl-lab-project.git
cd dl-lab-project/
git checkout --track origin/markus
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 test_imports.py
