#Python -m venv venv pytorch=1.3.0 torchvision python=3.7 -y
#pip uninstall Pillow

.\\venv\\Scripts\\activate

pip install pandas
pip install tensorflow
pip install keras
pip install requests
pip install opencv-contrib-python

read name
echo "Hello $name"