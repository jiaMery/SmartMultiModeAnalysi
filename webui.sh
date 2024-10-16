# Update
sudo apt update -y
sudo apt-get install ffmpeg -y
# Install Python3 and Pip
sudo apt install python3.12 -y
sudo apt install python3.12-venv -y
# Create python virtual environment
python3 -m venv env
source env/bin/activate
#Install the required packages
pip install -r requirements.txt
#start webui
python webui2.py