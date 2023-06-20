GIT_SSH_COMMAND='ssh -i /root/.ssh/id_rsa -o IdentitiesOnly=yes' git pull
export DISPLAY=:1
X :1 -config /root/xorg.conf &
echo 'waiting for X to start..'
sleep 1
x11vnc -display :1 -forever -rfbport 5901 -localhost &
cd /root/noVNC
sleep 3
./utils/novnc_proxy --vnc 0.0.0.0:5901 --listen 0.0.0.0:5001 &
cd /root/Dreamfuser/frontend
mkdir /root/Dreamfuser/frontend/data
mkdir /root/Dreamfuser/frontend/flaskr/static
flask --app flaskr init-db
DISPALY=:1 flask --app ./flaskr --debug run --host 0.0.0.0
