# PiCam
Survelliance system with deep learning based pedestrain detection (<a href="https://www.github.com/allanzelener/YAD2K">YAD2K</a> <a href="https://pjreddie.com/darknet/yolo/">YOLO</a> implementation ), and notification with Telegram.
<hr></hr>

Requirements:<br />
  -Linux (tested on raspberry pi3 with raspbian)<br />
  -Python3
  -<a href="http://opencv.org/">OpenCV for Python3</a><br />
  -<a href="https://www.apache.org/">Apache2 http-server</a> (remember to enable the apache process eg "#systemctl enable apache2;reboot")<br />
  -<a href="https://www.tensorflow.org">Tensorflow</a><br />
  -<a href="https://github.com/vysheng/tg.git">telegram-cli</a>(follow the installation instructions and log in with your account)</br> 
  -<a href="http://www.numpy.org/">Numpy</a><br />
  -<a href="https://www.keras.io">Keras</a> <br />
  -<a href="http://www.h5py.org/">h5py</a> <br />
  
  <h1>Instructions for <a href="https://www.raspberrypi.org/downloads/raspbian/">Raspbian</a>:</h1>
  
  Follow those instructions after having installed all the requirements!
 <br/> <code>git clone https://github.com/PiSimo/PiCam.git</code>
 <br/> <code>cd PiCam</code>
 <br/> <code>sudo mv index.html /var/www/html/</code><br />
 (NOTE: If you aren't on raspbian apache's base folder might not be /var/www/html/ so check before!)<br /><br />
 <p>Before starting the main script modify in picam.py some main variables:</p>
 <p><code>maxDays = 7</code> If you have stored more then maxDays videos on your devices the oldest one will be removed</p>
 <p><code>baseFolder = "/var/www/html/" </code> Change this variable if your apache hasn't created that folder </p>
 <p><code>scriptFolder = "/home/pi/PiCam/"</code> Change this variable with the path to the PiCam folder </p>
 <p><code>num_cam = -1</code> Number of cam to use (-1 means open the first one the system has read)</p>
 <p><code>frame_check = 17 </code> Number of empty frames to wait before killing the main process</p>
 <p><code>time_chunck = 15 </code> Seconds to wait before considering a new action</p>
 <p><code>telegram_user = ""</code> Your Telegram username you will se all the images on the chat with yourself</p>

<br />
<p>To run your code :</p>
<p><code>sudo python3 picam.py</code></p>
<p>After the main loop is started,every time a person get detected by the neural net you will recive the photo on <I>telegram</I> (on the chat with yourself).</p>
<p>To see the recorded videos, from your local network you have to go with your browser on the ip of your device which is running <i>PiCam</i> and from that page you will be able to download all the videos (eg. "http://192.168.0.17").</p>
