# PiCamNN
Survelliance system with deep learning based people detection (<a href="https://www.github.com/allanzelener/YAD2K">YAD2K</a> <a href="https://pjreddie.com/darknet/yolo/">YOLO</a> implementation ), and notification with Telegram.
<p>The program is made of two different threads, one is always looking for movements and if there are some it's also writing the frames to video-file.The other thread get the frames in which were detected movements and then with a deep neural network (<a href="https://arxiv.org/abs/1612.08242#">YOLO</a>) is searching for persons, and if there are some it's sending the images to you with telegram.</p>
<p>The code has been tested on raspberry pi 3B with whom I got 2/3s per frame (if you run it on GPU you should reach 200frames per second), with previous versions of the raspberry pi you probably will not get good perormances :'( .</p>
<hr></hr>

<h1>Requirements:</h1>
  -Linux (tested on raspberry pi3 with raspbian)<br />
  -Python3<br/>
  -<a href="http://opencv.org/">OpenCV for Python3</a><br />
  -<a href="https://www.apache.org/">Apache2 http-server</a> (remember to enable the apache process eg "#systemctl enable apache2;reboot")<br />
  -<a href="https://www.tensorflow.org">Tensorflow</a> (<a href="https://github.com/samjabrahams/tensorflow-on-raspberry-pi"> link</a> For Tensorflow on raspberry Pi)<br />
  -<a href="https://github.com/vysheng/tg">telegram-cli</a>(follow the installation instructions and log in with your account)</br> 
  -<a href="http://www.numpy.org/">Numpy</a><br />
  -<a href="https://www.keras.io">Keras 2</a> <br />
  
  <h1>Instructions for <a href="https://www.raspberrypi.org/downloads/raspbian/">Raspbian</a>:</h1>
  
  Follow those instructions after having installed all the requirements!
 <br/> <code>git clone https://github.com/PiSimo/PiCamNN.git</code>
 <br/> <code>cd PiCamNN</code>
 <br/> <code>mkdir imgs</code>
 <p>Download the tiny yolo weights(for keras 2) converted with <a href="https://www.github.com/allanzelener/YAD2K">YAD2k</a> :</p>
  <code>wget https://www.dropbox.com/s/xastcd4c0dv2kty/tiny.h5?dl=0 -O tiny.h5</code>
 <br/> <code>sudo mv index.html /var/www/html/</code><br />
 (NOTE: If you aren't on raspbian apache's base folder might not be /var/www/html/ so check before!)<br /><br />
 <p>Before starting the main script you should change in picam.py some variables:</p>
 <p><code>maxDays = 7</code> If you have stored more then maxDays videos on your devices the oldest one will be removed</p>
 <p><code>baseFolder = "/var/www/html/" </code> Change this variable if your apache hasn't created that folder </p>
 <p><code>scriptFolder = "/home/pi/PiCamNN/"</code> Change this variable with the path which contains the scripts and the weights </p>
 <p><code>num_cam = -1</code> Number of cam to use (-1 means open the first one the system has read)</p>
 <p><code>frame_check = 17 </code> Number of empty frames to wait before killing the main process</p>
 <p><code>time_chunck = 15 </code> Seconds to wait before considering a new action</p>
 <p><code>telegram_user = ""</code> Your Telegram username you will se all the images on the chat with yourself</p>

<br />
<h1>To run the code :</h1>
<p><code>sudo python3 picam.py</code></p>
<p>After the main loop is started,every time a person get detected by the neural net you will receive the photo on <I>telegram</I> (on the chat with yourself).</p>
<p>To see the recorded videos, from your local network you have to go with your browser on the ip of your device which is running <i>PiCamNN</i> and from that page you will be able to download all the videos (eg. http://192.168.0.17 ).</p>

