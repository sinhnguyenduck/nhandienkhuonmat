import os
from unittest import result
import playsound
import speech_recognition as sr
import time
import sys
import ctypes
import wikipedia
import datetime
import json
import webbrowser
import smtplib
import requests
import urllib
import gmaps
import urllib.request as urllib2
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from time import strftime
from gtts import gTTS
from youtube_search import YoutubeSearch


wikipedia.set_lang('vi')
language = 'vi'
path = ChromeDriverManager().install()



def speak(text):
    print("Bot: {}".format(text))
    output = gTTS(text,lang="vi", slow=False)
    output.save("output.mp3")
    playsound.playsound('output.mp3', True)
    os.remove("output.mp3")
    
    


def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as mic:
        print("Tôi: ", end='')
        audio = r.listen(mic, phrase_time_limit=5)
        try:
            text = r.recognize_google(audio, language="vi-VN")
            print(text)
            return text
        except:
            print("...")
            return 0


def stop():
    speak("Hẹn gặp lại anh yêu")


def get_text():
    for i in range(3):
        text = get_audio()
        if text:
            return text.lower()
        elif i < 2:
            speak("Em không nghe rõ. Anh nói lại được không!")
    time.sleep(2)
    stop()
    return 0



def hello(name):
    day_time = int(strftime('%H'))
    if day_time < 12:
        speak("Chào buổi sáng bạn {}. Chúc anh một ngày tốt lành.".format(name))
    elif 12 <= day_time < 18:
        speak("Chào buổi chiều bạn {}. Anh đã dự định gì cho chiều nay chưa.".format(name))
    else:
        speak("Chào buổi tối bạn {}. Anh đã ăn tối chưa nhỉ.".format(name))


def get_time(text):
    now = datetime.datetime.now()
    if "giờ" in text:
        speak('Bây giờ là %d giờ %d phút' % (now.hour, now.minute))
    elif "ngày" in text:
        speak("Hôm nay là ngày %d tháng %d năm %d" %
              (now.day, now.month, now.year))
    else:
        speak("Bot chưa hiểu ý của anh. Anh nói lại được không?")


def google_search(text):
    search_for = text.split("tìm kiếm", 1)[1]
    speak("Oke la")
    driver = webdriver.Chrome(path)
    driver.get("http://www.google.com")
    que = driver.find_element_by_xpath("//input[@name='q']")
    que.send_keys(str(search_for))
    que.send_keys(Keys.RETURN)

# def open_google_and_search(text):
#     speak("Anh cần tìm gì ạ")
#     search_for = get_text()
#     speak('Okay anh iu!')
#     tim = get_text()
#     while True:
#         result = (path)
#         if result:
#             break
#     url = 'https://www.google.com' + result[0]['url_suffix']
#     webbrowser.open(url)
#     speak("Đã có kết quả cho."+ tim)



def tell_me_about():
    try:
        text = get_text()
        contents = wikipedia.summary(text.split("định nghĩa",1)).split('\n')
        speak(contents[0])
        time.sleep(10)
        for content in contents[1:]:
            speak("anh muốn nghe thêm không")
            ans = get_text()
            if "có" not in ans:
                break    
            speak(content)
            time.sleep(10)

        speak('oke anh')
    except:
        speak("Em không định nghĩa được thuật ngữ của anh. mời anh nói lại")




def play_song(text):
    # speak("đang mở youtobe")
    # search_for = text.split("tìm", 1)[1]
    # os.startfile('C:\Program Files\Google\Chrome\Application\chrome.exe')
    # speak("Bài hát bạn yêu cầu đã được mở.")
    speak('Anh muốn nghe gì')
    mysong = get_text()
    while True:
        result = YoutubeSearch(mysong, max_results=10).to_dict()
        if result:
            break
    url = 'https://www.youtube.com' + result[0]['url_suffix']
    webbrowser.open(url)
    speak("Em mở rồi.")


def google_map(text):
    dia_chi = get_text()
    speak("Em đang tìm đường đến")
    while True:
        result = gmaps(dia_chi.split("đến", 1)[1])
        if result:
            break
    url =  'https://www.https://www.google.com/maps/@9.779349,105.6189045,11z?hl=vi-VN.com' + result[0]['url_suffix']
    webbrowser.open(url)
    speak("Em mở rồi.")   


def change_wallpaper():
    api_key = 'RF3LyUUIyogjCpQwlf-zjzCf1JdvRwb--SLV6iCzOxw'
    url = 'https://api.unsplash.com/photos/random?client_id=' + \
        api_key  # ảnh lấy từ unspalsh.com
    f = urllib2.urlopen(url)
    json_string = f.read()
    f.close()
    parsed_json = json.loads(json_string)
    photo = parsed_json['urls']['full']
    # nơi tải ảnh xuống
    urllib2.urlretrieve(photo, "C:/Users/a.png")
    image=os.path.join("C:/Users/a.png")
    ctypes.windll.user32.SystemParametersInfoW(20,0,image,3)
    speak('Hình nền máy tính vừa được thay đổi')







def help_me():
    speak("""Bot có thể giúp anh thực hiện các câu lệnh sau đây:
    1. Chào hỏi
    2. Hiển thị giờ
    4. Tìm kiếm trên Google
    7. Mở video nhạc
    8. Thay đổi hình nền máy tính
    10. Kể bạn biết về thế giới """)






def assistant():
    speak("Hú hú, anh tên gì?")
    name = get_text()
    if name:
        speak("Chào {}".format(name)+ " anh cần em giúp gì ạ")
        #speak("Anh cần em giúp gì ạ?")
        while True:
            text = get_text()
            if not text:
                break
            elif "dừng" in text or "tắt đi" in text or "cút" in text or "tạm biệt" in text:
                stop()
                break
            elif "có thể làm gì" in text:
                help_me()
            elif "tìm đường" in text:
                google_map(text)    
            elif "chào em" in text:
                hello(name)
            elif "nay ngày mấy" in text:
                get_time(text)
            elif "google" in text:  
                open_google_and_search(text)
            elif "định nghĩa" in text:
                tell_me_about()
            elif "mở nhạc" in text:
                play_song(text)
            elif "hình nền" in text:
                change_wallpaper()
            else:
                speak("Anh cần gì ở em baby")
                
assistant()