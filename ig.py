import requests
import re
import os
import time
import random

class Instagram:
    def __init__(self):
        # initialization
        self.name = ''
        self.ig_url = ''
        self.user_id = ''
        self.filename = ''
        self.photo_number = 0
        self.path = os.getcwd()
        self.ua = [
            "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
            "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
            "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:38.0) Gecko/20100101 Firefox/38.0",
            "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; .NET4.0C; .NET4.0E; .NET CLR 2.0.50727; .NET CLR 3.0.30729; .NET CLR 3.5.30729; InfoPath.3; rv:11.0) like Gecko",
            "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)",
            "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)",
            "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)",
            "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
            "Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
            "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11",
            "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
            "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)",
            "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; TencentTraveler 4.0)",
            "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
            "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; The World)",
            "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SE 2.X MetaSr 1.0; SE 2.X MetaSr 1.0; .NET CLR 2.0.50727; SE 2.X MetaSr 1.0)",
            "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)",
            "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Avant Browser)",
            "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
            "Mozilla/5.0 (iPhone; U; CPU iPhone OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
            "Mozilla/5.0 (iPod; U; CPU iPhone OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
            "Mozilla/5.0 (iPad; U; CPU OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
            "Mozilla/5.0 (Linux; U; Android 2.3.7; en-us; Nexus One Build/FRF91) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
            "MQQBrowser/26 Mozilla/5.0 (Linux; U; Android 2.3.7; zh-cn; MB200 Build/GRJ22; CyanogenMod-7) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
            "Opera/9.80 (Android 2.3.4; Linux; Opera Mobi/build-1107180945; U; en-GB) Presto/2.8.149 Version/11.10",
            "Mozilla/5.0 (Linux; U; Android 3.0; en-us; Xoom Build/HRI39) AppleWebKit/534.13 (KHTML, like Gecko) Version/4.0 Safari/534.13",
            "Mozilla/5.0 (BlackBerry; U; BlackBerry 9800; en) AppleWebKit/534.1+ (KHTML, like Gecko) Version/6.0.0.337 Mobile Safari/534.1+",
            "Mozilla/5.0 (hp-tablet; Linux; hpwOS/3.0.0; U; en-US) AppleWebKit/534.6 (KHTML, like Gecko) wOSBrowser/233.70 Safari/534.6 TouchPad/1.0",
            "Mozilla/5.0 (SymbianOS/9.4; Series60/5.0 NokiaN97-1/20.0.019; Profile/MIDP-2.1 Configuration/CLDC-1.1) AppleWebKit/525 (KHTML, like Gecko) BrowserNG/7.1.18124",
            "Mozilla/5.0 (compatible; MSIE 9.0; Windows Phone OS 7.5; Trident/5.0; IEMobile/9.0; HTC; Titan)",
            "UCWEB7.0.2.37/28/999",
            "NOKIA5700/ UCWEB7.0.2.37/28/999",
            "Openwave/ UCWEB7.0.2.37/28/999",
            "Mozilla/4.0 (compatible; MSIE 6.0; ) Opera/UCWEB7.0.2.37/28/999",
            "Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25"
            ]
        self.headers = {
            'cookie': 'mid=XQDL5AAEAAF1xs159fwePxfoAfCR; fbm_124024574287414=base_domain=.instagram.com; fbsr_124024574287414=DXAlRLCDHLsS_nRYqxJDjxwqI7i4HSJ5AJVqdOwomC8.eyJjb2RlIjoiQVFCM3lkTlctSWVZZVd5ckdTbnduZDFobllYQ3NKeDVKLXlJMUI2WjNVOXZXTXpOeGpTM1VkdXB6ZHM4ZFZyZFZVUzhTX29KRFFpY1U4SlhNNWpSX2ZQcGJPVUJJb3lpMlR6WXZ3UmpYWXg1WV9QNHhROVVZMHFiSmxld1dlaWlkdDFsS25uMVY2a0lCM0ROak40SXNxbWZscG1FLVlSNlhHXzNiMTJuRkl1TXdacVFHb3NURDBUSjlZdU9STHpYQTNab0tXMjNGRkZuaUxJQmViWHFoOHlzbE1ORExYZlZGdDRyeEZ1SjFfZ1I3UGxsT3lrc1l6NlpTODZieV9mYmVveUdpeTRVSjVCUkJTYXctN1V5bGF4RG1QRzlLdEh0SlpjSTdCQThmRW0zQ1RqdU0xVHZ4MEZEYVRIYXRJMGUySjRQUXhQRlNUOXo3Y1A0LWUzaDdhd2FDd1B5dEN0N0dfazZRUkd5cmxBUU9nIiwidXNlcl9pZCI6IjEwMDAwMjk1OTkyMzYzNCIsImFsZ29yaXRobSI6IkhNQUMtU0hBMjU2IiwiaXNzdWVkX2F0IjoxNTYwMzMzMzUxfQ; shbid=3982; shbts=1560333352.5522082; ds_user_id=12895225475; sessionid=12895225475%3AzJ2uLgxbWVLMEI%3A6; csrftoken=umCHtlFhIgcxhchFSo1dtqO6Zoz57D8q; rur=ATN; urlgen="{\"2001:b011:c005:376d:692:26ff:fed3:3fea\": 3462}:1hb04b:KM6Gy4DC8vsXtblD8RMMCjxZf6s"'
        }
        self.uri = 'https://www.instagram.com/graphql/query/?query_hash=f2405b236d85e8296cf30347c9f08c2a&variables=%7B%22id%22%3A%22{user_id}%22%2C%22first%22%3A12%2C%22after%22%3A%22{cursor}%3D%3D%22%7D'
        self.all_images = []
        self.images = []
        self.search = []
        self.cursors = []
        self.profilePage = re.compile('"profilePage_([0-9]+)"')
        self.end_cursor = re.compile('"end_cursor":"([a-zA-Z0-9]+)=')
        self.page_info_s = '{"data":{"user":{"edge_owner_to_timeline_media":{"count":'
        self.ender = ',"page_info":{"has_next_page":([a-z]+),"'
        #self.page_info = re.compile('{"data":{"user":{"edge_owner_to_timeline_media":{"count":{count},"page_info":{"has_next_page":([a-z]+),"')
        self.count = re.compile('"edge_owner_to_timeline_media":{"count":([0-9]+),"page_info"')
        self.index_display_url = re.compile('"display_url":"([-a-zA-Z0-9/_.:=?]+)","edge_liked_by"')
        self.display_url = re.compile('"display_url":"([-a-zA-Z0-9/_.:=?]+)","display_resources"')
        if not os.path.exists('download'):
            os.makedirs('download')

    def start(self):
        # start to work
        self.input_id()
        self.read_file()
        self.file_process()
        self.travel()
        self.output_file()

    def input_id(self):
        # input id
        self.name = input('input id = ')
        self.ig_url = 'https://www.instagram.com/' + self.name
        print(self.ig_url)
        # os.chdir(self.path + r'download')
        os.chdir(os.path.join(self.path, 'download'))
    def read_file(self):
        # read file
        if not os.path.exists(self.name):
            os.makedirs(self.name)
        #os.chdir(os.getcwd() + '\\' + self.name)
        os.chdir(os.path.join(os.getcwd(), self.name))
        self.filename = self.name + '.txt'

        if self.filename not in os.listdir():
            try:
                f = open(self.filename, 'a')
            except:
                print('read file error')
            finally:
                f.write('i = 0\n')
                f.close()
        f = open(self.filename, 'r')
        self.search = f.readlines()
        f.close()

    def file_process(self):
        # file process
        self.search.pop(0)
        for i in range(len(self.search)):
            self.search[i] = self.search[i].replace('\n', '')

    def travel(self):
        self.get_index()
        self.start_download()

    def get_index(self):
        # get data from index
        req = requests.get(self.ig_url)
        user_id = re.findall(self.profilePage, req.text)
        print(user_id)
        self.user_id = str(user_id[0])
        count = re.findall(self.count, req.text)
        count = list(count)
        count = str(count[0])
        self.page_info = self.page_info_s + str(count) + self.ender
        end_cursor = re.findall(self.end_cursor, req.text)
        self.cursors.append(str(end_cursor[0]))
        images = re.findall(self.index_display_url, req.text)
        images = list(images)
        self.photo_number = len(self.search) + 1
        self.url_search(images)

    def start_download(self):
        # get new data
        while len(self.cursors) > 0:
            url = self.uri.format(user_id=self.user_id, cursor=self.cursors[0])
            req = requests.get(url, headers=self.headers)
            self.cursors.pop(0)
            page_info = re.findall(self.page_info, req.text)
            if len(page_info) == 0:
                break
            else:
                check = str(page_info[0])
            end_cursor = re.findall(self.end_cursor, req.text)
            end_cursor = list(end_cursor)
            if len(end_cursor) == 0:
                break
            cursor = end_cursor[0]
            self.cursors.append(cursor)
            images = re.findall(self.display_url, req.text)
            images = list(images)
            self.url_search(images)

            for i in range(len(self.images)):

                file_id = self.get_file_id()
                if self.photo_number > 29 and self.photo_number % 30 == 0:
                    time.sleep(10)
                    print('wait 10s')

                self.download_photo(self.images[i], file_id)
                f = open(self.filename, 'a+')
                f.write(self.images[i] + '\n')
                f.close()

                print(file_id)
                self.photo_number += 1
                time.sleep(2)
            self.images.clear()
            if not check == 'true':
                self.cursors.pop(0)

    def url_search(self, images):
        # check url
        for image in images:
            if image not in self.search:
                self.images.append(image)
                self.all_images.append(image)
                self.search.append(image)
            else:
                pass
                #print('have')

    def get_file_id(self):
        # set file id
        file_id = '{:0>4d}'.format(self.photo_number)
        file_id = str(file_id) + '.jpg'
        return file_id

    def download_photo(self, url, file_id):
        # check the photo
        # download
        r = requests.get(url, headers=self.get_header())
        if not r.status_code == 200:
            print('requests error' , r.status_code)
            self.photo_number -= 1
        else:
            with open(file_id, 'wb') as f:
                f.write(r.content)

    def get_header(self):
        headers = {'user-agent': random.choice(self.ua)}
        return headers


    def output_file(self):
        # output url data
        f = open(self.filename, 'w')
        f.write('i = ' + str(len(self.search)) + '\n')
        for each in self.search:
            f.write(each + '\n')

if __name__ == '__main__':
    ig = Instagram()
    ig.start()







