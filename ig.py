import requests
import re
from urllib.request import urlretrieve
import os
import time

class Instagram:
    def __init__(self):
        # initialization
        # folder name
        # ig's url
        # user id
        # current path
        # set faker header include user-agent and cookie
        # faker uri to convect new url
        # save all images url
        # save every loading's images url
        # all end cursors
        # set regular expression
        # check the folder

        self.name = []
        self.ig_url = []
        self.user_id = ''
        self.path = os.getcwd()
        self.headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36',
            'cookie': 'use your cookie' 
        }
        self.uri = 'https://www.instagram.com/graphql/query/?query_hash=f2405b236d85e8296cf30347c9f08c2a&variables=%7B%22id%22%3A%22{user_id}%22%2C%22first%22%3A12%2C%22after%22%3A%22{cursor}%3D%3D%22%7D'
        self.all_images = []
        self.images = []
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
        # call self.read_file()
        # call self.travel()
        self.read_file()
        self.travel()

    def read_file(self):
        # try to enter the folder. name is txt_files
        # if can't enter. that print folder can not found
        # change path to xxx\tet_files
        # finally get the folder's file
        # try to read the txt file. use utf-8 code
        # if can't read. that print read error
        # finally get the file's content
        # close the file
        # call self.file_process()

        try:
            os.chdir(self.path + r'\txt_files')
        except Exception:
            print('folder can not found')
        finally:
            txt_files = os.listdir()
        try:
            ig_url = open(txt_files[0], 'r', encoding='UTF-8')
        except Exception:
            print('read error')
        finally:
            self.ig_url = ig_url.readlines()
            ig_url.close()
        self.file_process()

    def file_process(self):
        # file process
        # remove the first string. avoid special character
        # do len(self.ig_url) times
        # use ' ' to replace '\n'
        # change to original path

        self.ig_url.pop(0)
        for i in range(len(self.ig_url)):
            self.ig_url[i] = self.ig_url[i].replace('\n', '')
            self.name.append(self.ig_url[i][26:len(self.ig_url[i])-1:])
        os.chdir(self.path)

    def travel(self):
        # do len(self.ig_url) times
        # change path to xxx\download
        # check the folder. name is self.name[0]
        # change path to xxx\download\self.name[0]
        # send request and get url's data
        # use re to find user_id
        # use re to find count
        # use re to find end_cursor
        # push end_cursor to cursor
        # use re to find image url
        # pash image to images and all_images
        # call self.start_download()
        # remove self.ig_url[0] and self.name[0]
        # wait 30s

        times = len(self.ig_url)
        for i in range(times):
            os.chdir(self.path + r'\download')
            if not os.path.exists(self.name[0]):
                os.makedirs(self.name[0])
            os.chdir(os.getcwd() + '\\' + self.name[0])
            req = requests.get(self.ig_url[0])
            user_id = re.findall(self.profilePage, req.text)
            self.user_id = str(user_id[0])
            count = re.findall(self.count, req.text)
            count = list(count)
            count = str(count[0])
            self.page_info = self.page_info_s + str(count) + self.ender
            end_cursor = re.findall(self.end_cursor, req.text)
            self.cursors.append(str(end_cursor[0]))
            images = re.findall(self.index_display_url, req.text)
            images = list(images)
            for image in images:
                self.images.append(image)
                self.all_images.append(image)

            self.start_download()
            self.ig_url.pop(0)
            self.name.pop(0)
            time.sleep(30)


    def start_download(self):
        # start dawonload
        # convert url
        # send requests and get url's that use header
        # remove self.cursors[0] let loop can stop
        # use re to find page_info
        # check page_info. if no data. break
        # if have data. check = str(page_info[0]) true or false
        # use re to find end_cursor
        # cursor = end_cursor[0] only need end_cursor[0]
        # push cursor to cursors
        # use re to find images url
        # do linear search. avoid to download repeated photo
        # do len(self.images) times
        # check photo_number to set file name
        # range 0001 ~ 9999
        # every 50 photo wait 10s
        # download photo
        # wait 1.2s
        # clear self.images
        # if check == false remove self.cursors[0]

        photo_number = 1
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
            cursor = end_cursor[0]
            self.cursors.append(cursor)
            images = re.findall(self.display_url, req.text)
            images = list(images)
            for image in images:
                test = False
                for i in range(len(self.all_images)):
                    if image == self.all_images[i]:
                        test = True
                if not test:
                    self.images.append(image)
                    self.all_images.append(image)

            for i in range(len(self.images)):
                if photo_number < 10:
                    file_id = '000' + str(photo_number) + '.jpg'
                elif photo_number < 100:
                    file_id = '00' + str(photo_number) + '.jpg'
                elif photo_number < 1000:
                    file_id = '0' + str(photo_number) + '.jpg'
                else:
                    file_id = str(photo_number) + '.jpg'

                if photo_number > 49 and photo_number % 50 == 0:
                    time.sleep(10)
                    
                urlretrieve(self.images[i], file_id)

                print(file_id)
                photo_number += 1
                time.sleep(1.2)
            self.images.clear()
            if not check == 'true':
                self.cursors.pop(0)

if __name__ == '__main__':
    ig = Instagram()
    ig.start()








