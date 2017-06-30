#Searching and Downloading Google Images/Image Links

#Import Libraries

import time       #Importing the time library to check the time of code execution
import sys    #Importing the System Library
import os
from PIL import Image
import configparser
max_images = 20
keyword_file = "keywords.txt"
convert_size = 128
config_file_path = "config.ini"
images_folder_path = "images"

#This list is used to search keywords. You can edit this list to search for google images of your choice. You can simply add and remove elements of the list.
search_keyword = []

#This list is used to further add suffix to your search term. Each element of the list will help you download 100 images. First element is blank which denotes that no suffix is added to the search keyword of the above list. You can edit the list by adding/deleting elements from it.So if the first element of the search_keyword is 'Australia' and the second element of keywords is 'high resolution', then it will search for 'Australia High Resolution'
keywords = ['']

#Downloading entire Web Document (Raw Page Content)
def download_page(url):
    version = (3,0)
    cur_version = sys.version_info
    if cur_version >= version:     #If the Current Version of Python is 3.0 or above
        import urllib.request    #urllib library for Extracting web pages
        try:
            headers = {}
            headers['User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"
            req = urllib.request.Request(url, headers = headers)
            resp = urllib.request.urlopen(req)
            respData = str(resp.read())
            return respData
        except Exception as e:
            print(str(e))
    else:                        #If the Current Version of Python is 2.x
        import urllib2
        try:
            headers = {}
            headers['User-Agent'] = "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"
            req = urllib2.Request(url, headers = headers)
            response = urllib2.urlopen(req)
            page = response.read()
            return page
        except:
            return"Page Not found"


#Finding 'Next Image' from the given raw page
def _images_get_next_item(s):
    start_line = s.find('rg_di')
    if start_line == -1:    #If no links are found then give an error!
        end_quote = 0
        link = "no_links"
        return link, end_quote
    else:
        start_line = s.find('"class="rg_meta"')
        start_content = s.find('"ou"',start_line+1)
        end_content = s.find(',"ow"',start_content+1)
        content_raw = str(s[start_content+6:end_content-1])
        return content_raw, end_content


#Getting all links with the help of '_images_get_next_image'
def _images_get_all_items(page):
    items = []
    global max_images
    count = 0
    while True:
        item, end_content = _images_get_next_item(page)
        if item == "no_links" or count >= max_images:
            break
        else:
            items.append(item)      #Append all the links in the list named 'Links'
            time.sleep(0.1)        #Timer could be used to slow down the request for image downloads
            page = page[end_content:]
        count += 1
    return items

def download_images():
    t0 = time.time()   #start the timer

    #Download Image Links
    i= 0
    items = {}
    while i<len(search_keyword):
        items[search_keyword[i]] = []
        iteration = "Item no.: " + str(i+1) + " -->" + " Item name = " + str(search_keyword[i])
        print (iteration)
        print ("Evaluating...")
        search_keywords = search_keyword[i]
        search = search_keywords.replace(' ','%20')
        j = 0
        while j<len(keywords):
            pure_keyword = keywords[j].replace(' ','%20')
            url = 'https://www.google.com/search?q=' + search + pure_keyword + '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg'
            #url3 = "http://www.bing.com/images/search?sp=-1&pq=" + search + pure_keyword +"&sc=8-2&sk=&q=cat&qft=+filterui:imagesize-medium&FORM=R5IR1"
            #url4 = "http://www.bing.com/images/search?sp=-1&pq=" + search + pure_keyword + "&sc=8-2&sk=&q=cat&qft=+filterui:imagesize-small&FORM=R5IR1"
            #url3 = "https://www.bing.com/images/search?sp=-1&pq=cat&sc=8-2&sk=&q=cat&qft=+filterui:imagesize-medium&FORM=R5IR1"
            raw_html =  (download_page(url))
            #raw_html3 =  (download_page(url3))
            #raw_html4 =  (download_page(url4))
            time.sleep(0.1)
            items[search_keyword[i]] = items[search_keyword[i]] + (_images_get_all_items(raw_html))
            #items[search_keyword[i]] = items[search_keyword[i]] + (_images_get_all_items(raw_html3))
            #items[search_keyword[i]] = items[search_keyword[i]] + (_images_get_all_items(raw_html4))
            j = j + 1
        #print ("Image Links = "+str(items))
        print ("Total Image Links = "+str(len(items)))
        print ("\n")
        i = i+1

    t1 = time.time()    #stop the timer
    total_time = t1-t0   #Calculating the total time required to crawl, find and download all the links of 60,000 images
    print("Total time taken: "+str(total_time)+" Seconds")
    print ("Starting Download...")

    # IN this saving process we are just skipping the URL if there is any error
    
    errorCount=0
    #while(k<len(items)):
    if not os.path.exists(images_folder_path):
        os.makedirs(images_folder_path)
    for key,  value in items.items():
        if not os.path.exists(images_folder_path + key):
            os.makedirs(images_folder_path + key)
        else:
            print("WARNING: Keyword " + key + " already downloaded in " + images_folder_path + key + ", please delete the folder first.")
            print("Skipping keyword " + key + ".")
            errorCount += 1
            continue
        os.chdir(images_folder_path + key)
        k=0
        while(k<len(value)):
            from urllib.request import Request, urlopen
            from urllib.error import URLError, HTTPError
            try:
                req = Request(value[k], headers={"User-Agent": "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"})
                response = urlopen(req)
                output_file = open(str(k+1),'wb')
                data = response.read()
                output_file.write(data)
                response.close();
                im = Image.open(str(k+1)).convert("RGBA")
                bg = Image.new("RGB",  im.size)
                bg.paste(im, im)
                bg.thumbnail((convert_size, convert_size),  Image.ANTIALIAS)
                bg.save(str(k+1),  "JPEG")
                print("completed ====> "+str(k+1))
                k=k+1;

            except IOError:   #If there is any IOError
                errorCount+=1
                print("IOError on image "+str(k+1))
                k=k+1;
            except HTTPError:  #If there is any HTTPError
                errorCount+=1
                print("HTTPError"+str(k))
                k=k+1;
            except URLError:
                errorCount+=1
                print("URLError "+str(k))
                k=k+1;
        os.chdir("..")

    print("\n")
    print("Downloads complete!")
    print("\n"+str(errorCount)+" ----> total Errors")
    
def create_config(path):
    config = configparser.ConfigParser()
    config.add_section("Settings")
    config.set("Settings", "convert_size", "128")
    config.set("Settings", "max_images", "20")
    config.set("Settings", "keywords_path", "keywords.txt")
    config.set("Settings",  "images_folder_path",  "images")
    with open(path, "w") as config_file:
        config.write(config_file)

def read_config(path):
    global keyword_file
    global max_images
    global convert_size
    global images_folder_path
    config = configparser.ConfigParser()
    config.read(config_file_path)
    keyword_file = config.get("Settings", "keywords_path")
    max_images = int(config.get("Settings", "max_images"))
    if(max_images == 0):
        max_images = 999
    convert_size = int(config.get("Settings", "convert_size"))
    images_folder_path = config.get("Settings",  "images_folder_path") 
    if not images_folder_path.endswith('/'):
        images_folder_path += '/'
    with open(keyword_file,  'r') as f:
        for line in f:
            for word in line.split():
                search_keyword.append(word)

def main():
    if not os.path.exists(config_file_path):
        create_config(config_file_path)
        print("Creating config file, run script again after you've configured it from config.ini")
        exit(0)
    read_config(config_file_path)
    download_images()
        
if __name__ == '__main__':
    main()
  
