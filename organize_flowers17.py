import os
import glob
import datetime
import tarfile
import urllib.request

def download_dataset(filename, url, work_dir):
    if not os.path.exists(filename):
        print("[INFO] Downloading flowers17 dataset.....")
        filename, _= urllib.request.urlretrieve(url + filename, filename)
        statinfo = os.stat(filename)
        print("[INFO] Succesfully downloaded " + filename + " " + str(statinfo.st_size) + untar(filename, work_dir))
def jpg_files(members):
    for tarinfo in members:
        if os.path.splitext(tarinfo.name)[1] == ".jpg":
            yield tarinfo
def untar(fname, path):
    tar = tarfile.open(fname)
    tar.extractall(path=path, members=jpg_files(tar))
    tar.close()
    print("[INFO] Dataset extracted successfully.")

if __name__ =='__main__':
    flowers17_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/17/"
    flowers17_name = "17flowers.tgz"
    train_dir      = "datase t"
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    download_dataset(flowers17_name, flowers17_url, train_dir)   
    if os.path.exists(train_dir + "\\jpg"):
        os.rename(train_dir + "\\jpg", train_dir + "\\train")

    class_limit = 17
    image_path = glob.glob(train_dir + "\\train\\*.jpg")
    label = 0
    i = 0
    j = 80

    class_names = ["daffodil", "snowdrop", "lilyvalley", "bluebell", "crocus", "iris", "tigerlily", "tulip", "fritillary", "sunflower",
                  "daisy", "coltsfoot", "dandelion", "cowslip", "buttercup", "windflower", "pansy"]

    for x in range(1, class_limit+1):
        os.makedirs(train_dir + "\\train\\" + class_names[label])
        cur_path = train_dir + "\\train\\" + class_names[label] + "\\"

        for index, image_path in enumerate(image_path[i:j], start=1):
            origin_path     = image_path
            image_path      = image_path.split("\\")
            image_file_name = str(index) + ".jpg"
            os.rename(origin_path, cur_path + image_file_name)
        
        i += 80
        j += 80
        label += 1