import ee
from tqdm import tqdm
import os
import requests

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'
ee.Initialize()
print("start")

data_list=['2015_1','2015_2','2015_3','2015_4','2016_1','2016_2','2016_3','2016_4','2017_1','2017_2','2017_3','2017_4','2018_1','2018_2',
           '2018_3','2018_4','2019_1','2019_2','2019_3','2019_4']
filelist=['users/qxcnwu/effunet_demo_left'+data_list[i]+"_1" for i in range(len(data_list))]

collection_list=[ee.Image(i) for i in filelist]
for i in collection_list:
    try:
        url=i.getDownloadURL({'scale':100000,'maxPixels': 1e13})
    except:
        collection_list.remove(i)

print(len(collection_list))
imagecollection=ee.ImageCollection(collection_list)
mean=imagecollection.mean()
task = ee.batch.Export.image.toDrive(image=mean,
                                     description='mock_export',
                                     folder='user/qxcnwu',
                                     fileNamePrefix='mean_1',
                                     region= ee.Geometry.Polygon([[92.0614845441127,27.911597760742698],
                                                            [99.5157081769252,27.911597760742698],
                                                            [99.5157081769252,32.92540841581852],
                                                            [92.0614845441127,32.92540841581852],
                                                            [92.0614845441127,27.911597760742698]]
                                     ))

task.start()