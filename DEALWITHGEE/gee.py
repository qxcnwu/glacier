import ee
from tqdm import tqdm
import os
import requests

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'
ee.Initialize()
print("start")

x_max=93.70073334
y_max=30.28452958
x_min=94.16666568
y_min=29.94563793
json_list=[[x_max,y_max],[x_max,y_min],[x_min,y_min],[x_min,y_max]]
polygon = ee.Geometry.Polygon(json_list)
# Specify inputs (Landsat bands) to the model and the response variable.
opticalBands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
thermalBands = ['B10', 'B11']
BANDS = opticalBands + thermalBands
RESPONSE = 'impervious'
FEATURES = BANDS + [RESPONSE]

json_list=[[x_max,y_max],[x_max,y_min],[x_min,y_min],[x_min,y_max]]
# Use Landsat 8 surface reflectance data.
l8sr = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')

# Cloud masking function.
def maskL8sr(image):
  cloudShadowBitMask = ee.Number(2).pow(3).int()
  cloudsBitMask = ee.Number(2).pow(5).int()
  qa = image.select('pixel_qa')
  mask1 = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
    qa.bitwiseAnd(cloudsBitMask).eq(0))
  mask2 = image.mask().reduce('min')
  mask3 = image.select(opticalBands).gt(0).And(
          image.select(opticalBands).lt(10000)).reduce('min')
  mask = mask1.And(mask2).And(mask3)
  return image.select(opticalBands).divide(10000).addBands(
          image.select(thermalBands).divide(10).clamp(273.15, 373.15)
            .subtract(273.15).divide(100)).updateMask(mask)

# The image input data is a cloud-masked median composite.
image = l8sr.filterDate(
    '2017-01-01', '2017-12-31').filterBounds(ee.Geometry.Point(95.3760140369, 30.2416382636)).map(maskL8sr).median().select(["B2","B3","B7"]).float()
print(image.getInfo())

model = ee.Model.fromAiPlatformPredictor(
    projectName='qxcgee',
    modelName='geemodel7',
    inputTileSize = [512,512],
    inputOverlapSize = [0, 0],
    proj = ee.Projection('EPSG:4326').atScale(10000),
    fixInputProj = True,
    outputBands = {'impervious': {
        'type': ee.PixelType.float(),
        'dimensions': 1
      }
    }
)
predictions = model.predictImage(image.toArray())
url = predictions.getDownloadURL({'scale':10000,'maxPixels': 1e13})
print(url)
