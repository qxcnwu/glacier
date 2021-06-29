var image =ee.Image(ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA')
      .filterDate('2017-01-01', '2017-12-31')
      .filterBounds(geometry2)
      .sort('CLOUD_COVER')
      .first());

// var rgb = image.divide(ee.Number(256));
var rgb = image.select(["B7","B3","B2"])
var model = ee.Model.fromAiPlatformPredictor({
    projectName:'qxcgee',
    modelName:'geemodel4',
    version:'v1',
    inputTileSize : [256,256],
    inputOverlapSize : [0, 0],
    proj : ee.Projection('EPSG:4326').atScale(15),
    fixInputProj : true,
    outputBands : {'impervious': {
        'type': ee.PixelType.float(),
        'dimensions': 1
      }
    }
  }
)

function normalization(image,scale){
var mean_std = image.reduceRegion({
  reducer: ee.Reducer.mean()
            .combine(ee.Reducer.stdDev(),null, true),
  scale: scale,
  maxPixels: 10e13,
  // tileScale: 16
});
// use unit scale to normalize the pixel values
var unitScale = ee.ImageCollection.fromImages(
  image.bandNames().map(function(name){
    name = ee.String(name);
    var band = image.select(name);
    var mean=ee.Number(mean_std.get(name.cat('_mean')));
    var std=ee.Number(mean_std.get(name.cat('_stdDev')));
    var result_band=band.subtract(mean).divide(std);
    return result_band;
})).toBands().rename(image.bandNames());
  return unitScale;
}

var normal_image=normalization(image.select(["B7","B3","B2"]),30)

var normal_image=rgb.clip(geometry2).float()


var predictions = model.predictImage(normal_image.toArray());
var probabilities = predictions.arrayFlatten([['bare','water']]);
var label = predictions.arrayArgmax().arrayGet([0]).rename('label');

print(label,predictions)

Export.image.toDrive({
  image: rgb,
  description: "l8ImageDrive",
  fileNamePrefix: "l8Img3",
  scale: 15,
  region: geometry,
  maxPixels: 1e13
});

Map.centerObject(image,9);
// Map.addLayer(normal_image)
Map.addLayer(rgb)
Map.addLayer(label, label_vis);