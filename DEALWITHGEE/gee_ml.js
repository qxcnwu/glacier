Map.centerObject(image2,9);
var model = ee.Model.fromAiPlatformPredictor({
    projectName:'qxcgee',
    modelName:'geemodel7',
    inputTileSize : [128,128],
    inputOverlapSize : [64, 64],
    proj : ee.Projection('EPSG:4326').atScale(30),
    fixInputProj : true,
    outputBands : {'p': {
            'type': ee.PixelType.float(),
            'dimensions': 1
        }
    }
});

var image =ee.Image(ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA')
    .filterDate('2017-01-01', '2017-12-31')
    .filterBounds(geometry3)
    .sort('CLOUD_COVER')
    .first());

var normal_image = image.select(["B7","B3","B2"]).clip(geometry2).float();
var predictions = model.predictImage(normal_image.toArray());
var probabilities = predictions.arrayFlatten([['land','glacer']]);
var label1 = predictions.arrayArgmax().arrayGet([0]).rename('label');

var normal_image = image.select(["B7","B3","B2"]).clip(geometry4).float();
var predictions = model.predictImage(normal_image.toArray());
var probabilities = predictions.arrayFlatten([['land','glacer']]);
var label2 = predictions.arrayArgmax().arrayGet([0]).rename('label');

var normal_image = image.select(["B7","B3","B2"]).clip(geometry5).float();
var predictions = model.predictImage(normal_image.toArray());
var probabilities = predictions.arrayFlatten([['land','glacer']]);
var label3 = predictions.arrayArgmax().arrayGet([0]).rename('label');

var answer=label1.max(label2).max(label3)

Map.addLayer(answer, label_vis,'all')
Map.addLayer(label1, label_vis,'label1')
Map.addLayer(label2, label_vis,'label2')
Map.addLayer(label3, label_vis,'label3')

Export.image.toAsset({
    image: answer,
    description: "l8ImageDrive",
    maxPixels: 1e13
});
