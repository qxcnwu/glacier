var imagecollection=ee.ImageCollection([image,image2,image3,image4,image5,image6,image7,image8,image9,image10,image11,image12,image13,image14,image15,image16,image17,image18,image19,image20])
var mean=imagecollection.mean()
Map.addLayer(mean)
Export.image.toDrive({
    image: mean,
    description: 'imageToAssetExample',
    scale: 30,
    region :geometry
});