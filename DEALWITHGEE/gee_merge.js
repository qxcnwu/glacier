var month_end=["01-31","02-28","03-31","04-30","05-31","06-30","07-31","08-31","09-30","10-31","11-30","12-31"];
var month_start=["01-01","02-01","03-01","04-01","05-01","06-01","07-01","08-01","09-01","10-01","11-01","12-01"];
var start=2015;
var end=2021;

for (var i=start;i<end;i++)
{
    for (var j=1;j<13;j++)
    {
        var start_date=i+"-"+month_start[0];
        var end_date=i+"-"+month_end[11];
        var image =ee.Image(ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA')

            .filterDate(start_date, end_date)

            .filterBounds(ee.Geometry.Point(95.3760140369, 30.2416382636))

            .sort('CLOUD_COVER')

            .first());

        var rgb = image.select('B7', 'B3', 'B2');

        var pan = image.select('B8');

        var huesat = rgb.rgbToHsv().select('hue','saturation');

        var upres = ee.Image.cat(huesat,pan).hsvToRgb();

        var name=i+"_"+j;
        Export.image.toDrive({
            image: upres,
            description: name,
            fileNamePrefix: name,
            scale: 15,
            maxPixels: 1e13
        });
    }
}



var image =ee.Image(ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA')

    .filterDate('2016-12-01', '2016-12-31')

    .filterBounds(ee.Geometry.Point(95.3760140369, 30.2416382636))

    .sort('CLOUD_COVER')

    .first());

var rgb = image.select('B7', 'B3', 'B2');

var pan = image.select('B8');

var huesat = rgb.rgbToHsv().select('hue','saturation');

var upres = ee.Image.cat(huesat,pan).hsvToRgb();
