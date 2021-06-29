import Image
import shapely, geopandas
from shapely import geometry
from osgeo import ogr, osr, gdal
import numpy as np
import os
import shapefile
from tqdm import tqdm


class TIFF_READ:
    def __init__(self, tiff_path):
        self.path = tiff_path

        self.dataset = gdal.Open(self.path)
        self.X = self.dataset.RasterXSize # 网格的X轴像素数量
        self.Y = self.dataset.RasterYSize # 网格的Y轴像素数量
        self.GeoTransform = self.dataset.GetGeoTransform()  # 投影转换信息
        self.ProjectionInfo = self.dataset.GetProjection()  # 投影信息

    def get_band(self,band):
        """
        获取波段信息
        :param band:波段
        :return: np.array(band)
        """
        band = self.dataset.GetRasterBand(band)
        data = band.ReadAsArray()
        return data

    def get_lon_lat(self,lon_path,lat_path):
        """
        读取经纬度信息
        :return:
        """
        gtf = self.GeoTransform
        x_range = range(0, self.X)
        y_range = range(0, self.Y)
        x, y = np.meshgrid(x_range, y_range)
        lon = gtf[0] + x * gtf[1] + y * gtf[2]
        lat = gtf[3] + x * gtf[4] + y * gtf[5]
        np.save(lon_path,lon)
        np.save(lat_path,lat)
        return lon, lat

    def transform(self,R,G,B,save_path):
        data=np.dstack((R,G,B))
        im=Image.fromarray(data.astype("uint8"))
        im.save(save_path)
        return


class SHP_READ:
    def __init__(self,shp_path:str):
        self.shp=shp_path

    def read_shp(self):
        shp_df = shapefile.Reader(self.shp,encoding='gb18030')
        shapes = shp_df.shapes()
        for i in tqdm(shapes):
            geo = i.points
            print(geo,"\n")
        return


if __name__ == '__main__':
    tif_file = "shp/tif/ans2321.tif"
    jpg_path="answer.jpg"
    shp_file = "shp/shp/get_square.shp"
    lon_txt="232_lon"
    lat_txt="232_lat"

    # 读取tiff转换成为jpg
    TIFF=TIFF_READ(tiff_path=tif_file)
    u=TIFF.dataset
    R=TIFF.get_band(1)
    # lon,lat=TIFF.get_lon_lat(lon_txt,lat_txt)
    TIFF.transform(R,R,R,jpg_path)

    SHP=SHP_READ(shp_file)
    SHP.read_shp()


