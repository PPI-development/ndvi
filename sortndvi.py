import tkinter as tk
from tkinter import filedialog, messagebox

import rasterio
from rasterio import features
import numpy as np
from scipy.ndimage import gaussian_filter
from shapely.geometry import shape, LineString, Polygon, Point, MultiLineString
from shapely.geometry import shape, LineString, Polygon, Point, MultiPolygon

import geopandas as gpd
import simplekml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import PolygonSelector
import pyproj

class NDVIRoutePlanner(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NDVI Route Planner")
        self.geometry("900x700")

        top = tk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        tk.Button(top, text="Открыть NDVI GeoTIFF", command=self.open_file).pack(side=tk.LEFT, padx=5)
        self.select_btn = tk.Button(top, text="Выбрать зону", command=self.toggle_selector, state=tk.DISABLED)
        self.select_btn.pack(side=tk.LEFT, padx=5)
        self.save_btn = tk.Button(top, text="Сохранить KML/KMZ", command=self.save_file, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        lf = tk.Frame(self)
        lf.pack(side=tk.BOTTOM, fill=tk.X)
        self.log_text = tk.Text(lf, height=6)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(lf, command=self.log_text.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=sb.set)
        self.log("Программа запущена. Загрузите NDVI GeoTIFF.")

        self.polygons_gdf = None
        self.polygons_wgs = None
        self.routes_by_poly = []
        self.selector = None
        self.selection_poly = None
        self.full_route = []

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        print(msg)

    def open_file(self):
        path = filedialog.askopenfilename(title="Выберите NDVI GeoTIFF",
                                          filetypes=[("GeoTIFF", "*.tif *.tiff"), ("Все файлы", "*.*")])
        if not path:
            return
        self.log(f"Открытие: {path}")

        with rasterio.open(path) as src:
            ndvi = src.read(1, masked=True).astype("float32")
            crs = src.crs
            transform = src.transform

        data = ndvi.copy()
        if hasattr(data, "mask"):
            mx = data[~data.mask].max()
        else:
            mx = data.max()
        if mx > 1:
            factor = 100.0 if mx <= 100 else 10000.0
            self.log(f"Масштабирование NDVI делением на {factor}")
            data /= factor

        self.log("Сглаживание NDVI фильтром Гаусса (sigma=5)...")
        data = gaussian_filter(data, sigma=5)

        self.log("Классификация NDVI...")
        classified = np.zeros(data.shape, np.uint8)
        valid = ~ndvi.mask if hasattr(ndvi, "mask") else np.ones_like(data, bool)
        classified[valid & (data >= 0.2) & (data < 0.4)] = 2
        classified[valid & (data >= 0.4) & (data < 0.6)] = 3
        classified[valid & (data >= 0.6)] = 4

        self.log("Векторизация классов 2–4...")
        polys, classes = [], []
        for geom, v in features.shapes(classified, mask=classified > 1, transform=transform):
            if v < 2: continue
            poly = shape(geom).buffer(0)
            if poly.is_empty: continue
            if isinstance(poly, Polygon):
                polys.append(poly); classes.append(int(v))
            else:
                for part in poly.geoms:
                    polys.append(part); classes.append(int(v))

        if not polys:
            messagebox.showinfo("Результат", "Нет зон увядания.")
            return

        self.polygons_gdf = gpd.GeoDataFrame({"class": classes, "geometry": polys}, crs=crs)
        self.polygons_gdf = self.polygons_gdf[self.polygons_gdf["class"] != 4]
        if self.polygons_gdf.empty:
            messagebox.showinfo("Результат", "Нет зон обработки.")
            return

        if self.polygons_gdf.crs.is_geographic:
            b = self.polygons_gdf.total_bounds
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            zone = int((cx + 180) // 6) + 1
            utm_crs = f"EPSG:{32600 + zone if cy >= 0 else 32700 + zone}"
            self.log(f"Перевод в {utm_crs}")
            self.polygons_gdf = self.polygons_gdf.to_crs(utm_crs)

        self.log("Сглаживание границ (buffer→simplify→buffer)...")
        buf, simp = 10.0, 5.0
        self.polygons_gdf["geometry"] = (
            self.polygons_gdf.geometry.buffer(buf).simplify(simp, preserve_topology=True).buffer(-buf))

        self.routes_by_poly = [
            r for r in (self.build_zigzag(poly, step=3.0) for poly in self.polygons_gdf.geometry) if r
        ]

        self.polygons_wgs = self.polygons_gdf.to_crs("EPSG:4326")
        self.plot_preview()
        self.select_btn.config(state=tk.NORMAL)
        self.log("Предпросмотр готов. Нажмите «Выбрать зону».")

    def build_zigzag(self, poly: Polygon, step: float):
        segments = []
        minx, miny, maxx, maxy = poly.bounds
        y = miny
        while y <= maxy:
            ln = LineString([(minx, y), (maxx, y)])
            inter = poly.intersection(ln)
            if not inter.is_empty:
                if isinstance(inter, MultiLineString):
                    segments.extend(inter.geoms)
                elif isinstance(inter, LineString):
                    segments.append(inter)
            y += step
        if not segments:
            return []

        used = [False] * len(segments)
        idx0 = min(range(len(segments)), key=lambda i: segments[i].centroid.y)
        route = list(segments[idx0].coords)
        used[idx0] = True
        cur = Point(route[-1])

        for _ in range(len(segments) - 1):
            best_j, best_d, rev = None, float('inf'), False
            for j, seg in enumerate(segments):
                if used[j]:
                    continue
                pts = list(seg.coords)
                d0 = cur.distance(Point(pts[0]))
                d1 = cur.distance(Point(pts[-1]))
                if d0 < best_d:
                    best_d, best_j, rev = d0, j, False
                if d1 < best_d:
                    best_d, best_j, rev = d1, j, True
            used[best_j] = True
            pts = list(segments[best_j].coords)
            if rev:
                pts = pts[::-1]
            route.append((cur.x, cur.y))
            route.append(pts[0])
            route.extend(pts)
            cur = Point(pts[-1])

        return route

    def plot_preview(self):
        self.ax.clear()
        cols = {2: "#ffff70", 3: "#ffa500"}
        for geom, cls in zip(self.polygons_wgs.geometry, self.polygons_wgs["class"]):
            if isinstance(geom, Polygon):
                geoms = [geom]
            elif isinstance(geom, MultiPolygon):
                geoms = geom.geoms
            else:
                continue

            for g in geoms:
                arr = np.array(g.exterior.coords)
                self.ax.add_patch(MplPolygon(arr, fill=False, edgecolor=cols[cls], lw=1.2))

        b = self.polygons_wgs.total_bounds
        self.ax.set_xlim(b[0], b[2])
        self.ax.set_ylim(b[1], b[3])
        self.ax.set_aspect('equal', 'box')
        self.ax.set_title("Выбери область (произвольный полигон)")
        self.canvas.draw()

    def toggle_selector(self):
        if self.selector is None:
            self.selector = PolygonSelector(
                self.ax, self.on_select,
                useblit=True,
                props=dict(color='red', linestyle='--', linewidth=1),
                handle_props=dict(marker='o', markersize=5,
                                  markerfacecolor='red', markeredgecolor='red')
            )
            self.select_btn.config(text="Завершить выбор")
        else:
            self.selector.disconnect_events()
            self.selector = None
            self.select_btn.config(text="Выбрать зону")

    def on_select(self, verts):
        self.selection_poly = Polygon(verts)
        self.log("Пользовательский полигон выбран.")
        if self.selector:
            self.selector.disconnect_events()
            self.selector = None
            self.select_btn.config(text="Выбрать зону")
        self.apply_selection()

    def apply_selection(self):
        t2utm = pyproj.Transformer.from_crs("EPSG:4326", self.polygons_gdf.crs, always_xy=True)
        utm_verts = [t2utm.transform(x, y) for x, y in self.selection_poly.exterior.coords]
        sel_poly_utm = Polygon(utm_verts)

        mask = self.polygons_gdf.intersects(sel_poly_utm)
        idxs = np.where(mask)[0]
        if not len(idxs):
            self.log("Нет зон в выбранном полигоне.")
            return
        self.log(f"Зон для обработки: {len(idxs)}")

        utm_routes = []
        for i in idxs:
            clipped = self.polygons_gdf.geometry.iat[i].intersection(sel_poly_utm)
            if clipped.is_empty:
                continue
            pts = self.build_zigzag(clipped, step=3.0)
            if pts:
                utm_routes.append((i, pts))

        if not utm_routes:
            self.log("Нет маршрутов внутри зон.")
            return

        t2geo = pyproj.Transformer.from_crs(self.polygons_gdf.crs, "EPSG:4326", always_xy=True)
        full = []
        last = None
        for i, pts in utm_routes:
            geo = [t2geo.transform(x, y) for x, y in pts]
            if not full:
                full.extend(geo)
            else:
                full.append(last)
                full.append(geo[0])
                full.extend(geo)
            last = geo[-1]
        self.full_route = full

        self.ax.clear()
        for i, _ in utm_routes:
            arr = np.array(self.polygons_wgs.geometry.iat[i].exterior.coords)
            self.ax.add_patch(MplPolygon(arr, fill=False, edgecolor='blue', lw=1.5))
        if full:
            lon, lat = zip(*full)
            self.ax.plot(lon, lat, 'k-', lw=1.3)
        bb = self.polygons_wgs.total_bounds
        self.ax.set_xlim(bb[0], bb[2]); self.ax.set_ylim(bb[1], bb[3])
        self.ax.set_aspect('equal', 'box')
        self.ax.set_title("Обработанные зоны и маршрут")
        self.canvas.draw()

        self.save_btn.config(state=tk.NORMAL)
        self.log("Маршрут построен.")

    def save_file(self):
        if not self.full_route:
            messagebox.showwarning("Нет данных", "Сначала выберите зону и постройте маршрут.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".kml",
                                           filetypes=[("KML", "*.kml"), ("KMZ", "*.kmz")])
        if not out:
            return

        kml = simplekml.Kml()
        for geom, cls in zip(self.polygons_wgs.geometry, self.polygons_gdf["class"]):
            if not geom.intersects(self.selection_poly):
                continue
            pol = kml.newpolygon(name=f"Zone {cls}")
            pol.outerboundaryis = list(geom.exterior.coords)
            if geom.interiors:
                pol.innerboundaryis = [list(h.coords) for h in geom.interiors]
            pol.style.polystyle.fill = 0
            pol.style.linestyle.width = 2
            pol.style.linestyle.color = {2: simplekml.Color.yellow, 3: simplekml.Color.orange}[cls]

        ls = kml.newlinestring(name="Drone Route")
        ls.coords = self.full_route
        ls.style.linestyle.color = simplekml.Color.green
        ls.style.linestyle.width = 3

        try:
            if out.lower().endswith(".kmz"):
                kml.savekmz(out)
            else:
                kml.save(out)
            self.log(f"Сохранено: {out}")
            messagebox.showinfo("Готово", f"Файл сохранён:\n{out}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))


if __name__ == "__main__":
    app = NDVIRoutePlanner()
    app.mainloop()
