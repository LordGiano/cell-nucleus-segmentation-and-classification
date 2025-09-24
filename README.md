# Sejtmag szegmentálás és osztályozás

Ez a projekt egy Python szkriptet tartalmaz **sejtmagok szegmentálására és osztályozására** digitalizált szövetrészleteken. A szkript a következő fő lépéseket hajtja végre:

1. **Festés szétválasztása (HED dekonvolúció)** – az RGB képet három független csatornára bontja szét:
   - **Hematoxylin (H)** – kiemeli a **kékes-lilás területeket**, amelyek a sejtmagokhoz köthetők. Ez adja a legjobb kontrasztot a magok és a háttér között, ezért a szegmentálás ebből készül.
   - **Eosin (E)** – elkülöníti a **rózsaszínes, világosabb területeket**, amelyek inkább a háttérhez tartoznak. Ebben a feladatban nem használjuk.
   - **DAB (D)** – kiemeli a **barna árnyalatokat**, amelyek a feladat szempontjából a „barnaság” mértékét jelentik. Ez a klasszifikáció alapja.

   A H csatorna tehát a **szegmentáláshoz** szükséges információt adja (maszk készítése, watershed), a D csatorna pedig az **osztályozáshoz** szükséges jellemzőt (átlagintenzitás minden magra).

2. **Sejtmag szegmentálás** – a H csatornából készített képen:
   - Gauss-szűrés, normalizálás,
   - Otsu-küszöbölés,
   - morfológiai nyitás (zajcsökkentés),
   - marker-controlled watershed (a szomszédos magok szétválasztására).

3. **Osztályozás** – minden detektált sejtmag D csatorna szerinti intenzitása alapján kerül négy kategóriába:
   - **0 – kék** – leggyengébb barna intenzitás,
   - **1 – sárga**, 
   - **2 – narancs**, 
   - **3 – piros** – legerősebb barna intenzitás.

   Az osztályozás kétféleképpen történhet:
   - **KMeans klaszterezéssel**, majd az intenzitás sorrendjének újrarendezésével,
   - vagy **kvantilis küszöbökkel**, ha a klaszterezés nem stabil.

4. **Utófeldolgozás (Remaining Cells)** – a watershed esetleg kihagyhat sejteket. Ezért:
   - készül egy globális bináris maszk az összes sejtre,
   - eltávolítjuk belőle azokat, amelyek már szerepelnek a watershed eredményében,
   - a maradék komponenseket újraosztályozzuk a D csatorna alapján.

5. **Vizualizáció** – az összes detektált és osztályozott sejtmag körvonala **2 pixeles vastagsággal**, a kategóriának megfelelő színnel jelenik meg a kimeneti képen.

---

## Telepítés

Virtuális környezet javasolt:

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

## Használat

Futtatás:

```bash
python nucleus_segmentation.py
```

Alapértelmezés szerint a `src11.jpg` fájlt dolgozza fel, és az eredményt a `segmented.jpg` fájlba menti.

Saját képekhez módosítsd a fájl végét:

```python
if __name__ == "__main__":
    input_path = "sajat_bemenet.jpg"
    output_path = "eredmeny.jpg"
    classify_nuclei(input_path, output_path)
```

## Követelmények

- Python 3.8+
- OpenCV
- NumPy
- scikit-image
- scikit-learn

## Kimenet példája

Az eredmény egy kép, amelyen minden detektált sejtmag körvonallal van kiemelve, színezve a „barnaság” erőssége szerint:

- **Kék (0)** – leggyengébb,
- **Sárga (1)**,
- **Narancs (2)**,
- **Piros (3)** – legerősebb.

## Megjegyzések

- A HED dekonvolúció biztosítja, hogy a sejtmagokat és a barna intenzitást külön csatornákon tudjuk feldolgozni, így pontosabb a szegmentálás és az osztályozás.
- A kontúrvastagság alapértelmezés szerint 2 pixel, de a kódban állítható.
- A paraméterek (kernel méretek, küszöbértékek) a dataset függvényében módosíthatók.
- A szkript tartalmazza a „Remaining Cells” második osztályozási fázist is, amely javítja a lefedettséget.
