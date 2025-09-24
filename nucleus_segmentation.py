import cv2, numpy as np
from skimage import color  as skcolor
from sklearn.cluster import KMeans

def get_all_cell_mask(img):
    """
        Általános sejtmag-maszk készítése (0/255).
        Lépések: szürkeárnyalat → Gauss-szűrés → küszöbölés → invertálás →
                 morfológiai nyitás → zárás (lyukak betöltése).
    """
    # Kép beolvasása
    #img = cv2.imread("src.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Zajszűrés + küszöbölés
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)
    # _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Invertálás (hogy a sötét sejtmagok legyenek előtérben = fehérek)
    thresh = cv2.bitwise_not(thresh)

    # # Morfológiai tisztítás: nyitás (zajszűrés), majd zárás (lyukak betöltése)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # opening_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel, iterations=2) -> ezzel sok sejtmag elveszik

    # Closing = dilatáció + erózió (lyukak betöltése)
    clean = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    return clean

def remove_blobs_overlapping_contours(all_cells: np.ndarray,
                                      cell_contours: np.ndarray,
                                      pad: int = 1) -> np.ndarray:
    # all_cells -> 0/1 bináris
    bin_all = (all_cells > 0).astype(np.uint8)

    # cell_contours -> kontúr-maszk (0/1); bármelyik csatorna > 0 számít "kontúrnak"
    # RGB esetén bármely csatorna > 0 kontúrnak számít
    if cell_contours.ndim == 3:
        contour_mask = ( (cell_contours[...,0] > 0) |
                         (cell_contours[...,1] > 0) |
                         (cell_contours[...,2] > 0) ).astype(np.uint8)
    else:
        contour_mask = (cell_contours > 0).astype(np.uint8)

    # opcionális vastagítás, hogy biztos legyen az érintkezés
    if pad > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2*pad+1, 2*pad+1))
        contour_mask = cv2.dilate(contour_mask, k, iterations=1)

    # Címkézés az all_cells maszkon
    num_labels, labels = cv2.connectedComponents(bin_all)

    # Mely label(ek) érintkeznek kontúrral?
    # (Azok a címkék, amelyek területéből legalább 1 pixel esik kontúrra)
    labels_on_contour = np.unique(labels[contour_mask.astype(bool)])
    labels_on_contour = labels_on_contour[labels_on_contour != 0]  # 0 = háttér

    # Ezen label-ek törlése az all_cells-ből
    # Érintkező komponensek eltávolítása
    to_remove = np.isin(labels, labels_on_contour)
    result = bin_all.copy()
    result[to_remove] = 0

    # Vissza 0/255 formátumban
    return (result * 255).astype(np.uint8)

def classify_from_mask(mask, D, color_map, method, km=None, remap=None, q=None, min_area=80):
    # Címkézés a bináris maszkon
    bin_mask = (mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(bin_mask)

    # Jellemzők: komponensenkénti DAB-átlag (kis régiók kihagyása)
    feats, label_ids  = [], []
    for lb in range(1, num_labels):
        m = (labels == lb)
        area = int(m.sum())
        if area < min_area:
            continue
        feats.append([float(D[m].mean())])
        label_ids .append(lb)

    overlay = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    if not label_ids :
        return {}, overlay

    feats = np.asarray(feats)

    # Kategorizálás: KMeans + remap VAGY kvantilis küszöbök (q)
    if method == "kmeans":
        raw = km.predict(feats) # ugyanazt a km + remap párost használjuk
        class_map = {lid: remap[int(c)] for lid, c in zip(label_ids, raw)}
    else:
        class_map = {
            lid: (0 if v <= q[0] else 1 if v <= q[1] else 2 if v <= q[2] else 3)
            for lid, v in zip(label_ids, feats[:, 0])
        }

    # Kontúrok kirajzolása kategória színével
    for lid in label_ids:
        m = (labels == lid).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, color_map[class_map[lid]], thickness=2, lineType=cv2.LINE_8)

    return class_map, overlay


def classify_nuclei(input_path, output_path) -> None:
    """
    Sejtmagok szegmentálása HED dekonvolúció + watershed, majd DAB-intenzitás szerinti (0..3) osztályozás
    és kontúrok kirajzolása. Eredmény a `output_path` képfájlba íródik.
    """
    method="kmeans"
    # Beolvasás és RGB-re váltás
    bgr = cv2.imread(input_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Optikai denzitás + HED szétválasztás (H=hematoxylin, E=eozin, D=DAB)
    optical_density = -np.log((rgb.astype(np.float32) + 1) / 256.0)
    hed_stains = skcolor.separate_stains(optical_density, skcolor.hed_from_rgb)
    hematoxylin, dab = hed_stains[..., 0], hed_stains[..., 2]  # Hematoxylin, DAB (pozitívabb = erősebb festés)

    # Nucleus-maszk H-ból
    # Nucleus-maszk: hematoxylin csatorna elmosás + normalizálás + Otsu küszöb
    h_blur = cv2.GaussianBlur(hematoxylin, (5, 5), 0)
    # Hn normalizálása 8 bitre
    h_u8 = cv2.normalize(h_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Otsu küszöbölés 8-bites képen
    _, thresh = cv2.threshold(h_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Bináris maszk (0 vagy 1) a további lépésekhez
    bin_mask = (thresh > 0).astype(np.uint8)

    # Morfológiai tisztítás
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 2)

    # marker-controlled watershed
    dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)
    fg = (dist > 0.4 * dist.max()).astype(np.uint8)
    bg = cv2.dilate(bin_mask, np.ones((3, 3), np.uint8), 3)
    unknown = cv2.subtract(bg, fg)
    _, markers = cv2.connectedComponents(fg)
    markers = markers + 1
    markers[unknown == 1] = 0
    markers = cv2.watershed(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), markers)

    # Per-blob DAB átlag jellemzők
    labels = np.unique(markers[(markers > 1)])
    feats, label_ids = [], []
    for label_id in labels:
        m = (markers == label_id)
        if m.sum() < 80:  # kis zaj kiszűrés
            continue
        feats.append([dab[m].mean()])  # bővíthető pl. [dab.mean(), hematoxylin.mean()]
        label_ids.append(label_id)
    feats = np.asarray(feats)

    km = None
    remap = None
    q = None

    # Osztályozás: KMeans + DAB-erősség szerinti remap, vagy kvantilis küszöbök
    if method == "kmeans":
        km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(feats)
        class_map = dict(zip(label_ids, km.labels_))
        order = np.argsort([feats[km.labels_ == i].mean() for i in range(4)])[::-1]
        remap = {int(order[i]): i for i in range(4)}
        class_map = {lid: remap[c] for lid, c in class_map.items()}
    else:
        q = np.quantile(feats[:, 0], [0.25, 0.5, 0.75])
        class_map = {
            lid: (0 if v <= q[0] else 1 if v <= q[1] else 2 if v <= q[2] else 3)
            for lid, v in zip(label_ids, feats[:, 0])
        }

    #color_map = {0:(255,0,0), 1:(0,255,255), 2:(0,165,255), 3:(0,0,255)}  # kék, sárga, narancs, piros (BGR)
    color_map = {0:(255,0,0), 1:(0,255,255), 2:(0,128,255), 3:(0,0,255)}

    # Kontúrok rajzolása
    segmented = bgr.copy()
    contours_only = np.zeros_like(bgr)
    for label_id in label_ids:
        m = (markers == label_id).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cls_id = class_map[label_id]
        color = color_map[cls_id]
        cv2.drawContours(segmented, cnts, -1, color, thickness=2, lineType=cv2.LINE_AA)
        cv2.drawContours(contours_only, cnts, -1, color, thickness=2, lineType=cv2.LINE_AA)

    # Maradék sejtek: minden-sejt maszkból vedd le a kontúrral fedetteket, majd osztályozd
    all_cells_mask = get_all_cell_mask(bgr)
    remaining_cells_mask = remove_blobs_overlapping_contours(all_cells_mask, contours_only, pad=2)

    _, rem_overlay = classify_from_mask(
        remaining_cells_mask, dab, color_map, method, km=km, remap=remap, q=q, min_area=80
    )

    # Overly összeolvasztása és mentés
    out_all = segmented.copy()
    overlay_mask = cv2.cvtColor(rem_overlay, cv2.COLOR_BGR2GRAY) > 0
    out_all[overlay_mask] = rem_overlay[overlay_mask]
    cv2.imwrite(output_path, out_all)


if __name__ == "__main__":
    input_path = "src11.jpg"
    output_path = "segmented.jpg"
    classify_nuclei(input_path, output_path)
