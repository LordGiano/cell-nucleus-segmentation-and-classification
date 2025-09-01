import cv2, numpy as np
from skimage import color
from sklearn.cluster import KMeans

def classify_nuclei(input_path, output_path, method="kmeans"):
    print(method)
    bgr = cv2.imread(input_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 1) OD + HED dekonvolúció
    img_od = -np.log((rgb.astype(np.float32)+1)/256.0)
    hed = color.separate_stains(img_od, color.hed_from_rgb)  # H, E, D
    H, D = hed[...,0], hed[...,2]  # Hematoxylin, DAB (pozitívabb = erősebb festés)

    # 2) Nucleus-maszk H-ból
    Hn = cv2.GaussianBlur(H, (5,5), 0)
    # Hn normalizálása 8 bitre
    Hn_u8 = cv2.normalize(Hn, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Otsu küszöbölés 8-bites képen
    _, th = cv2.threshold(Hn_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Bináris maszk (0 vagy 1) a további lépésekhez
    binmask = (th > 0).astype(np.uint8)

    # Morfológiai tisztítás
    binmask = cv2.morphologyEx(binmask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), 2)

    # marker-controlled watershed
    dist = cv2.distanceTransform(binmask, cv2.DIST_L2, 5)
    fg = (dist > 0.4*dist.max()).astype(np.uint8)
    bg = cv2.dilate(binmask, np.ones((3,3),np.uint8), 3)
    unknown = cv2.subtract(bg, fg)
    n, markers = cv2.connectedComponents(fg)
    markers = markers + 1
    markers[unknown==1] = 0
    markers = cv2.watershed(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), markers)

    # 3) Per-blob DAB átlag
    labels = np.unique(markers[(markers>1)])
    feats, idx = [], []
    for lb in labels:
        m = (markers==lb)
        if m.sum() < 80:    # kis zaj kiszűrés
            continue
        feats.append([D[m].mean()])  # vagy több jellemző: [D.mean(), H.mean()]
        idx.append(lb)
    feats = np.asarray(feats)

    # 4) Négy kategória
    if method == "kmeans":
        km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(feats)
        cats = dict(zip(idx, km.labels_))
        # rendezzük a klasztereket DAB erősség szerint
        order = np.argsort([feats[km.labels_==i].mean() for i in range(4)])[::-1]
        remap = {int(order[i]):i for i in range(4)}
        cats = {lb: remap[c] for lb,c in cats.items()}
    else:
        # kvantilis küszöbök
        q = np.quantile(feats[:,0], [0.25, 0.5, 0.75])
        cats = {lb: (0 if v<=q[0] else 1 if v<=q[1] else 2 if v<=q[2] else 3)
                for lb, v in zip(idx, feats[:,0])}

    # 5) Outline kirajzolás
    color_map = {0:(255,0,0), 1:(0,255,255), 2:(0,165,255), 3:(0,0,255)}  # kék, sárga, narancs, piros (BGR)
    out = bgr.copy()
    for lb in idx:
        m = (markers==lb).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, color_map[cats[lb]], thickness=2)
    cv2.imwrite(output_path, out)

if __name__ == "__main__":
    input_path = "src1.jpg"
    output_path = "segmented.jpg"
    classify_nuclei(input_path, output_path)
