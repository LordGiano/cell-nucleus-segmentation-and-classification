import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ez a függvény arra szolgál, hogy a teljes sejtmagokat tartalmazó maszkot visszaadja
def get_full_nucles_mask(input_path):
    img = cv2.imread("src.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Zajszűrés + küszöbölés
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 190, 255, cv2.THRESH_BINARY_INV)

    #thresh = cv2.bitwise_not(thresh)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    clean = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    masked_img = cv2.bitwise_and(img, img, mask=clean)
    cv2.imshow("Maszkolt sejtmagok", masked_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def nucleus_segmentation_old(input_path, output_path):
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.35 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    cv2.imshow("markers", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plan():
    """
    1. Teljes sejtmag maszk előállítása ✅️
    2. Sejtmagokat elválasztó vonalakat tartalmazó (watershed) maszk előállítása
    3. A teljes maszkból a sejtmagokat elválasztó maszkon található értékek kivonása
    (Ezt követően tisztítás, open, stb)
    4. Színes kép előállítása a 3. lépésben előállított maszk alapján
    5. A színes képről valamilyen statisztika/osztályozás készítése (blobokon belüli átlag szín számítás, hisztogram, KNN, stb)
    6. Az előállított osztályok alapján a sejtmagok körüli részek színezése
    :return:
    """

def cell_classification(input_path, output_path):
    line_color = (0, 0, 255)
    line_thickness = 3

    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret_full, thresh_full = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    ret_red, thresh_red = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening_full = cv2.morphologyEx(thresh_full, cv2.MORPH_OPEN, kernel, iterations=2)
    opening_red = cv2.morphologyEx(thresh_red, cv2.MORPH_OPEN, kernel, iterations=2)

    # cell mask
    full_cell_mask = cv2.morphologyEx(opening_full, cv2.MORPH_CLOSE, kernel, iterations=2)
    masked_img = cv2.bitwise_and(img, img, mask=full_cell_mask)
    #cv2.imshow("Teljes", masked_img)
    #cv2.imshow("Maszkolt sejtmagok", clean_full)

    # sure background area
    sure_bg = cv2.dilate(opening_red, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening_red, cv2.DIST_L2, 5)
    ret_red, sure_fg = cv2.threshold(dist_transform, 0.35 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret_red, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)

    boundary = np.uint8(markers == -1)

    if line_thickness > 1:
        dilate_kernel = np.ones((line_thickness, line_thickness), np.uint8)
        boundary = cv2.dilate(boundary, dilate_kernel, iterations=1)

    img[boundary == 1] = line_color  # színezés
    #masked_img[boundary == 1] = line_color # színezés
    masked_img[boundary == 1] = (0,0,0)

    #img[markers == -1] = [255, 0, 0] # eredeti
    #cv2.imshow("markers", img)
    cv2.imwrite("markers.jpg", img)
    cv2.imwrite("masked_markers.jpg", masked_img)

    # --- BLOB ÁTLAG SZÍNEK ---
    gray_masked = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    _, bin_mask = cv2.threshold(gray_masked, 1, 255, cv2.THRESH_BINARY)

    num_labels, labels = cv2.connectedComponents(bin_mask)

    rgb_means = []
    gray_means = []

    for i in range(1, num_labels):  # 0 = háttér
        mask = np.uint8(labels == i)
        mean_val = cv2.mean(masked_img, mask=mask)  # (B, G, R, alpha)
        gray_val = cv2.mean(gray_masked, mask=mask)[0]

        rgb_means.append((mean_val[2], mean_val[1], mean_val[0]))  # átrendezve (R,G,B)
        gray_means.append(gray_val)

    # --- HISZTROGRAMOK ---
    rgb_means = np.array(rgb_means)
    gray_means = np.array(gray_means)

    plt.figure(figsize=(12, 5))

    # RGB hisztogram
    plt.subplot(1, 2, 1)
    plt.hist(rgb_means[:, 0], bins=20, color="r", alpha=0.5, label="Red")
    plt.hist(rgb_means[:, 1], bins=20, color="g", alpha=0.5, label="Green")
    plt.hist(rgb_means[:, 2], bins=20, color="b", alpha=0.5, label="Blue")
    plt.title("Átlag RGB értékek hisztogramja")
    plt.xlabel("Intenzitás")
    plt.ylabel("Darabszám")
    plt.legend()

    # Szürkeárnyalatos hisztogram
    plt.subplot(1, 2, 2)
    plt.hist(gray_means, bins=20, color="gray", alpha=0.7)
    plt.title("Átlag szürkeárnyalat értékek hisztogramja")
    plt.xlabel("Intenzitás")
    plt.ylabel("Darabszám")

    plt.tight_layout()
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_imgs():
    img = cv2.imread("src.jpg")
    h, w = img.shape[:2]

    # Fél szélesség és magasság
    half_h, half_w = h // 2, w // 2

    # 4 darabra vágás
    top_left = img[0:half_h, 0:half_w]
    top_right = img[0:half_h, half_w:w]
    bottom_left = img[half_h:h, 0:half_w]
    bottom_right = img[half_h:h, half_w:w]

    # Mentés
    cv2.imwrite("src1.jpg", top_left)  # bal felső
    cv2.imwrite("src2.jpg", top_right)  # jobb felső
    cv2.imwrite("src3.jpg", bottom_left)  # bal alsó
    cv2.imwrite("src4.jpg", bottom_right)  # jobb alsó


if __name__ == "__main__":
    input_path = "src1.jpg"
    output_path = "segmented.jpg"
    #nucleus_segmentation(input_path, output_path)
    #get_full_nucles_mask(input_path)
    cell_classification(input_path, output_path)
    #create_imgs()