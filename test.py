import win32gui, win32api, win32con, win32ui, time, ctypes, cv2, ddddocr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import deque

# ========== 环境配置 ==========
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except:
    pass

matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

print("正在初始化 OCR 引擎...")
ocr = ddddocr.DdddOcr(show_ad=False)
WINDOW_NAME = "我要玩数独游戏"


# ==========================================
# 1. 窗口获取
# ==========================================
def get_window_image(window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    if not hwnd: raise Exception("❌ 未找到窗口！")
    if win32gui.IsIconic(hwnd): win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(0.2)

    l, t, r, b = win32gui.GetClientRect(hwnd)
    w, h = r - l, b - t
    hdc = win32gui.GetWindowDC(hwnd)
    mfc = win32ui.CreateDCFromHandle(hdc)
    save_dc = mfc.CreateCompatibleDC()
    bit_map = win32ui.CreateBitmap()
    bit_map.CreateCompatibleBitmap(mfc, w, h)
    save_dc.SelectObject(bit_map)
    ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)
    img = np.frombuffer(bit_map.GetBitmapBits(True), dtype='uint8')
    img.shape = (h, w, 4)
    win32gui.DeleteObject(bit_map.GetHandle())
    save_dc.DeleteDC()
    mfc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hdc)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), hwnd


# ==========================================
# 2. 精准定位 (方法二精华：HSV 蓝色背景识别)
# ==========================================
def detect_grid_and_hints(img):
    """同时精准定位：网格主体 + 顶部提示区框 + 左侧提示区框"""
    # 1. 定位网格
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    sums_v = np.sum(edges, axis=0)
    vcols = np.where(sums_v > sums_v.max() * 0.4)[0].tolist()
    sums_h = np.sum(edges, axis=1)
    hrows = np.where(sums_h > sums_h.max() * 0.4)[0].tolist()

    if not vcols or not hrows: return None
    gx, gy = vcols[0], hrows[0]
    gw, gh = vcols[-1] - gx, hrows[-1] - gy

    # 判断尺寸
    lines_x = sum(1 for i in range(1, len(vcols)) if vcols[i] - vcols[i - 1] > 10) + 1
    g_size = 15 if lines_x > 12 else 10

    # 2. 定位蓝色提示区 (完美避开游戏 UI 杂字)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue, upper_blue = np.array([100, 80, 80]), np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 默认回退边界（以防万一）
    row_hint_left = max(0, gx - 100)
    col_hint_top = max(0, gy - 100)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 行提示区：在网格左侧，且高度差不多
        if abs((x + w) - gx) < 15 and h > gh * 0.7:
            row_hint_left = x
        # 列提示区：在网格正上方，且宽度差不多
        if abs((y + h) - gy) < 15 and w > gw * 0.7:
            col_hint_top = y

    return gx, gy, gw, gh, g_size, row_hint_left, col_hint_top


# ==========================================
# 3. 图像分割与识别 (你代码的精华，一字未改)
# ==========================================
def segment_and_ocr(slice_img, direction, grid_size):
    lower_white = np.array([200, 200, 200], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(slice_img, lower_white, upper_white)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_HEIGHT] >= 6 and stats[i, cv2.CC_STAT_AREA] >= 10:
            clean_mask[labels == i] = 255

    proj = np.sum(clean_mask, axis=0) if direction == 'h' else np.sum(clean_mask, axis=1)
    blocks = []
    start = -1
    GAP_LIMIT = 3
    gap_count = 0

    for i, val in enumerate(proj):
        if val > 0:
            if start == -1: start = i
            gap_count = 0
        else:
            if start != -1:
                gap_count += 1
                if gap_count > GAP_LIMIT:
                    blocks.append((start, i - gap_count))
                    start = -1
                    gap_count = 0
    if start != -1: blocks.append((start, len(proj) - gap_count))

    results = []
    for s, e in blocks:
        roi_loose = clean_mask[:, s:e] if direction == 'h' else clean_mask[s:e, :]
        coords = cv2.findNonZero(roi_loose)
        if coords is None: continue
        x, y, w, h = cv2.boundingRect(coords)
        roi_tight = roi_loose[y:y + h, x:x + w]
        aspect_ratio = w / h

        padded = cv2.copyMakeBorder(roi_tight, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=0)
        inverted = cv2.bitwise_not(padded)
        inverted = cv2.resize(inverted, (inverted.shape[1] * 3, inverted.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
        _, img_bytes = cv2.imencode('.png', inverted)
        res = ocr.classification(img_bytes.tobytes())

        correction_map = {'o': '0', 'O': '0', 'D': '0', 'Q': '0', 'l': '1', 'I': '1', 's': '5', 'S': '5', 'B': '8',
                          'g': '9', 'A': '4'}
        corrected = "".join([correction_map.get(c, c) for c in res])
        num_str = ''.join(filter(str.isdigit, corrected))

        if num_str:
            val = int(num_str)
            if val == 0 or val == 8:
                mid_band = roi_tight[int(h * 0.4):int(h * 0.6), :]
                density = cv2.countNonZero(mid_band) / (w * mid_band.shape[0] + 1)
                val = 8 if density > 0.4 else 0

            if aspect_ratio > 0.85 and val in [0, 1]:
                val = 10

            if val > grid_size: val = int(str(val)[0])
            results.append(val)

    return results if results else [0]


def extract_all_clues_auto(image, gx, gy, gw, gh, g_size, row_hint_left, col_hint_top):
    cell_w, cell_h = gw / g_size, gh / g_size
    t_clues, l_clues = [], []

    for i in range(g_size):
        x_start = int(gx + i * cell_w)
        x_end = int(gx + (i + 1) * cell_w)
        # 精确切割：从检测到的顶部边缘 col_hint_top 切到网格线 gy
        roi = image[col_hint_top:gy, x_start:x_end]
        t_clues.append(segment_and_ocr(roi, 'v', g_size))

    for i in range(g_size):
        y_start = int(gy + i * cell_h)
        y_end = int(gy + (i + 1) * cell_h)
        # 精确切割：从检测到的左侧边缘 row_hint_left 切到网格线 gx
        roi = image[y_start:y_end, row_hint_left:gx]
        l_clues.append(segment_and_ocr(roi, 'h', g_size))

    return t_clues, l_clues


# ==========================================
# 4. 你的极速求解算法
# ==========================================
def get_line_masks(clues, length):
    if not clues or sum(clues) == 0: return [0]
    if len(clues) == 1 and clues[0] == length: return [(1 << length) - 1]
    memo = {}

    def build(c_idx, cur_len):
        state = (c_idx, cur_len)
        if state in memo: return memo[state]
        if c_idx == len(clues): return [0]
        res = []
        c = clues[c_idx]
        min_needed = sum(clues[c_idx:]) + (len(clues) - c_idx - 1)
        for sp in range(length - cur_len - min_needed + 1):
            current_bits = ((1 << c) - 1) << (length - cur_len - sp - c)
            if c_idx < len(clues) - 1:
                suffixes = build(c_idx + 1, cur_len + sp + c + 1)
                for s in suffixes: res.append(current_bits | s)
            else:
                res.append(current_bits)
        memo[state] = res
        return res

    return build(0, 0)


def solve_nonogram_fast(row_clues, col_clues, size):
    row_poss = [get_line_masks(r, size) for r in row_clues]
    col_poss = [get_line_masks(c, size) for c in col_clues]
    row_must_black, row_must_white = [0] * size, [0] * size
    col_must_black, col_must_white = [0] * size, [0] * size
    queue = deque([('r', i) for i in range(size)] + [('c', i) for i in range(size)])
    in_queue = set(queue)

    while queue:
        mode, idx = queue.popleft()
        in_queue.remove((mode, idx))
        if mode == 'r':
            current_black, current_white = 0, 0
            for c in range(size):
                if (col_must_black[c] >> (size - 1 - idx)) & 1: current_black |= (1 << (size - 1 - c))
                if (col_must_white[c] >> (size - 1 - idx)) & 1: current_white |= (1 << (size - 1 - c))
            new_poss = [p for p in row_poss[idx] if (p & current_black) == current_black and (p & current_white) == 0]
            if not new_poss: return None
            row_poss[idx] = new_poss
            nb = new_poss[0]
            nw = ~new_poss[0] & ((1 << size) - 1)
            for p in new_poss[1:]:
                nb &= p
                nw &= ~p
            if nb != row_must_black[idx] or nw != row_must_white[idx]:
                row_must_black[idx], row_must_white[idx] = nb, nw
                for c in range(size):
                    if not (('c', c) in in_queue):
                        queue.append(('c', c))
                        in_queue.add(('c', c))
        else:
            current_black, current_white = 0, 0
            for r in range(size):
                if (row_must_black[r] >> (size - 1 - idx)) & 1: current_black |= (1 << (size - 1 - r))
                if (row_must_white[r] >> (size - 1 - idx)) & 1: current_white |= (1 << (size - 1 - r))
            new_poss = [p for p in col_poss[idx] if (p & current_black) == current_black and (p & current_white) == 0]
            if not new_poss: return None
            col_poss[idx] = new_poss
            nb = new_poss[0]
            nw = ~new_poss[0] & ((1 << size) - 1)
            for p in new_poss[1:]:
                nb &= p
                nw &= ~p
            if nb != col_must_black[idx] or nw != col_must_white[idx]:
                col_must_black[idx], col_must_white[idx] = nb, nw
                for r in range(size):
                    if not (('r', r) in in_queue):
                        queue.append(('r', r))
                        in_queue.add(('r', r))
    res = np.zeros((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            if (row_must_black[r] >> (size - 1 - c)) & 1: res[r][c] = 1
    return res


# ==========================================
# 5. 主程序
# ==========================================
def main():
    try:
        img, hwnd = get_window_image(WINDOW_NAME)
    except Exception as e:
        print(e)
        return

    grid_info = detect_grid_and_hints(img)
    if not grid_info:
        return print("❌ 未能识别出网格。")

    gx, gy, gw, gh, g_size, row_left, col_top = grid_info
    print(f"✅ 定位成功！模式: {g_size}x{g_size}")

    t_clues, l_clues = extract_all_clues_auto(img, gx, gy, gw, gh, g_size, row_left, col_top)

    print(f"\n提取结果:\n顶部: {t_clues}\n左侧: {l_clues}")

    if input("\n线索无误请回车开始秒解，有误输入 'n'：").lower() == 'n': return

    start_time = time.time()
    res = solve_nonogram_fast(l_clues, t_clues, g_size)

    if res is not None:
        print(f"✅ 求解成功！耗时: {time.time() - start_time:.4f}s")
        plt.imshow(res, cmap='gray')
        plt.title("极速版结果确认 - 请手动关闭此窗口以开始自动填涂")
        plt.show()

        cw, ch = gw / g_size, gh / g_size
        print("👉 开始自动填涂...")
        for r in range(g_size):
            for c in range(g_size):
                if res[r][c] == 1:
                    lp = win32api.MAKELONG(int(gx + c * cw + cw / 2), int(gy + r * ch + ch / 2))
                    win32gui.SendMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lp)
                    win32gui.SendMessage(hwnd, win32con.WM_LBUTTONUP, 0, lp)
                    time.sleep(0.015)
        print("🎉 秒杀完成！")
    else:
        print("❌ 逻辑矛盾。")


if __name__ == "__main__":
    main()