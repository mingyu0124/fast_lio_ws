#!/usr/bin/env python3
"""
data 폴더의 PCD 파일을 시각화하는 스크립트 (matplotlib / plotly HTML)
사용법: python3 view_pcd.py [파일경로]   (기본값: ./test.pcd)

HTML 시각화 시 x,y,z 축 범위 설정 (None이면 자동):
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PCD = os.path.join(SCRIPT_DIR, "test_global.pcd")

# 축 범위: None = 자동, [최소, 최대] = 고정 (예: [-10, 10])
AXIS_X_RANGE = [-10.0, 10.0]  # 예: [-5.0, 5.0]
AXIS_Y_RANGE = [-10.0, 10.0]  # 예: [-5.0, 5.0]
AXIS_Z_RANGE = [-10.0, 0.5]  # 예: [-1.0, 2.0]


def read_pcd_simple(path):
    """PCD 파일에서 x,y,z 포인트만 읽기 (ASCII/binary 모두 시도)"""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header.append(line)
            if line.startswith("DATA"):
                break
        # FIELDS에서 x,y,z 위치 확인
        fields_line = [l for l in header if l.startswith("FIELDS ")][0]
        fields = fields_line.split()[1:]
        try:
            x_idx = fields.index("x")
            y_idx = fields.index("y")
            z_idx = fields.index("z")
        except ValueError:
            x_idx, y_idx, z_idx = 0, 1, 2
        num_fields = len(fields)
        # POINTS
        points_line = [l for l in header if l.startswith("POINTS ")][0]
        n_points = int(points_line.split()[1])
        # DATA ascii / binary
        if "DATA ascii" in " ".join(header):
            import numpy as np
            data = []
            for _ in range(n_points):
                line = f.readline().decode("ascii")
                parts = line.split()
                if len(parts) >= 3:
                    data.append([float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])])
            return np.array(data) if data else None
        else:
            # binary
            import numpy as np
            size_line = [l for l in header if l.startswith("SIZE ")][0]
            sizes = list(map(int, size_line.split()[1:]))
            type_line = [l for l in header if l.startswith("TYPE ")][0]
            types = type_line.split()[1:]
            dtype_map = {"F": np.float32, "I": np.int32, "U": np.uint8}
            row_size = sum(sizes)
            buf = f.read(n_points * row_size)
            # 각 필드 시작 오프셋 (바이트)
            offsets = [sum(sizes[:k]) for k in range(num_fields)]
            xs, ys, zs = [], [], []
            for i in range(n_points):
                base = i * row_size
                x = np.frombuffer(buf[base + offsets[x_idx] : base + offsets[x_idx] + sizes[x_idx]], dtype=dtype_map.get(types[x_idx], np.float32))[0]
                y = np.frombuffer(buf[base + offsets[y_idx] : base + offsets[y_idx] + sizes[y_idx]], dtype=dtype_map.get(types[y_idx], np.float32))[0]
                z = np.frombuffer(buf[base + offsets[z_idx] : base + offsets[z_idx] + sizes[z_idx]], dtype=dtype_map.get(types[z_idx], np.float32))[0]
                xs.append(x); ys.append(y); zs.append(z)
            return np.column_stack([xs, ys, zs])
    return None


def read_pcd_fallback(path):
    """간단한 ASCII PCD 파서 (DATA ascii 전제)"""
    import numpy as np
    with open(path, "r") as f:
        lines = f.readlines()
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("DATA ascii"):
            data_start = i + 1
            break
    points = []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) >= 3:
            try:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    return np.array(points) if points else None


def view_with_matplotlib(xyz, title="PCD"):
    import numpy as np
    n = len(xyz)
    if n > 100000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=100000, replace=False)
        xyz = xyz[idx]
        print(f"(포인트가 많아 100000개만 표시합니다. 원본: {n}개)")

    # 1) plotly로 3D HTML 생성 (matplotlib/GTK 로드 없이 경고 제거)
    try:
        import plotly.express as px
        import pandas as pd
        n_show = min(50000, len(xyz))
        df = pd.DataFrame(xyz[:n_show], columns=["x", "y", "z"])
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="z", title=title)
        fig.update_traces(marker=dict(size=1.0, opacity=0.7))
        # x,y,z 축 범위 설정 (AXIS_*_RANGE)
        scene = {}
        if AXIS_X_RANGE is not None:
            scene["xaxis"] = dict(range=AXIS_X_RANGE)
        if AXIS_Y_RANGE is not None:
            scene["yaxis"] = dict(range=AXIS_Y_RANGE)
        if AXIS_Z_RANGE is not None:
            scene["zaxis"] = dict(range=AXIS_Z_RANGE)
        if scene:
            fig.update_layout(scene=scene)
        html_path = os.path.join(SCRIPT_DIR, "pcd_preview.html")
        fig.write_html(html_path)
        print(f"미리보기: {html_path}")
        try:
            fig.show()
        except Exception:
            pass
        return
    except ImportError:
        print("plotly 없음: pip install plotly pandas")
    except Exception as e:
        print(f"plotly 실패: {e}")

    # 2) 2D (XY) PNG로 저장
    try:
        import matplotlib
        matplotlib.use("Agg")  # 창 없이 저장만
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.scatter(xyz[:, 0], xyz[:, 1], s=0.2, c=xyz[:, 2], cmap="viridis", alpha=0.6)
        plt.xlabel("X"); plt.ylabel("Y"); plt.title(title + " (XY)")
        plt.axis("equal")
        plt.tight_layout()
        png_path = os.path.join(SCRIPT_DIR, "pcd_preview_xy.png")
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"2D 미리보기 저장됨: {png_path}")
    except Exception as e:
        print(f"저장 실패: {e}")


def main():
    pcd_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PCD
    if not os.path.isfile(pcd_path):
        print(f"파일을 찾을 수 없습니다: {pcd_path}")
        sys.exit(1)

    xyz = read_pcd_simple(pcd_path)
    if xyz is None:
        xyz = read_pcd_fallback(pcd_path)
    if xyz is None or len(xyz) == 0:
        print("포인트를 읽을 수 없습니다. PCD가 ASCII 형식인지 확인해 주세요.")
        sys.exit(1)
    print(f"포인트 수: {len(xyz)}")
    view_with_matplotlib(xyz, os.path.basename(pcd_path))


if __name__ == "__main__":
    main()
