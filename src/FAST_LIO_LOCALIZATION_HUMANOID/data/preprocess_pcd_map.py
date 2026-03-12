#!/usr/bin/env python3
"""
PCD 맵 전처리: localization용 global map 생성 (Open3D 미사용)
- 아웃라이어 제거 (Statistical, 1회) + (선택) XY 평면 2D 투영
- 결과를 HTML로 미리보기 (plotly)

사용법:
  python3 preprocess_pcd_map.py [입력.pcd] [출력.pcd]
  python3 preprocess_pcd_map.py [입력.pcd]              # 출력: 입력_global.pcd
  python3 preprocess_pcd_map.py                          # 기본: ./test.pcd -> ./test_global.pcd

  옵션: --no-outlier (아웃라이어 제거 생략), --project-2d (2D 투영), --z 0 (투영 시 z값), --no-html (HTML 미리보기 생략)
  생성된 PCD는 open3d_loc의 path_map(launch 파일)에 지정하여 global map으로 사용하면 됨.
"""
import sys
import os
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, "test.pcd")
DEFAULT_PREVIEW_HTML = "global_map_preview.html"

# 축 범위: None = 자동, [최소, 최대] = 고정 (예: [-10, 10])
AXIS_X_RANGE = [-10.0, 10.0]
AXIS_Y_RANGE = [-10.0, 10.0]
AXIS_Z_RANGE = [-10.0, 0.5]


def _read_pcd_numpy(path):
    """PCD에서 x,y,z만 읽기."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header.append(line)
            if line.startswith("DATA"):
                break
        fields_line = [l for l in header if l.startswith("FIELDS ")][0]
        fields = fields_line.split()[1:]
        try:
            x_idx = fields.index("x")
            y_idx = fields.index("y")
            z_idx = fields.index("z")
        except (ValueError, IndexError):
            x_idx, y_idx, z_idx = 0, 1, 2
        num_fields = len(fields)
        points_line = [l for l in header if l.startswith("POINTS ")][0]
        n_points = int(points_line.split()[1])
        if "DATA ascii" in " ".join(header):
            data = []
            for _ in range(n_points):
                line = f.readline().decode("ascii")
                parts = line.split()
                if len(parts) >= 3:
                    data.append([float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])])
            return np.array(data) if data else None
        size_line = [l for l in header if l.startswith("SIZE ")][0]
        sizes = list(map(int, size_line.split()[1:]))
        type_line = [l for l in header if l.startswith("TYPE ")][0]
        types = type_line.split()[1:]
        dtype_map = {"F": np.float32, "I": np.int32, "U": np.uint8}
        row_size = sum(sizes)
        buf = f.read(n_points * row_size)
        offsets = [sum(sizes[:k]) for k in range(num_fields)]
        xs = [np.frombuffer(buf[i * row_size + offsets[x_idx]:i * row_size + offsets[x_idx] + sizes[x_idx]], dtype=dtype_map.get(types[x_idx], np.float32))[0] for i in range(n_points)]
        ys = [np.frombuffer(buf[i * row_size + offsets[y_idx]:i * row_size + offsets[y_idx] + sizes[y_idx]], dtype=dtype_map.get(types[y_idx], np.float32))[0] for i in range(n_points)]
        zs = [np.frombuffer(buf[i * row_size + offsets[z_idx]:i * row_size + offsets[z_idx] + sizes[z_idx]], dtype=dtype_map.get(types[z_idx], np.float32))[0] for i in range(n_points)]
        return np.column_stack([xs, ys, zs])
    return None


def _outlier_remove_statistical_pure_numpy(xyz, k=20, std_ratio=2.0, batch_size=2000):
    """거리 기반 Statistical Outlier Removal (numpy만 사용).
    각 점에서 k개 최근접 이웃까지 평균 거리를 구하고,
    median + std_ratio*std 보다 큰 점을 아웃라이어로 제거.
    """
    n = len(xyz)
    k_use = min(k, n - 1)
    if k_use < 1:
        return xyz
    mean_d = np.zeros(n, dtype=np.float64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # (end-start, n) 거리
        D_batch = np.sqrt(((xyz[start:end, None, :] - xyz[None, :, :]) ** 2).sum(axis=2))
        for i in range(end - start):
            d = D_batch[i].copy()
            d[start + i] = np.inf
            part = np.partition(d, k_use)[: k_use + 1]
            part.sort()
            mean_d[start + i] = part[1 : k_use + 1].mean()
    thresh = np.median(mean_d) + std_ratio * (np.std(mean_d) + 1e-9)
    return xyz[mean_d < thresh]


def _outlier_remove_numpy(xyz, k=20, std_ratio=2.0):
    """거리 기반 Statistical Outlier Removal (순수 numpy만 사용, scipy 미사용)."""
    return _outlier_remove_statistical_pure_numpy(xyz, k=k, std_ratio=std_ratio)


def _write_pcd_ascii(xyz, path):
    """numpy (N,3)를 ASCII PCD로 저장."""
    with open(path, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {len(xyz)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(xyz)}\n")
        f.write("DATA ascii\n")
        for row in xyz:
            f.write(f"{row[0]} {row[1]} {row[2]}\n")


def _write_preview_html(xyz, html_path, title="Global map (3D)"):
    """plotly로 3D scatter HTML 저장 (view_pcd.py와 동일 방식)."""
    n = len(xyz)
    if n > 50000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=50000, replace=False)
        xyz_show = xyz[idx]
    else:
        xyz_show = xyz
    try:
        import plotly.express as px
        import pandas as pd
        df = pd.DataFrame(xyz_show, columns=["x", "y", "z"])
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="z", title=title)
        fig.update_traces(marker=dict(size=1.0, opacity=0.7))
        scene = {}
        if AXIS_X_RANGE is not None:
            scene["xaxis"] = dict(range=AXIS_X_RANGE)
        if AXIS_Y_RANGE is not None:
            scene["yaxis"] = dict(range=AXIS_Y_RANGE)
        if AXIS_Z_RANGE is not None:
            scene["zaxis"] = dict(range=AXIS_Z_RANGE)
        if scene:
            fig.update_layout(scene=scene)
        fig.write_html(html_path)
        return True
    except ImportError:
        return False
    except Exception:
        return False


def run_preprocess(input_path, output_path, nb_neighbors, std_ratio, project_2d, z_value, skip_outlier=False):
    """numpy만 사용한 전처리: outlier 제거(1회) → (선택) 2D 투영 → PCD 저장."""
    xyz = _read_pcd_numpy(input_path)
    if xyz is None or len(xyz) == 0:
        return False, "포인트를 읽을 수 없습니다."
    n_before = len(xyz)
    if not skip_outlier:
        xyz = _outlier_remove_numpy(xyz, k=nb_neighbors, std_ratio=std_ratio)
    n_after = len(xyz)
    if project_2d:
        xyz = xyz.copy()
        xyz[:, 2] = z_value
    _write_pcd_ascii(xyz, output_path)
    return True, (n_before, n_after, xyz)


def main():
    parser = argparse.ArgumentParser(description="PCD 맵 전처리: outlier 제거 (1회, 3D 그대로 / Open3D 미사용)")
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT, help="입력 PCD 경로")
    parser.add_argument("output", nargs="?", default=None, help="출력 PCD 경로 (미지정 시 입력_global.pcd)")
    parser.add_argument("--no-outlier", action="store_true", help="아웃라이어 제거 생략")
    parser.add_argument("--nb-neighbors", type=int, default=20, help="아웃라이어 이웃 수 (기본 20)")
    parser.add_argument("--std-ratio", type=float, default=2.0, help="아웃라이어 표준편차 배수 (기본 2.0)")
    parser.add_argument("--project-2d", action="store_true", help="XY 평면으로 2D 투영 (z를 고정)")
    parser.add_argument("--z", type=float, default=0.0, help="2D 투영 시 z 값 (기본 0)")
    parser.add_argument("--no-html", action="store_true", help="HTML 미리보기 생성 생략")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.abspath(input_path)
    if not os.path.isfile(input_path):
        print(f"파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    if args.output is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_global" + ext
    else:
        output_path = args.output

    ok, result = run_preprocess(
        input_path,
        output_path,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
        project_2d=args.project_2d,
        z_value=args.z,
        skip_outlier=args.no_outlier,
    )
    if not ok:
        print(result)
        sys.exit(1)
    n_before, n_after, xyz = result
    print(f"완료: {output_path}")
    if args.project_2d:
        print(f"  원본: {n_before} → outlier 제거: {n_after} → 2D 투영(z={args.z}): {n_after}")
    else:
        print(f"  원본: {n_before} → outlier 제거: {n_after}")

    if not args.no_html:
        html_path = os.path.join(SCRIPT_DIR, DEFAULT_PREVIEW_HTML)
        title_suffix = " (XY 2D)" if args.project_2d else " (3D)"
        if _write_preview_html(xyz, html_path, title=os.path.basename(output_path) + title_suffix):
            print(f"미리보기: {html_path}")
            import webbrowser
            webbrowser.open("file://" + os.path.abspath(html_path))
        else:
            print("HTML 미리보기 실패 (plotly/pandas 필요: pip install plotly pandas)")


if __name__ == "__main__":
    main()
